#!/usr/bin/env python3
"""
Run a single 1M configuration with robust error handling.
Usage: python run_one_1m_config.py <num_gpus> <k_bins>
"""

import os
import sys
import time
import signal
from datetime import datetime
import pandas as pd
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'stress_test_v3_results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'step1_grid_search.csv')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

D_SLA_TOKEN = 0.030
D_SLA_REQUEST = 20.0
RPS_SCALING = 200.0
NUM_REQUESTS = 1_000_000


def run_experiment(num_gpus, k_bins):
    """Run a single 1M experiment."""
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Running: 1M requests, GPUs={num_gpus}, K={k_bins}")
    print(f"{'='*60}")
    
    cfg = SchedulerConfig(
        NUM_GPUS=num_gpus,
        K_BINS=k_bins,
        NUM_REQUESTS=NUM_REQUESTS,
        EXPERIMENT_MODE="multi_bin_dynamic",
        B_MIN=1,
        B_MAX=128,
        D_SLA=D_SLA_TOKEN,
        D_SLA_TOKEN=D_SLA_TOKEN,
        D_SLA_REQUEST=D_SLA_REQUEST,
        LATENCY_EPSILON=0.010,
        USE_REAL_CALIBRATION=True,
        CALIBRATION_CSV_PATH=CALIBRATION_CSV,
        WORKLOAD_SOURCE="burstgpt_dataset",
        DATASET_PATH=DATASET_PATH,
        USE_REAL_TIMESTAMPS=False,
        RPS_SCALING=RPS_SCALING,
        SEED=42,
    )
    
    print("Loading workload...")
    requests = generate_workload(cfg)
    print(f"Loaded {len(requests)} requests")
    
    print("Running simulation...")
    simulator = Simulator(cfg, requests, "multi_bin_dynamic")
    completed_requests = simulator.run()
    
    print("Computing metrics...")
    metrics = compute_metrics(
        completed_requests, 
        d_sla_token=D_SLA_TOKEN,
        d_sla_request=D_SLA_REQUEST
    )
    
    gpu_stats = simulator.get_gpu_stats()
    gpu_util = compute_gpu_utilization(gpu_stats)
    batch_stats = compute_batch_statistics(completed_requests)
    
    execution_time = time.time() - start_time
    actual_rps = NUM_REQUESTS / metrics.get('total_time', 1) if metrics.get('total_time', 0) > 0 else 0
    
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_requests': NUM_REQUESTS,
        'num_gpus': num_gpus,
        'k_bins': k_bins,
        'scheduler': "multi_bin_dynamic",
        'token_sla_pct': round(metrics.get('token_sla_pct', 0), 3),
        'request_sla_pct': round(metrics.get('request_sla_pct', 0), 3),
        'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
        'gpu_utilization': round(gpu_util * 100, 1),
        'rps_scaling': RPS_SCALING,
        'actual_rps': round(actual_rps, 1),
        'execution_time': round(execution_time, 1),
        'completed_requests': len(completed_requests),
        'avg_latency_ms': round(metrics.get('avg_latency', 0) * 1000, 2),
        'p50_latency_ms': round(metrics.get('p50_latency', 0) * 1000, 2),
        'p99_latency_ms': round(metrics.get('p99_latency', 0) * 1000, 2),
    }
    
    return result


def save_result(result):
    """Append result to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df_new = pd.DataFrame([result])
    
    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_csv(RESULTS_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(RESULTS_FILE, index=False)
    print(f"Saved result. Total rows: {len(df_combined)}")


def get_next_missing():
    """Find the next missing 1M configuration."""
    gpu_counts = [1, 2, 4, 8, 16, 32, 64, 100]
    bin_counts = [1, 2, 4, 8, 16, 32]
    
    if not os.path.exists(RESULTS_FILE):
        return gpu_counts[0], bin_counts[0]
    
    df = pd.read_csv(RESULTS_FILE)
    existing_1m = df[df['num_requests'] == NUM_REQUESTS][['num_gpus', 'k_bins']].values.tolist()
    existing_set = set([(int(r[0]), int(r[1])) for r in existing_1m])
    
    for g in gpu_counts:
        for k in bin_counts:
            if (g, k) not in existing_set:
                return g, k
    
    return None, None


def main():
    if len(sys.argv) == 3:
        # Explicit args provided
        num_gpus = int(sys.argv[1])
        k_bins = int(sys.argv[2])
    else:
        # Find next missing config
        num_gpus, k_bins = get_next_missing()
        if num_gpus is None:
            print("All 1M configurations complete!")
            return
    
    # Check if already exists
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        exists = len(df[(df['num_requests'] == NUM_REQUESTS) & 
                       (df['num_gpus'] == num_gpus) & 
                       (df['k_bins'] == k_bins)]) > 0
        if exists:
            print(f"Config already exists: GPUs={num_gpus}, K={k_bins}")
            return
    
    try:
        result = run_experiment(num_gpus, k_bins)
        save_result(result)
        print(f"\n✓ SUCCESS: Token={result['token_sla_pct']:.1f}%, Req={result['request_sla_pct']:.1f}%, Time={result['execution_time']:.1f}s")
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
