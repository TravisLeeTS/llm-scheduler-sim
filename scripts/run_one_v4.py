#!/usr/bin/env python3
"""
Run stress test v4 one config at a time for robustness.
"""
import os
import sys
import time
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'stress_test_v4_results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'step1_grid_search.csv')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# v4 SLA: TTFT/TBT Separation
D_SLA_TOKEN = 0.010  # 10ms decode TBT only
D_SLA_REQUEST = 20.0
RPS_SCALING = 200.0

GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 100]
BIN_COUNTS = [1, 2, 4, 8, 16, 32]
REQUEST_COUNTS = [1000, 10000, 100000, 1000000]


def run_one_config(num_requests, num_gpus, k_bins):
    """Run a single config and return result dict."""
    start = time.time()
    
    cfg = SchedulerConfig(
        NUM_GPUS=num_gpus,
        K_BINS=k_bins,
        NUM_REQUESTS=num_requests,
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
    
    requests = generate_workload(cfg)
    simulator = Simulator(cfg, requests, "multi_bin_dynamic")
    completed = simulator.run()
    
    metrics = compute_metrics(completed, d_sla_token=D_SLA_TOKEN, d_sla_request=D_SLA_REQUEST)
    gpu_stats = simulator.get_gpu_stats()
    gpu_util = compute_gpu_utilization(gpu_stats)
    batch_stats = compute_batch_statistics(completed)
    
    # Extract GPU utilization percentage
    if isinstance(gpu_util, dict):
        gpu_util_pct = gpu_util.get('avg_utilization', 0) * 100
    else:
        gpu_util_pct = float(gpu_util) * 100
    
    exec_time = time.time() - start
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_requests': num_requests,
        'num_gpus': num_gpus,
        'k_bins': k_bins,
        'scheduler': 'multi_bin_dynamic',
        'token_sla_pct': round(metrics.get('token_sla_pct', 0), 3),
        'request_sla_pct': round(metrics.get('request_sla_pct', 0), 3),
        'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
        'gpu_utilization': round(gpu_util_pct, 1),
        'rps_scaling': RPS_SCALING,
        'execution_time': round(exec_time, 1),
        'completed_requests': len(completed),
        'avg_ttft_ms': round(metrics.get('avg_ttft', 0) * 1000, 2),
        'avg_decode_tbt_ms': round(metrics.get('avg_decode_tbt', 0) * 1000, 2),
        'avg_latency_ms': round(metrics.get('avg_latency', 0) * 1000, 2),
        'p50_latency_ms': round(metrics.get('p50_latency', 0) * 1000, 2),
        'p99_latency_ms': round(metrics.get('p99_latency', 0) * 1000, 2),
    }


def get_existing_configs():
    """Load existing results and return set of completed configs."""
    if not os.path.exists(RESULTS_FILE):
        return set()
    df = pd.read_csv(RESULTS_FILE)
    return set(zip(df['num_requests'], df['num_gpus'], df['k_bins']))


def save_result(result):
    """Append one result to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_new = pd.DataFrame([result])
    
    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_csv(RESULTS_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(RESULTS_FILE, index=False)
    return len(df_combined)


def main():
    existing = get_existing_configs()
    print(f"Existing configs: {len(existing)}/192")
    
    # Build list of missing configs
    missing = []
    for num_req in REQUEST_COUNTS:
        for num_gpus in GPU_COUNTS:
            for k_bins in BIN_COUNTS:
                if (num_req, num_gpus, k_bins) not in existing:
                    missing.append((num_req, num_gpus, k_bins))
    
    if not missing:
        print("All configs complete!")
        return
    
    print(f"Missing configs: {len(missing)}")
    
    # Run just ONE config
    num_req, num_gpus, k_bins = missing[0]
    print(f"\nRunning: {num_req:,} req, {num_gpus} GPUs, K={k_bins}")
    
    try:
        result = run_one_config(num_req, num_gpus, k_bins)
        count = save_result(result)
        print(f"OK: Token={result['token_sla_pct']}%, Req={result['request_sla_pct']}%, "
              f"TBT={result['avg_decode_tbt_ms']}ms, Time={result['execution_time']}s")
        print(f"Saved: {count}/192")
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
