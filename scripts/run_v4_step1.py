#!/usr/bin/env python3
"""
V4 Stress Test Step 1: Grid search with TTFT/TBT separation model.
Runs one config at a time for robustness.
"""

import os
import sys
import time
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

# v4 SLA Model: TTFT/TBT Separation
D_SLA_TOKEN = 0.010      # 10ms for decode TBT only (not TTFT)
D_SLA_REQUEST = 20.0     # 20 seconds for total request latency
RPS_SCALING = 200.0      # Scale native BurstGPT RPS

# Grid search parameters
GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 100]
BIN_COUNTS = [1, 2, 4, 8, 16, 32]
REQUEST_COUNTS = [1000, 10000, 100000, 1000000]


def run_single_experiment(num_requests, num_gpus, k_bins):
    """Run a single experiment and return results."""
    start_time = time.time()
    
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
    completed_requests = simulator.run()
    
    metrics = compute_metrics(
        completed_requests,
        d_sla_token=D_SLA_TOKEN,
        d_sla_request=D_SLA_REQUEST
    )
    
    gpu_stats = simulator.get_gpu_stats()
    gpu_util = compute_gpu_utilization(gpu_stats)
    batch_stats = compute_batch_statistics(completed_requests)
    
    # Extract v4 metrics (TTFT/TBT separation)
    avg_ttft = metrics.get('avg_ttft', 0.0)
    avg_decode_tbt = metrics.get('avg_decode_tbt', 0.0)
    avg_tbt = metrics.get('avg_tbt', 0.0)  # Legacy total TBT
    
    execution_time = time.time() - start_time
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_requests': num_requests,
        'num_gpus': num_gpus,
        'k_bins': k_bins,
        'scheduler': "multi_bin_dynamic",
        'token_sla_pct': round(metrics.get('token_sla_pct', 0) * 100, 3),
        'request_sla_pct': round(metrics.get('request_sla_pct', 0) * 100, 3),
        'avg_ttft_ms': round(avg_ttft * 1000, 2),
        'avg_decode_tbt_ms': round(avg_decode_tbt * 1000, 2),
        'avg_tbt_ms': round(avg_tbt * 1000, 2),
        'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
        'gpu_utilization': round(gpu_util.get('avg_utilization', 0) * 100, 1),
        'throughput_tok_s': round(metrics.get('throughput_tokens_per_sec', 0), 1),
        'rps_scaling': RPS_SCALING,
        'actual_rps': round(metrics.get('throughput_requests_per_sec', 0), 1),
        'execution_time': round(execution_time, 1),
        'completed_requests': len(completed_requests),
        'p50_latency_ms': round(metrics.get('p50_latency', 0) * 1000, 2),
        'p99_latency_ms': round(metrics.get('p99_latency', 0) * 1000, 2),
        'avg_latency_ms': round(metrics.get('avg_latency', 0) * 1000, 2),
    }


def get_missing_configs():
    """Find all missing configurations."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(RESULTS_FILE):
        missing = []
        for n in REQUEST_COUNTS:
            for g in GPU_COUNTS:
                for k in BIN_COUNTS:
                    missing.append((n, g, k))
        return missing
    
    df = pd.read_csv(RESULTS_FILE)
    existing = set()
    for _, row in df.iterrows():
        existing.add((int(row['num_requests']), int(row['num_gpus']), int(row['k_bins'])))
    
    missing = []
    for n in REQUEST_COUNTS:
        for g in GPU_COUNTS:
            for k in BIN_COUNTS:
                if (n, g, k) not in existing:
                    missing.append((n, g, k))
    
    return missing


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
    return len(df_combined)


def main():
    print("=" * 70)
    print("V4 STRESS TEST - STEP 1: GRID SEARCH")
    print("TTFT/TBT Separation Model")
    print("=" * 70)
    print(f"D_SLA_TOKEN: {D_SLA_TOKEN*1000}ms (decode TBT only)")
    print(f"D_SLA_REQUEST: {D_SLA_REQUEST}s")
    print(f"RPS_SCALING: {RPS_SCALING}x")
    print()
    
    missing = get_missing_configs()
    total = len(missing)
    
    if total == 0:
        print("[OK] All 192 configurations complete!")
        return 0
    
    print(f"Missing configurations: {total}")
    print()
    
    for i, (num_req, num_gpus, k_bins) in enumerate(missing):
        print(f"[{i+1}/{total}] {num_req:,} req, {num_gpus} GPUs, K={k_bins}")
        
        try:
            result = run_single_experiment(num_req, num_gpus, k_bins)
            count = save_result(result)
            print(f"    Token={result['token_sla_pct']:.1f}% | Req={result['request_sla_pct']:.1f}% | "
                  f"Batch={result['avg_batch_size']:.1f} | Time={result['execution_time']:.1f}s | "
                  f"Total={count}/192")
        except Exception as e:
            print(f"    [FAILED] {e}")
            continue
    
    print()
    print("[OK] Step 1 complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
