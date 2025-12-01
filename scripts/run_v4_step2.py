#!/usr/bin/env python3
"""
V4 Stress Test Step 2: Method comparison with optimal configs from Step 1.
Compares: Static FIFO, Dynamic No-Bins, MultiBin 1GPU, MultiBin Optimal
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
STEP1_FILE = os.path.join(RESULTS_DIR, 'step1_grid_search.csv')
STEP2_FILE = os.path.join(RESULTS_DIR, 'step2_comparison.csv')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# v4 SLA Model
D_SLA_TOKEN = 0.010      # 10ms for decode TBT
D_SLA_REQUEST = 20.0     # 20 seconds
RPS_SCALING = 200.0

REQUEST_COUNTS = [1000, 10000, 100000, 1000000]


def find_optimal_config(df, num_requests):
    """Find optimal (GPU, K) config for given request count from Step 1 results."""
    subset = df[df['num_requests'] == num_requests].copy()
    
    if len(subset) == 0:
        return 8, 8  # Default fallback
    
    # Score: maximize request SLA compliance, then minimize GPU count
    # Higher request_sla_pct means fewer violations (we want high %)
    subset['score'] = (100 - subset['request_sla_pct']) * 1000 + subset['num_gpus']
    
    # Find best config (lowest score = fewest violations + fewer GPUs)
    best_idx = subset['score'].idxmin()
    best_row = subset.loc[best_idx]
    
    return int(best_row['num_gpus']), int(best_row['k_bins'])


def run_experiment(num_requests, num_gpus, k_bins, scheduler_type):
    """Run a single experiment."""
    start_time = time.time()
    
    cfg = SchedulerConfig(
        NUM_GPUS=num_gpus,
        K_BINS=k_bins,
        NUM_REQUESTS=num_requests,
        EXPERIMENT_MODE=scheduler_type,
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
    simulator = Simulator(cfg, requests, scheduler_type)
    completed_requests = simulator.run()
    
    metrics = compute_metrics(
        completed_requests,
        d_sla_token=D_SLA_TOKEN,
        d_sla_request=D_SLA_REQUEST
    )
    
    gpu_stats = simulator.get_gpu_stats()
    gpu_util = compute_gpu_utilization(gpu_stats)
    batch_stats = compute_batch_statistics(completed_requests)
    
    execution_time = time.time() - start_time
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_requests': num_requests,
        'method': scheduler_type,
        'num_gpus': num_gpus,
        'k_bins': k_bins,
        'token_sla_pct': round(metrics.get('token_sla_pct', 0) * 100, 3),
        'request_sla_pct': round(metrics.get('request_sla_pct', 0) * 100, 3),
        'avg_ttft_ms': round(metrics.get('avg_ttft', 0) * 1000, 2),
        'avg_decode_tbt_ms': round(metrics.get('avg_decode_tbt', 0) * 1000, 2),
        'avg_tbt_ms': round(metrics.get('avg_tbt', 0) * 1000, 2),
        'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
        'gpu_utilization': round(gpu_util.get('avg_utilization', 0) * 100, 1),
        'throughput_tok_s': round(metrics.get('throughput_tokens_per_sec', 0), 1),
        'actual_rps': round(metrics.get('throughput_requests_per_sec', 0), 1),
        'execution_time': round(execution_time, 1),
        'completed_requests': len(completed_requests),
        'p50_latency_ms': round(metrics.get('p50_latency', 0) * 1000, 2),
        'p99_latency_ms': round(metrics.get('p99_latency', 0) * 1000, 2),
    }


def main():
    print("=" * 70)
    print("V4 STRESS TEST - STEP 2: METHOD COMPARISON")
    print("=" * 70)
    
    # Load Step 1 results
    if not os.path.exists(STEP1_FILE):
        print("[ERROR] Step 1 results not found. Run Step 1 first.")
        return 1
    
    df_step1 = pd.read_csv(STEP1_FILE)
    print(f"Loaded {len(df_step1)} Step 1 results")
    print()
    
    results = []
    
    for num_req in REQUEST_COUNTS:
        print(f"\n{'='*60}")
        print(f"Workload: {num_req:,} requests")
        print(f"{'='*60}")
        
        # Find optimal config from Step 1
        opt_gpus, opt_k = find_optimal_config(df_step1, num_req)
        print(f"Optimal config: {opt_gpus} GPUs, K={opt_k}")
        print()
        
        # Method 1: Static FIFO (1 GPU, no bins)
        print("[1/4] Static FIFO (1 GPU, no batching)...")
        try:
            result = run_experiment(num_req, 1, 1, "static_fifo")
            result['method_name'] = 'Static_FIFO'
            results.append(result)
            print(f"      Token={result['token_sla_pct']:.1f}% | Req={result['request_sla_pct']:.1f}%")
        except Exception as e:
            print(f"      [FAILED] {e}")
        
        # Method 2: Dynamic No-Bins (1 GPU, K=1)
        print("[2/4] Dynamic No-Bins (1 GPU, K=1)...")
        try:
            result = run_experiment(num_req, 1, 1, "dynamic_no_bins")
            result['method_name'] = 'Dynamic_NoBins'
            results.append(result)
            print(f"      Token={result['token_sla_pct']:.1f}% | Req={result['request_sla_pct']:.1f}%")
        except Exception as e:
            print(f"      [FAILED] {e}")
        
        # Method 3: MultiBin Dynamic (1 GPU, K=8)
        print("[3/4] MultiBin Dynamic (1 GPU, K=8)...")
        try:
            result = run_experiment(num_req, 1, 8, "multi_bin_dynamic")
            result['method_name'] = 'MultiBin_1GPU'
            results.append(result)
            print(f"      Token={result['token_sla_pct']:.1f}% | Req={result['request_sla_pct']:.1f}%")
        except Exception as e:
            print(f"      [FAILED] {e}")
        
        # Method 4: MultiBin Dynamic (Optimal GPUs & K)
        print(f"[4/4] MultiBin Dynamic (Optimal: {opt_gpus} GPUs, K={opt_k})...")
        try:
            result = run_experiment(num_req, opt_gpus, opt_k, "multi_bin_dynamic")
            result['method_name'] = 'MultiBin_Optimal'
            results.append(result)
            print(f"      Token={result['token_sla_pct']:.1f}% | Req={result['request_sla_pct']:.1f}%")
        except Exception as e:
            print(f"      [FAILED] {e}")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(STEP2_FILE, index=False)
    print(f"\n[OK] Saved {len(results)} results to {STEP2_FILE}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("STEP 2 RESULTS SUMMARY")
    print("=" * 70)
    
    summary = df_results.pivot_table(
        index='num_requests',
        columns='method_name',
        values=['token_sla_pct', 'request_sla_pct', 'avg_batch_size'],
        aggfunc='first'
    )
    print(summary.to_string())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
