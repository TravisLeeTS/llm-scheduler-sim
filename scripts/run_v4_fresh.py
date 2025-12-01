#!/usr/bin/env python3
"""
V4 Fresh Run: Complete grid search with corrected SLA controller.
The bug fix ensures decode_tbt (not legacy TBT) is fed to controller.
"""

import os
import sys
import time
import shutil
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'stress_test_v4_results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'step1_grid_search_v4fixed.csv')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# v4 SLA Model: TTFT/TBT Separation with corrected controller feedback
D_SLA_TOKEN = 0.010      # 10ms for decode TBT only (not TTFT)
D_SLA_REQUEST = 20.0     # 20 seconds for total request latency
RPS_SCALING = 200.0      # Scale native BurstGPT RPS (~0.27 -> ~54 req/s)

# Grid search parameters - smaller for faster execution
GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 100]
BIN_COUNTS = [1, 2, 4, 8, 16, 32]


def run_single_experiment(num_requests, num_gpus, k_bins):
    """Run a single experiment and return results dict."""
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
    
    execution_time = time.time() - start_time
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_requests': num_requests,
        'num_gpus': num_gpus,
        'k_bins': k_bins,
        'scheduler': "multi_bin_dynamic",
        'sla_violation_rate_token': round(metrics.get('sla_violation_rate_token', 0) * 100, 3),
        'sla_violation_rate_request': round(metrics.get('sla_violation_rate_request', 0) * 100, 3),
        'avg_ttft_ms': round(metrics.get('avg_ttft', 0) * 1000, 2),
        'avg_decode_tbt_ms': round(metrics.get('avg_decode_tbt', 0) * 1000, 2),
        'avg_tbt_ms': round(metrics.get('avg_tbt', 0) * 1000, 2),
        'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
        'gpu_utilization': round(gpu_util.get('avg_utilization', 0) * 100, 1),
        'throughput_tok_s': round(metrics.get('throughput_tokens_per_sec', 0), 1),
        'throughput_req_s': round(metrics.get('throughput_requests_per_sec', 0), 2),
        'avg_latency_s': round(metrics.get('avg_latency', 0), 2),
        'rps_scaling': RPS_SCALING,
        'execution_time_s': round(execution_time, 1),
        'completed_requests': len(completed_requests),
        'd_sla_token_ms': D_SLA_TOKEN * 1000,
        'd_sla_request_s': D_SLA_REQUEST,
    }


def get_completed_configs():
    """Get set of already completed (num_requests, num_gpus, k_bins) tuples."""
    if not os.path.exists(RESULTS_FILE):
        return set()
    df = pd.read_csv(RESULTS_FILE)
    return set(zip(df['num_requests'], df['num_gpus'], df['k_bins']))


def save_result(result):
    """Append a single result to the CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df_new = pd.DataFrame([result])
    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_csv(RESULTS_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(RESULTS_FILE, index=False)


def main():
    """Run grid search, resumable from interrupts."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Get what we've already done
    completed = get_completed_configs()
    print(f"Already completed: {len(completed)} configurations")
    
    # Only run small and medium workloads first for quick validation
    request_counts = [1000, 10000]  # Start with smaller workloads
    
    total_configs = len(request_counts) * len(GPU_COUNTS) * len(BIN_COUNTS)
    remaining = total_configs - len(completed)
    print(f"Total configurations: {total_configs}")
    print(f"Remaining: {remaining}")
    print()
    
    count = 0
    for num_requests in request_counts:
        for num_gpus in GPU_COUNTS:
            for k_bins in BIN_COUNTS:
                key = (num_requests, num_gpus, k_bins)
                if key in completed:
                    continue
                
                count += 1
                print(f"[{count}/{remaining}] Running: requests={num_requests}, GPUs={num_gpus}, bins={k_bins}")
                
                try:
                    result = run_single_experiment(num_requests, num_gpus, k_bins)
                    save_result(result)
                    
                    token_sla = result['sla_violation_rate_token']
                    req_sla = result['sla_violation_rate_request']
                    throughput = result['throughput_req_s']
                    exec_time = result['execution_time_s']
                    
                    print(f"  -> TokenSLA={token_sla:.1f}%, ReqSLA={req_sla:.1f}%, "
                          f"Throughput={throughput:.1f} req/s, Time={exec_time:.1f}s")
                    
                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    continue
    
    print(f"\nCompleted! Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
