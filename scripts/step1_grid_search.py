#!/usr/bin/env python3
"""
Comprehensive Stress Test - Step 1: Grid Search
Multi-bin + Dynamic batching across GPU counts, bin counts, and workload sizes.

This script uses RPS_SCALING=100x (~27 req/s) for meaningful differentiation.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics

# Output paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'stress_test_final')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'step1_grid_search.csv')

# Data paths
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# SLA Thresholds (v4 model with TTFT/TBT separation)
D_SLA_TOKEN = 0.010      # 10ms for decode TBT only
D_SLA_REQUEST = 20.0     # 20 seconds for total request latency
RPS_SCALING = 100.0      # 100x gives ~27 req/s, good differentiation

# Grid search parameters
GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 100]
BIN_COUNTS = [1, 2, 4, 8, 16, 32]
REQUEST_COUNTS = [1000, 10000, 100000, 1000000]
# Optional safety cap to skip extremely large configs (0 = no cap)
MAX_REQUESTS_CAP = int(os.environ.get("STEP1_MAX_REQUESTS", 0))


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
        LATENCY_EPSILON=0.001,
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
        'rps_scaling': RPS_SCALING,
        'd_sla_token_ms': D_SLA_TOKEN * 1000,
        'd_sla_request_s': D_SLA_REQUEST,
        # SLA metrics
        'token_sla_pct': round(metrics.get('sla_violation_rate_token', 0) * 100, 3),
        'request_sla_pct': round(metrics.get('sla_violation_rate_request', 0) * 100, 3),
        # Latency metrics
        'avg_ttft_ms': round(metrics.get('avg_ttft', 0) * 1000, 2),
        'avg_decode_tbt_ms': round(metrics.get('avg_decode_tbt', 0) * 1000, 2),
        'avg_tbt_ms': round(metrics.get('avg_tbt', 0) * 1000, 2),
        'avg_latency_s': round(metrics.get('avg_latency', 0), 2),
        'p50_latency_s': round(metrics.get('p50_latency', 0), 2),
        'p95_latency_s': round(metrics.get('p95_latency', 0), 2),
        'p99_latency_s': round(metrics.get('p99_latency', 0), 2),
        'avg_queue_delay_s': round(metrics.get('avg_queueing_delay', 0), 2),
        'avg_service_time_s': round(metrics.get('avg_service_time', 0), 2),
        # Throughput metrics
        'throughput_tok_s': round(metrics.get('throughput_tokens_per_sec', 0), 1),
        'throughput_req_s': round(metrics.get('throughput_requests_per_sec', 0), 3),
        # Batch and GPU metrics
        'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
        'gpu_utilization_pct': round(gpu_util.get('avg_utilization', 0) * 100, 1),
        # Execution info
        'execution_time_s': round(execution_time, 1),
        'completed_requests': len(completed_requests),
    }


def get_completed_configs():
    """Get set of already completed (num_requests, num_gpus, k_bins) tuples."""
    if not os.path.exists(RESULTS_FILE):
        return set()
    try:
        df = pd.read_csv(RESULTS_FILE)
        return set(zip(df['num_requests'], df['num_gpus'], df['k_bins']))
    except Exception:
        return set()


def save_result(result):
    """Append a single result to the CSV file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df_new = pd.DataFrame([result])
    df_new.to_csv(
        RESULTS_FILE,
        mode='a' if os.path.exists(RESULTS_FILE) else 'w',
        header=not os.path.exists(RESULTS_FILE),
        index=False,
    )


def main():
    """Run grid search, resumable from interrupts."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("STEP 1: Grid Search - Multi-bin + Dynamic Batching")
    print("=" * 70)
    print(f"RPS Scaling: {RPS_SCALING}x (~{0.27 * RPS_SCALING:.1f} req/s)")
    print(f"D_SLA_TOKEN: {D_SLA_TOKEN * 1000}ms")
    print(f"D_SLA_REQUEST: {D_SLA_REQUEST}s")
    print(f"Results: {RESULTS_FILE}")
    print()
    
    # Get what we've already done
    completed = get_completed_configs()
    print(f"Already completed: {len(completed)} configurations")
    
    total_configs = sum(
        1 for n in REQUEST_COUNTS
        for g in GPU_COUNTS
        for k in BIN_COUNTS
        if not MAX_REQUESTS_CAP or n <= MAX_REQUESTS_CAP
    )
    remaining_count = sum(
        1 for n in REQUEST_COUNTS
        for g in GPU_COUNTS
        for k in BIN_COUNTS
        if (not MAX_REQUESTS_CAP or n <= MAX_REQUESTS_CAP) and (n, g, k) not in completed
    )
    print(f"Total configurations: {total_configs}")
    print(f"Remaining: {remaining_count}")
    print()
    
    count = 0
    for num_requests in REQUEST_COUNTS:
        for num_gpus in GPU_COUNTS:
            for k_bins in BIN_COUNTS:
                if MAX_REQUESTS_CAP and num_requests > MAX_REQUESTS_CAP:
                    print(f"[skip] requests={num_requests}, cap={MAX_REQUESTS_CAP} (set STEP1_MAX_REQUESTS=0 to run)")
                    continue
                key = (num_requests, num_gpus, k_bins)
                if key in completed:
                    continue
                
                count += 1
                print(f"[{count}/{remaining_count}] requests={num_requests:>7}, GPUs={num_gpus:>3}, bins={k_bins:>2}", end=" ... ", flush=True)
                
                try:
                    result = run_single_experiment(num_requests, num_gpus, k_bins)
                    save_result(result)
                    
                    token_sla = result['token_sla_pct']
                    req_sla = result['request_sla_pct']
                    throughput = result['throughput_req_s']
                    exec_time = result['execution_time_s']
                    
                    print(f"TokenSLA={token_sla:.1f}%, ReqSLA={req_sla:.1f}%, "
                          f"Throughput={throughput:.2f} req/s, Time={exec_time:.0f}s")
                    
                except KeyboardInterrupt:
                    print("\nInterrupted by user. Progress saved.")
                    return
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
    
    print()
    print("=" * 70)
    print(f"STEP 1 COMPLETE! Results saved to: {RESULTS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
