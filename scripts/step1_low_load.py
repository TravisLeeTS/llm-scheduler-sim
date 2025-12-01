#!/usr/bin/env python3
"""
Step 1 Grid Search - LOW LOAD VERSION
RPS_SCALING = 10 (instead of 100) for more reasonable baseline performance.
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
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'stress_test_low_load')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'step1_grid_search.csv')

# Data paths
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# SLA Thresholds
D_SLA_TOKEN = 0.010      # 10ms for decode TBT only
D_SLA_REQUEST = 20.0     # 20 seconds for total request latency

# LOW LOAD: 10x scaling (~2.7 req/s instead of ~27 req/s)
RPS_SCALING = 10.0

# Grid search parameters
REQUEST_COUNTS = [1000, 10000, 100000, 1000000]
GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 100]
BIN_COUNTS = [1, 2, 4, 8, 16, 32]


def run_single_experiment(num_requests, num_gpus, k_bins):
    """Run a single experiment configuration."""
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
        'rps_scaling': RPS_SCALING,
        'd_sla_token_ms': D_SLA_TOKEN * 1000,
        'd_sla_request_s': D_SLA_REQUEST,
        'token_sla_pct': round(metrics.get('sla_violation_rate_token', 0) * 100, 3),
        'request_sla_pct': round(metrics.get('sla_violation_rate_request', 0) * 100, 3),
        'avg_ttft_ms': round(metrics.get('avg_ttft', 0) * 1000, 2),
        'avg_decode_tbt_ms': round(metrics.get('avg_decode_tbt', 0) * 1000, 2),
        'avg_latency_s': round(metrics.get('avg_latency', 0), 2),
        'p50_latency_s': round(metrics.get('p50_latency', 0), 2),
        'p95_latency_s': round(metrics.get('p95_latency', 0), 2),
        'p99_latency_s': round(metrics.get('p99_latency', 0), 2),
        'avg_queue_delay_s': round(metrics.get('avg_queueing_delay', 0), 2),
        'avg_service_time_s': round(metrics.get('avg_service_time', 0), 2),
        'throughput_tok_s': round(metrics.get('throughput_tokens_per_sec', 0), 1),
        'throughput_req_s': round(metrics.get('throughput_requests_per_sec', 0), 3),
        'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
        'gpu_utilization_pct': round(gpu_util.get('avg_utilization', 0) * 100, 1),
        'execution_time_s': round(execution_time, 1),
        'completed_requests': len(completed_requests),
    }


def main():
    """Run Step 1 grid search with resume support."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("STEP 1: Grid Search (LOW LOAD)")
    print("=" * 70)
    print(f"RPS Scaling: {RPS_SCALING}x (~{0.27 * RPS_SCALING:.1f} req/s)")
    print(f"D_SLA_TOKEN: {D_SLA_TOKEN * 1000}ms")
    print(f"D_SLA_REQUEST: {D_SLA_REQUEST}s")
    print()
    
    # Load existing results for resume
    existing = set()
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        for _, row in df.iterrows():
            existing.add((int(row['num_requests']), int(row['num_gpus']), int(row['k_bins'])))
        print(f"Resuming: {len(existing)} configs already completed")
    
    # Build list of configs to run
    configs = []
    for n in REQUEST_COUNTS:
        for g in GPU_COUNTS:
            for k in BIN_COUNTS:
                if (n, g, k) not in existing:
                    configs.append((n, g, k))
    
    total = len(configs)
    print(f"Remaining: {total} configs to run")
    print()
    
    if total == 0:
        print("All configs complete!")
        return
    
    for i, (n, g, k) in enumerate(configs, 1):
        print(f"[{i}/{total}] requests={n:>7}, GPUs={g:>3}, bins={k:>2}", end=" ... ", flush=True)
        
        try:
            result = run_single_experiment(n, g, k)
            
            # Append to CSV
            df_new = pd.DataFrame([result])
            if os.path.exists(RESULTS_FILE):
                df_new.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
            else:
                df_new.to_csv(RESULTS_FILE, index=False)
            
            print(f"TokenSLA={result['token_sla_pct']:.1f}%, "
                  f"ReqSLA={result['request_sla_pct']:.1f}%, "
                  f"Throughput={result['throughput_req_s']:.2f} req/s")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    print()
    print("=" * 70)
    print(f"STEP 1 COMPLETE! Results saved to: {RESULTS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
