#!/usr/bin/env python3
"""
Step 2 Method Comparison - LOW LOAD VERSION
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
STEP1_RESULTS = os.path.join(RESULTS_DIR, 'step1_grid_search.csv')
STEP2_RESULTS = os.path.join(RESULTS_DIR, 'step2_comparison.csv')

# Data paths
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# SLA Thresholds
D_SLA_TOKEN = 0.010      # 10ms for decode TBT only
D_SLA_REQUEST = 20.0     # 20 seconds for total request latency

# LOW LOAD: 10x scaling (~2.7 req/s instead of ~27 req/s)
RPS_SCALING = 10.0

# Workload sizes
REQUEST_COUNTS = [1000, 10000, 100000, 1000000]


def find_optimal_config(step1_df, num_requests, sla_threshold=5.0):
    """Find optimal GPU/bin configuration for a given workload size."""
    subset = step1_df[step1_df['num_requests'] == num_requests].copy()
    
    if subset.empty:
        return None, None, "No data"
    
    good_configs = subset[subset['request_sla_pct'] <= sla_threshold]
    
    if good_configs.empty:
        best = subset.loc[subset['request_sla_pct'].idxmin()]
        return int(best['num_gpus']), int(best['k_bins']), f"Best available (SLA={best['request_sla_pct']:.1f}%)"
    
    min_gpus = good_configs['num_gpus'].min()
    min_gpu_configs = good_configs[good_configs['num_gpus'] == min_gpus]
    best = min_gpu_configs.loc[min_gpu_configs['request_sla_pct'].idxmin()]
    
    return int(best['num_gpus']), int(best['k_bins']), f"SLA={best['request_sla_pct']:.1f}%"


def run_experiment(num_requests, num_gpus, k_bins, scheduler_type, method_name):
    """Run a single experiment and return results dict."""
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
        'method': method_name,
        'scheduler_type': scheduler_type,
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
    """Run Step 2 comparison."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("STEP 2: Method Comparison (LOW LOAD)")
    print("=" * 70)
    print(f"RPS Scaling: {RPS_SCALING}x (~{0.27 * RPS_SCALING:.1f} req/s)")
    print(f"D_SLA_TOKEN: {D_SLA_TOKEN * 1000}ms")
    print(f"D_SLA_REQUEST: {D_SLA_REQUEST}s")
    print()
    
    # Load Step 1 results
    if os.path.exists(STEP1_RESULTS):
        step1_df = pd.read_csv(STEP1_RESULTS)
        print(f"Loaded Step 1 results: {len(step1_df)} configurations")
    else:
        print("WARNING: Step 1 results not found!")
        step1_df = None
    
    print()
    
    results = []
    
    for num_requests in REQUEST_COUNTS:
        print(f"\n{'='*70}")
        print(f"Workload: {num_requests:,} requests")
        print(f"{'='*70}")
        
        # Find optimal config
        if step1_df is not None:
            opt_gpus, opt_bins, opt_reason = find_optimal_config(step1_df, num_requests, sla_threshold=5.0)
            if opt_gpus is None:
                opt_gpus, opt_bins = 8, 8
                opt_reason = "Default"
        else:
            opt_gpus, opt_bins = 8, 8
            opt_reason = "Default"
        
        print(f"Optimal config: GPUs={opt_gpus}, Bins={opt_bins} ({opt_reason})")
        print()
        
        methods = [
            ("1. Static FIFO (1 GPU)", "static_fifo", 1, 1),
            ("2. Dynamic No-Bins (1 GPU)", "dynamic_no_bins", 1, 1),
            ("3. Multi-Bin Dynamic (1 GPU)", "multi_bin_dynamic", 1, 8),
            (f"4. Multi-Bin Dynamic (Optimal: {opt_gpus} GPUs, {opt_bins} bins)", "multi_bin_dynamic", opt_gpus, opt_bins),
        ]
        
        for method_name, scheduler_type, gpus, bins in methods:
            print(f"  {method_name}...", end=" ", flush=True)
            
            try:
                result = run_experiment(num_requests, gpus, bins, scheduler_type, method_name)
                results.append(result)
                
                print(f"TokenSLA={result['token_sla_pct']:.1f}%, "
                      f"ReqSLA={result['request_sla_pct']:.1f}%, "
                      f"Throughput={result['throughput_req_s']:.2f} req/s, "
                      f"AvgLat={result['avg_latency_s']:.1f}s")
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        # Save intermediate
        df = pd.DataFrame(results)
        df.to_csv(STEP2_RESULTS, index=False)
    
    print()
    print("=" * 70)
    print(f"STEP 2 COMPLETE! Results saved to: {STEP2_RESULTS}")
    print("=" * 70)
    
    # Summary
    if results:
        print("\n=== SUMMARY TABLE ===\n")
        df = pd.DataFrame(results)
        
        for num_requests in REQUEST_COUNTS:
            subset = df[df['num_requests'] == num_requests]
            if subset.empty:
                continue
            
            print(f"\nWorkload: {num_requests:,} requests")
            print("-" * 100)
            print(f"{'Method':<50} {'TokenSLA':>8} {'ReqSLA':>8} {'AvgLat':>10} {'Throughput':>10}")
            print("-" * 100)
            
            for _, row in subset.iterrows():
                print(f"{row['method']:<50} {row['token_sla_pct']:>7.1f}% {row['request_sla_pct']:>7.1f}% "
                      f"{row['avg_latency_s']:>9.1f}s {row['throughput_req_s']:>9.2f}")


if __name__ == "__main__":
    main()
