#!/usr/bin/env python3
"""
Comprehensive Stress Test - Step 2: Method Comparison
Compare scheduling methods using optimal configurations from Step 1.

Methods:
1. static_fifo: 1 GPU, no bins, fixed batch size (local baseline)
2. dynamic_no_bins: 1 GPU, SLA-aware dynamic batching (local optimized)
3. multi_bin_dynamic (1 GPU): Multi-bin + dynamic batching (local advanced)
4. multi_bin_dynamic (optimal): Multi-bin + dynamic with optimal GPU/bin config
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

# Output paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'stress_test_final')
STEP1_RESULTS = os.path.join(RESULTS_DIR, 'step1_grid_search.csv')
STEP2_RESULTS = os.path.join(RESULTS_DIR, 'step2_comparison.csv')

# Data paths
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# SLA Thresholds (v4 model with TTFT/TBT separation)
D_SLA_TOKEN = 0.010      # 10ms for decode TBT only
D_SLA_REQUEST = 20.0     # 20 seconds for total request latency
RPS_SCALING = 100.0      # Same as Step 1

# Workload sizes
REQUEST_COUNTS = [1000, 10000, 100000, 1000000]


def find_optimal_config(step1_df, num_requests, sla_threshold=5.0):
    """
    Find optimal GPU/bin configuration for a given workload size.
    
    Criteria:
    1. Request SLA violation rate <= sla_threshold%
    2. Minimize GPU count (cost efficiency)
    3. Among same GPU count, pick best bins
    """
    subset = step1_df[step1_df['num_requests'] == num_requests].copy()
    
    if subset.empty:
        return None, None, "No data for this workload size"
    
    # Filter by SLA threshold
    good_configs = subset[subset['request_sla_pct'] <= sla_threshold]
    
    if good_configs.empty:
        # If no config meets threshold, pick best available
        best = subset.loc[subset['request_sla_pct'].idxmin()]
        return int(best['num_gpus']), int(best['k_bins']), f"Best available (SLA={best['request_sla_pct']:.1f}%)"
    
    # Find minimum GPU count that meets SLA
    min_gpus = good_configs['num_gpus'].min()
    min_gpu_configs = good_configs[good_configs['num_gpus'] == min_gpus]
    
    # Among configs with min GPUs, pick lowest SLA violation
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
        # SLA metrics
        'token_sla_pct': round(metrics.get('sla_violation_rate_token', 0) * 100, 3),
        'request_sla_pct': round(metrics.get('sla_violation_rate_request', 0) * 100, 3),
        # Latency metrics
        'avg_ttft_ms': round(metrics.get('avg_ttft', 0) * 1000, 2),
        'avg_decode_tbt_ms': round(metrics.get('avg_decode_tbt', 0) * 1000, 2),
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


def main():
    """Run Step 2 comparison."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("STEP 2: Method Comparison")
    print("=" * 70)
    print(f"RPS Scaling: {RPS_SCALING}x (~{0.27 * RPS_SCALING:.1f} req/s)")
    print(f"D_SLA_TOKEN: {D_SLA_TOKEN * 1000}ms")
    print(f"D_SLA_REQUEST: {D_SLA_REQUEST}s")
    print()
    
    # Load Step 1 results for optimal config
    if os.path.exists(STEP1_RESULTS):
        step1_df = pd.read_csv(STEP1_RESULTS)
        print(f"Loaded Step 1 results: {len(step1_df)} configurations")
    else:
        print("WARNING: Step 1 results not found. Using default optimal configs.")
        step1_df = None
    
    print()
    
    results = []
    
    for num_requests in REQUEST_COUNTS:
        print(f"\n{'='*70}")
        print(f"Workload: {num_requests:,} requests")
        print(f"{'='*70}")
        
        # Find optimal config from Step 1
        if step1_df is not None:
            opt_gpus, opt_bins, opt_reason = find_optimal_config(step1_df, num_requests, sla_threshold=5.0)
            if opt_gpus is None:
                opt_gpus, opt_bins = 8, 8  # Default fallback
                opt_reason = "Default (no Step 1 data)"
        else:
            opt_gpus, opt_bins = 8, 8
            opt_reason = "Default (no Step 1 data)"
        
        print(f"Optimal config: GPUs={opt_gpus}, Bins={opt_bins} ({opt_reason})")
        print()
        
        # Define methods to compare
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
                
                token_sla = result['token_sla_pct']
                req_sla = result['request_sla_pct']
                throughput = result['throughput_req_s']
                avg_lat = result['avg_latency_s']
                
                print(f"TokenSLA={token_sla:.1f}%, ReqSLA={req_sla:.1f}%, "
                      f"Throughput={throughput:.2f} req/s, AvgLat={avg_lat:.1f}s")
                
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv(STEP2_RESULTS, index=False)
    
    print()
    print("=" * 70)
    print(f"STEP 2 COMPLETE! Results saved to: {STEP2_RESULTS}")
    print("=" * 70)
    
    # Print summary table
    if results:
        print("\n=== SUMMARY TABLE ===\n")
        df = pd.DataFrame(results)
        
        for num_requests in REQUEST_COUNTS:
            subset = df[df['num_requests'] == num_requests]
            if subset.empty:
                continue
            
            print(f"\nWorkload: {num_requests:,} requests")
            print("-" * 90)
            print(f"{'Method':<45} {'TokenSLA':>8} {'ReqSLA':>8} {'Throughput':>10} {'AvgLat':>8}")
            print("-" * 90)
            
            for _, row in subset.iterrows():
                print(f"{row['method']:<45} {row['token_sla_pct']:>7.1f}% {row['request_sla_pct']:>7.1f}% "
                      f"{row['throughput_req_s']:>9.2f} {row['avg_latency_s']:>7.1f}s")


if __name__ == "__main__":
    main()
