#!/usr/bin/env python3
"""
Stress Test v4: TTFT/TBT Separation Model

v4 SLA Model Changes:
- Token SLA applies ONLY to decode TBT (β ≈ 5.74ms/token), NOT TTFT (α ≈ 60ms)
- This eliminates structural violations where TTFT/output_len dominates
- D_SLA_TOKEN = 10ms gives ~38% headroom over β=5.74ms baseline
- D_SLA_TOKEN = 30ms gives ~4× headroom for stress testing

Step 1: Grid Search (multi_bin_dynamic)
- Workloads: 1K, 10K, 100K, 1M requests
- GPUs: 1, 2, 4, 8, 16, 32, 64, 100
- Bins: 1, 2, 4, 8, 16, 32

Step 2: Method Comparison
- Static FIFO (1 GPU, no bins, no dynamic batching)
- Dynamic No-Bins (1 GPU, single queue + dynamic batching)
- MultiBin Dynamic (1 GPU, fair comparison)
- MultiBin Dynamic (optimal GPU count from Step 1)
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
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# v4 SLA Configuration (TTFT/TBT Separation)
# Token SLA applies to decode TBT only (β ≈ 5.74ms baseline)
D_SLA_TOKEN = 0.010      # 10ms decode TBT SLA (strict, ~38% headroom)
D_SLA_REQUEST = 20.0     # 20s request latency SLA

# RPS scaling for stress testing
RPS_SCALING = 200.0      # 200x → ~54 req/s

# Grid search parameters
GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 100]
BIN_COUNTS = [1, 2, 4, 8, 16, 32]
REQUEST_COUNTS = [1_000, 10_000, 100_000, 1_000_000]


def run_single_experiment(scheduler_type, num_requests, num_gpus, k_bins, verbose=True):
    """Run a single experiment."""
    start_time = time.time()
    
    try:
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
            LATENCY_EPSILON=0.005,
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
        gpu_util_dict = compute_gpu_utilization(gpu_stats)
        if isinstance(gpu_util_dict, dict):
            gpu_util = float(gpu_util_dict.get('avg_utilization', 0))
        else:
            gpu_util = float(gpu_util_dict) if gpu_util_dict else 0.0
        batch_stats = compute_batch_statistics(completed_requests)
        
        execution_time = time.time() - start_time
        actual_rps = num_requests / metrics.get('total_time', 1) if metrics.get('total_time', 0) > 0 else 0
        
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'k_bins': k_bins,
            'scheduler': scheduler_type,
            # v4 SLA metrics (TTFT/TBT separation)
            'token_sla_pct': round(metrics.get('sla_violation_rate_token', 0) * 100, 3),
            'request_sla_pct': round(metrics.get('sla_violation_rate_request', 0) * 100, 3),
            # Separated latency components
            'avg_ttft_ms': round(metrics.get('avg_ttft', 0) * 1000, 2),
            'avg_decode_tbt_ms': round(metrics.get('avg_decode_tbt', 0) * 1000, 2),
            'avg_tbt_ms': round(metrics.get('avg_tbt', 0) * 1000, 2),  # Legacy
            # Throughput metrics
            'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
            'gpu_utilization': round(gpu_util * 100, 1),
            'throughput_tok_s': round(metrics.get('throughput_tokens_per_sec', 0), 1),
            'rps_scaling': RPS_SCALING,
            'actual_rps': round(actual_rps, 1),
            'execution_time': round(execution_time, 1),
            'completed_requests': len(completed_requests),
            # Latency percentiles
            'p50_latency_ms': round(metrics.get('p50_latency', 0) * 1000, 2),
            'p99_latency_ms': round(metrics.get('p99_latency', 0) * 1000, 2),
        }
        
        return result
        
    except Exception as e:
        import traceback
        print(f"    [ERROR] {e}")
        traceback.print_exc()
        return None


def save_results(results, filename):
    """Save results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)
    
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(results)} results to {filepath}")


def load_existing_results(filename):
    """Load existing results from CSV."""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath).to_dict('records')
    return []


def run_step1_grid_search():
    """Step 1: Grid search for optimal GPU/bin combinations."""
    print("=" * 70)
    print("STEP 1: GRID SEARCH (multi_bin_dynamic)")
    print("=" * 70)
    print(f"Token SLA: {D_SLA_TOKEN*1000}ms (decode TBT only, excludes TTFT)")
    print(f"Request SLA: {D_SLA_REQUEST}s")
    print(f"RPS Scaling: {RPS_SCALING}x")
    print()
    
    results = load_existing_results('step1_grid_search.csv')
    existing_set = set()
    for r in results:
        key = (r['num_requests'], r['num_gpus'], r['k_bins'])
        existing_set.add(key)
    
    print(f"Loaded {len(results)} existing results")
    
    total_configs = len(REQUEST_COUNTS) * len(GPU_COUNTS) * len(BIN_COUNTS)
    current = 0
    
    for num_requests in REQUEST_COUNTS:
        for num_gpus in GPU_COUNTS:
            for k_bins in BIN_COUNTS:
                current += 1
                key = (num_requests, num_gpus, k_bins)
                
                if key in existing_set:
                    continue
                
                print(f"\n[{current}/{total_configs}] {num_requests:,} req, GPUs={num_gpus}, K={k_bins}")
                
                result = run_single_experiment(
                    "multi_bin_dynamic",
                    num_requests,
                    num_gpus,
                    k_bins
                )
                
                if result:
                    results.append(result)
                    print(f"    Token SLA: {result['token_sla_pct']:.1f}% | "
                          f"Request SLA: {result['request_sla_pct']:.1f}% | "
                          f"Decode TBT: {result['avg_decode_tbt_ms']:.2f}ms | "
                          f"Batch: {result['avg_batch_size']:.1f} | "
                          f"Time: {result['execution_time']:.1f}s")
                    
                    # Save incrementally
                    save_results(results, 'step1_grid_search.csv')
    
    print(f"\nStep 1 complete: {len(results)} total results")
    return results


def find_optimal_configs(results):
    """Find optimal GPU/bin combinations for each workload size."""
    df = pd.DataFrame(results)
    
    optimal = {}
    
    for num_requests in REQUEST_COUNTS:
        subset = df[df['num_requests'] == num_requests]
        if len(subset) == 0:
            continue
        
        # Find config with best token SLA compliance
        # If multiple have same token SLA, prefer lower GPU count (cost efficiency)
        subset = subset.sort_values(
            ['token_sla_pct', 'request_sla_pct', 'num_gpus'],
            ascending=[True, True, True]
        )
        
        best = subset.iloc[0]
        optimal[num_requests] = {
            'num_gpus': int(best['num_gpus']),
            'k_bins': int(best['k_bins']),
            'token_sla_pct': best['token_sla_pct'],
            'request_sla_pct': best['request_sla_pct'],
        }
        
        print(f"  {num_requests:,} requests: GPUs={optimal[num_requests]['num_gpus']}, "
              f"K={optimal[num_requests]['k_bins']}, "
              f"Token SLA={optimal[num_requests]['token_sla_pct']:.1f}%")
    
    return optimal


def run_step2_comparison(optimal_configs):
    """Step 2: Compare different scheduling methods."""
    print("\n" + "=" * 70)
    print("STEP 2: METHOD COMPARISON")
    print("=" * 70)
    
    results = []
    
    for num_requests in REQUEST_COUNTS:
        print(f"\n--- {num_requests:,} requests ---")
        
        optimal = optimal_configs.get(num_requests, {'num_gpus': 4, 'k_bins': 8})
        
        # Method 1: Static FIFO (1 GPU, no dynamic batching)
        print("  1. Static FIFO (1 GPU)")
        result = run_single_experiment("static_fifo", num_requests, 1, 1)
        if result:
            result['method'] = 'static_fifo_1gpu'
            results.append(result)
            print(f"     Token={result['token_sla_pct']:.1f}%, Req={result['request_sla_pct']:.1f}%")
        
        # Method 2: Dynamic No-Bins (1 GPU)
        print("  2. Dynamic No-Bins (1 GPU)")
        result = run_single_experiment("dynamic_no_bins", num_requests, 1, 1)
        if result:
            result['method'] = 'dynamic_no_bins_1gpu'
            results.append(result)
            print(f"     Token={result['token_sla_pct']:.1f}%, Req={result['request_sla_pct']:.1f}%")
        
        # Method 3: MultiBin Dynamic (1 GPU, fair comparison)
        print("  3. MultiBin Dynamic (1 GPU, K=8)")
        result = run_single_experiment("multi_bin_dynamic", num_requests, 1, 8)
        if result:
            result['method'] = 'multibin_dynamic_1gpu'
            results.append(result)
            print(f"     Token={result['token_sla_pct']:.1f}%, Req={result['request_sla_pct']:.1f}%")
        
        # Method 4: MultiBin Dynamic (optimal config from Step 1)
        opt_gpus = optimal['num_gpus']
        opt_bins = optimal['k_bins']
        print(f"  4. MultiBin Dynamic (optimal: {opt_gpus} GPUs, K={opt_bins})")
        result = run_single_experiment("multi_bin_dynamic", num_requests, opt_gpus, opt_bins)
        if result:
            result['method'] = f'multibin_dynamic_optimal_{opt_gpus}gpu'
            results.append(result)
            print(f"     Token={result['token_sla_pct']:.1f}%, Req={result['request_sla_pct']:.1f}%")
    
    save_results(results, 'step2_comparison.csv')
    return results


def generate_summary(step1_results, step2_results, optimal_configs):
    """Generate summary report."""
    summary = []
    summary.append("# Stress Test v4 Results Summary")
    summary.append("## v4 SLA Model: TTFT/TBT Separation\n")
    summary.append("**Key Change**: Token SLA applies ONLY to decode TBT (β ≈ 5.74ms/token),")
    summary.append("NOT to TTFT (α ≈ 60ms). This eliminates structural violations.\n")
    summary.append(f"- Token SLA: {D_SLA_TOKEN*1000}ms (decode TBT threshold)")
    summary.append(f"- Request SLA: {D_SLA_REQUEST}s")
    summary.append(f"- RPS Scaling: {RPS_SCALING}x\n")
    
    # Step 1 summary
    summary.append("## Step 1: Optimal Configurations\n")
    summary.append("| Workload | GPUs | Bins | Token SLA % | Request SLA % |")
    summary.append("|----------|------|------|-------------|---------------|")
    for num_requests, cfg in optimal_configs.items():
        summary.append(f"| {num_requests:,} | {cfg['num_gpus']} | {cfg['k_bins']} | "
                      f"{cfg['token_sla_pct']:.1f}% | {cfg['request_sla_pct']:.1f}% |")
    
    # Step 2 summary
    if step2_results:
        summary.append("\n## Step 2: Method Comparison\n")
        df = pd.DataFrame(step2_results)
        
        for num_requests in REQUEST_COUNTS:
            subset = df[df['num_requests'] == num_requests]
            if len(subset) == 0:
                continue
            
            summary.append(f"\n### {num_requests:,} Requests\n")
            summary.append("| Method | Token SLA % | Request SLA % | Decode TBT | Batch |")
            summary.append("|--------|-------------|---------------|------------|-------|")
            
            for _, row in subset.iterrows():
                summary.append(f"| {row['method']} | {row['token_sla_pct']:.1f}% | "
                              f"{row['request_sla_pct']:.1f}% | {row['avg_decode_tbt_ms']:.2f}ms | "
                              f"{row['avg_batch_size']:.1f} |")
    
    # Write summary
    summary_path = os.path.join(RESULTS_DIR, 'RESULTS_SUMMARY.md')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    print(f"\nSummary saved to {summary_path}")


def main():
    print("=" * 70)
    print("STRESS TEST v4: TTFT/TBT SEPARATION MODEL")
    print("=" * 70)
    print()
    print("v4 Changes:")
    print("- Token SLA applies ONLY to decode TBT (beta ~ 5.74ms/token)")
    print("- TTFT (alpha ~ 60ms) is tracked separately, not in token SLA")
    print("- Eliminates structural violations where TTFT dominates")
    print()
    
    # Step 1: Grid search
    step1_results = run_step1_grid_search()
    
    # Find optimal configs
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATIONS")
    print("=" * 70)
    optimal_configs = find_optimal_configs(step1_results)
    
    # Step 2: Method comparison
    step2_results = run_step2_comparison(optimal_configs)
    
    # Generate summary
    generate_summary(step1_results, step2_results, optimal_configs)
    
    print("\n" + "=" * 70)
    print("STRESS TEST v4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
