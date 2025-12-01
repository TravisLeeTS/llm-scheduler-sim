#!/usr/bin/env python3
"""
Continue Step 1 Grid Search - only 100K and 1M workloads with GPUs >= 8
This script picks up where the main test left off.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics

# ============================================================================
# CONFIGURATION - Must match comprehensive_stress_test_v3.py
# ============================================================================

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

D_SLA_TOKEN = 0.030  # 30ms token SLA
D_SLA_REQUEST = 20.0  # 20s request SLA
RPS_SCALING = 200.0  # 54 req/s (200x native)

# ============================================================================

def run_single_experiment(scheduler_type, num_requests, num_gpus, k_bins, verbose=True):
    """Run a single experiment configuration."""
    start_time = time.time()
    
    try:
        # Configure scheduler
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
        
        # Generate workload
        requests = generate_workload(cfg)
        
        # Run simulation
        simulator = Simulator(cfg, requests, scheduler_type)
        completed_requests = simulator.run()
        
        # Compute metrics
        metrics = compute_metrics(
            completed_requests, 
            d_sla_token=D_SLA_TOKEN,
            d_sla_request=D_SLA_REQUEST
        )
        
        gpu_stats = simulator.get_gpu_stats()
        gpu_util = compute_gpu_utilization(gpu_stats)
        batch_stats = compute_batch_statistics(completed_requests)
        
        execution_time = time.time() - start_time
        actual_rps = num_requests / metrics.get('total_time', 1) if metrics.get('total_time', 0) > 0 else 0
        
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'k_bins': k_bins,
            'rps_scaling': RPS_SCALING,
            'target_rps': RPS_SCALING * 0.27,
            'actual_rps': actual_rps,
            'd_sla_token': D_SLA_TOKEN * 1000,
            'd_sla_request': D_SLA_REQUEST,
            'throughput_tokens_per_sec': metrics.get('throughput_tokens_per_sec', 0),
            'throughput_requests_per_sec': metrics.get('throughput_requests_per_sec', 0),
            'avg_latency': metrics.get('avg_latency', 0),
            'p50_latency': metrics.get('p50_latency', 0),
            'p95_latency': metrics.get('p95_latency', 0),
            'p99_latency': metrics.get('p99_latency', 0),
            'avg_tbt_ms': metrics.get('avg_tbt', 0) * 1000,
            'p95_tbt_ms': metrics.get('p95_tbt', 0) * 1000,
            'sla_violation_rate_token': metrics.get('sla_violation_rate_token', 0),
            'sla_violation_rate_request': metrics.get('sla_violation_rate_request', 0),
            'avg_gpu_utilization': gpu_util.get('avg_utilization', 0),
            'avg_batch_size': batch_stats.get('avg_batch_size', 0),
            'avg_queueing_delay': metrics.get('avg_queueing_delay', 0),
            'avg_service_time': metrics.get('avg_service_time', 0),
            'total_time': metrics.get('total_time', 0),
            'execution_time_seconds': execution_time,
            'status': 'success'
        }
        
        if verbose:
            token_pass = (1 - result['sla_violation_rate_token']) * 100
            req_pass = (1 - result['sla_violation_rate_request']) * 100
            print(f"    Token: {token_pass:.1f}% | Req: {req_pass:.1f}% | "
                  f"Batch: {result['avg_batch_size']:.1f} | "
                  f"GPU: {result['avg_gpu_utilization']*100:.1f}% | "
                  f"Time: {execution_time:.1f}s")
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    ERROR: {str(e)[:80]}...")
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'k_bins': k_bins,
            'rps_scaling': RPS_SCALING,
            'target_rps': RPS_SCALING * 0.27,
            'actual_rps': 0,
            'd_sla_token': D_SLA_TOKEN * 1000,
            'd_sla_request': D_SLA_REQUEST,
            'throughput_tokens_per_sec': 0,
            'throughput_requests_per_sec': 0,
            'avg_latency': 0,
            'p50_latency': 0,
            'p95_latency': 0,
            'p99_latency': 0,
            'avg_tbt_ms': 0,
            'p95_tbt_ms': 0,
            'sla_violation_rate_token': np.nan,
            'sla_violation_rate_request': np.nan,
            'avg_gpu_utilization': np.nan,
            'avg_batch_size': np.nan,
            'avg_queueing_delay': 0,
            'avg_service_time': 0,
            'total_time': 0,
            'execution_time_seconds': elapsed,
            'status': f'error: {str(e)[:100]}'
        }


def main():
    """Continue Step 1 with remaining 100K and 1M workloads."""
    
    # Check existing results
    existing_file = 'stress_test_v3_results/step1_grid_search.csv'
    output_file = 'stress_test_v3_results/step1_grid_search.csv'  # Append to same file
    
    if os.path.exists(existing_file):
        existing_df = pd.read_csv(existing_file)
        existing_results = existing_df.to_dict('records')
        print(f"Loaded {len(existing_results)} existing results")
    else:
        existing_results = []
        print("Starting fresh")
    
    # What workloads do we have?
    existing_workloads = set()
    if existing_results:
        for r in existing_results:
            key = (r['num_requests'], r['num_gpus'], r['k_bins'])
            existing_workloads.add(key)
        print(f"Existing configs: {len(existing_workloads)}")
    
    # Remaining configurations to run
    # Only 100K and 1M with GPUs >= 8
    request_counts = [100_000, 1_000_000]
    gpu_counts = [8, 16, 32, 64, 100]  # Skip 1, 2, 4 GPUs
    bin_counts = [1, 2, 4, 8, 16, 32]
    
    total_remaining = len(request_counts) * len(gpu_counts) * len(bin_counts)
    
    print("\n" + "="*80)
    print("CONTINUING STEP 1: Grid Search for 100K and 1M workloads (GPUs >= 8)")
    print("="*80)
    print(f"Request counts: {request_counts}")
    print(f"GPU counts: {gpu_counts}")
    print(f"Bin counts: {bin_counts}")
    print(f"Max remaining configs: {total_remaining}")
    print(f"SLA: Token={D_SLA_TOKEN*1000}ms, Request={D_SLA_REQUEST}s")
    print()
    
    results = list(existing_results)  # Start with existing
    config_num = 0
    new_count = 0
    
    for num_requests in request_counts:
        print(f"\n{'='*60}")
        print(f"WORKLOAD: {num_requests:,} requests")
        print(f"{'='*60}")
        
        for num_gpus in gpu_counts:
            for k_bins in bin_counts:
                config_num += 1
                
                # Check if already done
                key = (num_requests, num_gpus, k_bins)
                if key in existing_workloads:
                    print(f"[{config_num}/{total_remaining}] GPUs={num_gpus}, K={k_bins} [ALREADY DONE]")
                    continue
                
                effective_rps = RPS_SCALING * 0.27
                print(f"\n[{config_num}/{total_remaining}] GPUs={num_gpus}, K={k_bins}, RPS={effective_rps:.0f}")
                
                result = run_single_experiment(
                    scheduler_type='multi_bin_dynamic',
                    num_requests=num_requests,
                    num_gpus=num_gpus,
                    k_bins=k_bins,
                    verbose=True
                )
                results.append(result)
                new_count += 1
                
                # Save after each config
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                print(f"    [Saved {len(results)} results to {output_file}]")
    
    print(f"\n[COMPLETE] Added {new_count} new results")
    print(f"Total results: {len(results)}")
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    main()
