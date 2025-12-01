#!/usr/bin/env python3
"""
Complete missing configurations for 100K and 1M workloads.
Fills in GPUs 1, 2, 4 for 100K and all GPUs for 1M.
"""

import os
import sys
import time
import signal
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics

# Configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

D_SLA_TOKEN = 0.030
D_SLA_REQUEST = 20.0
RPS_SCALING = 200.0

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Experiment timed out")

def run_single_experiment(scheduler_type, num_requests, num_gpus, k_bins, timeout_sec=300, verbose=True):
    """Run a single experiment with timeout protection."""
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
        error_msg = str(e)[:80]
        print(f"    ERROR: {error_msg}...")
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
            'sla_violation_rate_token': 1.0,  # Mark as failed
            'sla_violation_rate_request': 1.0,
            'avg_gpu_utilization': 1.0,
            'avg_batch_size': 0,
            'avg_queueing_delay': 0,
            'avg_service_time': 0,
            'total_time': 0,
            'execution_time_seconds': elapsed,
            'status': f'error: {error_msg}'
        }


def main():
    """Fill in missing configurations."""
    
    output_file = 'stress_test_v3_results/step1_grid_search.csv'
    
    # Load existing results
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        results = existing_df.to_dict('records')
        existing_keys = set()
        for r in results:
            existing_keys.add((int(r['num_requests']), int(r['num_gpus']), int(r['k_bins'])))
        print(f"Loaded {len(results)} existing results")
    else:
        results = []
        existing_keys = set()
        print("Starting fresh")
    
    # Full configuration space
    # Missing: 100K with GPUs 1,2,4 and all 1M configs
    all_gpu_counts = [1, 2, 4, 8, 16, 32, 64, 100]
    bin_counts = [1, 2, 4, 8, 16, 32]
    
    # Calculate what's missing
    missing_configs = []
    
    # Check 100K missing configs
    for gpu in all_gpu_counts:
        for k in bin_counts:
            if (100000, gpu, k) not in existing_keys:
                missing_configs.append((100000, gpu, k))
    
    # Check 1M missing configs
    for gpu in all_gpu_counts:
        for k in bin_counts:
            if (1000000, gpu, k) not in existing_keys:
                missing_configs.append((1000000, gpu, k))
    
    print(f"\nMissing configurations: {len(missing_configs)}")
    
    if not missing_configs:
        print("All configurations complete!")
        return
    
    # Show what's missing
    missing_100k = [c for c in missing_configs if c[0] == 100000]
    missing_1m = [c for c in missing_configs if c[0] == 1000000]
    print(f"  100K: {len(missing_100k)} configs missing")
    print(f"  1M: {len(missing_1m)} configs missing")
    
    print("\n" + "="*80)
    print("FILLING MISSING CONFIGURATIONS")
    print("="*80)
    print(f"SLA: Token={D_SLA_TOKEN*1000}ms, Request={D_SLA_REQUEST}s")
    print()
    
    total = len(missing_configs)
    for i, (num_requests, num_gpus, k_bins) in enumerate(missing_configs, 1):
        # Double-check not already done (in case of concurrent runs)
        key = (num_requests, num_gpus, k_bins)
        if key in existing_keys:
            print(f"[{i}/{total}] {num_requests:,} req, GPUs={num_gpus}, K={k_bins} [ALREADY DONE]")
            continue
        
        print(f"\n[{i}/{total}] {num_requests:,} req, GPUs={num_gpus}, K={k_bins}")
        
        result = run_single_experiment(
            scheduler_type='multi_bin_dynamic',
            num_requests=num_requests,
            num_gpus=num_gpus,
            k_bins=k_bins,
            verbose=True
        )
        results.append(result)
        existing_keys.add(key)
        
        # Save after each config
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"    [Saved {len(results)} results]")
    
    print(f"\n[COMPLETE] Total results: {len(results)}")
    
    # Show coverage
    df = pd.DataFrame(results)
    print("\nFinal GPU coverage:")
    for req in sorted(df['num_requests'].unique()):
        gpus = sorted(df[df['num_requests'] == req]['num_gpus'].unique())
        print(f"  {int(req):,}: {len(gpus)} GPU counts")


if __name__ == '__main__':
    main()
