#!/usr/bin/env python3
"""
Step 2: Method Comparison
Compare 4 methods using optimal configurations from Step 1.

Methods:
1. Static FIFO (1 GPU) - baseline
2. Dynamic No-Bins (1 GPU) - dynamic batching only  
3. Multi-Bin Dynamic (1 GPU, K=8)
4. Multi-Bin Dynamic (optimal from Step 1)
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
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

D_SLA_TOKEN = 0.030
D_SLA_REQUEST = 20.0
RPS_SCALING = 200.0

# Optimal configurations from Step 1 analysis
OPTIMAL_CONFIGS = {
    1000: {'num_gpus': 32, 'k_bins': 8},
    10000: {'num_gpus': 16, 'k_bins': 8},
    100000: {'num_gpus': 64, 'k_bins': 8},
}

def run_single_experiment(scheduler_type, num_requests, num_gpus, k_bins, verbose=True):
    """Run a single experiment configuration."""
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
        
        if scheduler_type == 'static_fifo':
            cfg.B_FIXED = 8
        
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
    """Run Step 2: Compare 4 methods for each workload."""
    
    output_file = 'stress_test_v3_results/step2_comparison.csv'
    
    # Load existing results if available
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        results = existing_df.to_dict('records')
        existing_keys = set()
        for r in results:
            existing_keys.add((r['num_requests'], r.get('method', '')))
        print(f"Loaded {len(results)} existing results")
    else:
        results = []
        existing_keys = set()
        print("Starting fresh")
    
    # Workloads to test (skip 1M for now)
    request_counts = [1000, 10000, 100000]
    
    print("\n" + "="*80)
    print("STEP 2: METHOD COMPARISON")
    print("="*80)
    print(f"Workloads: {request_counts}")
    print(f"SLA: Token={D_SLA_TOKEN*1000}ms, Request={D_SLA_REQUEST}s")
    print()
    
    for num_requests in request_counts:
        print(f"\n{'='*60}")
        print(f"WORKLOAD: {num_requests:,} requests")
        print(f"{'='*60}")
        
        opt = OPTIMAL_CONFIGS.get(num_requests, {'num_gpus': 16, 'k_bins': 8})
        
        # Method 1: Static FIFO (1 GPU)
        method_name = 'Static_FIFO_1GPU'
        if (num_requests, method_name) not in existing_keys:
            print(f"\n[1/4] Static FIFO (1 GPU)")
            result = run_single_experiment('static_fifo', num_requests, 1, 1)
            result['method'] = method_name
            results.append(result)
            pd.DataFrame(results).to_csv(output_file, index=False)
        else:
            print(f"\n[1/4] Static FIFO (1 GPU) [ALREADY DONE]")
        
        # Method 2: Dynamic No-Bins (1 GPU)
        method_name = 'Dynamic_NoBins_1GPU'
        if (num_requests, method_name) not in existing_keys:
            print(f"\n[2/4] Dynamic No-Bins (1 GPU)")
            result = run_single_experiment('dynamic_no_bins', num_requests, 1, 1)
            result['method'] = method_name
            results.append(result)
            pd.DataFrame(results).to_csv(output_file, index=False)
        else:
            print(f"\n[2/4] Dynamic No-Bins (1 GPU) [ALREADY DONE]")
        
        # Method 3: Multi-Bin Dynamic (1 GPU, K=8)
        method_name = 'MultiBin_1GPU_K8'
        if (num_requests, method_name) not in existing_keys:
            print(f"\n[3/4] Multi-Bin Dynamic (1 GPU, K=8)")
            result = run_single_experiment('multi_bin_dynamic', num_requests, 1, 8)
            result['method'] = method_name
            results.append(result)
            pd.DataFrame(results).to_csv(output_file, index=False)
        else:
            print(f"\n[3/4] Multi-Bin Dynamic (1 GPU, K=8) [ALREADY DONE]")
        
        # Method 4: Optimal Multi-Bin
        opt_gpus = opt['num_gpus']
        opt_bins = opt['k_bins']
        method_name = f'MultiBin_Optimal_{opt_gpus}GPU_K{opt_bins}'
        if (num_requests, method_name) not in existing_keys:
            print(f"\n[4/4] Multi-Bin Optimal ({opt_gpus} GPUs, K={opt_bins})")
            result = run_single_experiment('multi_bin_dynamic', num_requests, opt_gpus, opt_bins)
            result['method'] = method_name
            results.append(result)
            pd.DataFrame(results).to_csv(output_file, index=False)
        else:
            print(f"\n[4/4] Multi-Bin Optimal ({opt_gpus} GPUs, K={opt_bins}) [ALREADY DONE]")
    
    print(f"\n[COMPLETE] Step 2 results saved to {output_file}")
    print(f"Total results: {len(results)}")
    
    # Print summary table
    print("\n" + "="*80)
    print("STEP 2 SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    df['token_pass'] = (1 - df['sla_violation_rate_token']) * 100
    df['req_pass'] = (1 - df['sla_violation_rate_request']) * 100
    
    for req in sorted(df['num_requests'].unique()):
        subset = df[df['num_requests'] == req]
        print(f"\n{req:,} requests:")
        for _, row in subset.iterrows():
            print(f"  {row['method']:30s}: Token={row['token_pass']:5.1f}%, Req={row['req_pass']:5.1f}%, "
                  f"Batch={row['avg_batch_size']:5.1f}")
    
    return df


if __name__ == '__main__':
    main()
