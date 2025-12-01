#!/usr/bin/env python3
"""Run 1M request tests with multi_bin_dynamic scheduler for various GPU counts."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.config import SchedulerConfig, compute_equal_mass_boundaries
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics
import pandas as pd
from datetime import datetime
import csv
import time

# Cache for workloads
_workload_cache = {}
_bin_cache = {}

def run_test(num_requests, num_gpus, scheduler_type, k_bins=8):
    """Run a single test and return results."""
    print(f"Testing {num_requests:,} requests with {scheduler_type}, {num_gpus} GPUs, K={k_bins}...")
    start_time = time.time()
    
    # Get or generate workload
    cache_key = num_requests
    if cache_key not in _workload_cache:
        print(f"  Loading {num_requests:,} requests (first time)...")
        temp_cfg = SchedulerConfig(
            NUM_REQUESTS=num_requests,
            USE_REAL_TIMESTAMPS=True,
        )
        _workload_cache[cache_key] = generate_workload(temp_cfg)
    
    workload = _workload_cache[cache_key]
    
    # Get or compute bin boundaries
    bin_key = (num_requests, k_bins)
    if bin_key not in _bin_cache:
        print(f"  Computing bin boundaries for K={k_bins}...")
        predicted_lengths = [r.output_len for r in workload]
        _bin_cache[bin_key] = compute_equal_mass_boundaries(predicted_lengths, k_bins)
    
    bin_boundaries = _bin_cache[bin_key]
    
    # Create config with correct boundaries
    cfg = SchedulerConfig(
        NUM_REQUESTS=num_requests,
        NUM_GPUS=num_gpus,
        K_BINS=k_bins,
        USE_REAL_TIMESTAMPS=True,
        BIN_BOUNDARIES=bin_boundaries,
    )
    
    sim = Simulator(cfg, workload, scheduler_type=scheduler_type)
    completed = sim.run()
    
    m = compute_metrics(completed, cfg.D_SLA)
    util = compute_gpu_utilization(sim.get_gpu_stats())
    batch_stats = compute_batch_statistics(completed)
    
    execution_time = time.time() - start_time
    
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'scheduler_type': scheduler_type,
        'num_requests': num_requests,
        'num_gpus': num_gpus,
        'use_real_timestamps': True,
        'rps_scaling': '',
        'd_sla': cfg.D_SLA,
        'k_bins': k_bins,
        'sla_violation_rate': m['sla_violation_rate'],
        'capacity_qps_under_sla': m['capacity_qps_under_sla'],
        'throughput_requests_per_sec': m['throughput_requests_per_sec'],
        'throughput_tokens_per_sec': m['throughput_tokens_per_sec'],
        'avg_latency': m['avg_latency'],
        'p50_latency': m['p50_latency'],
        'p95_latency': m['p95_latency'],
        'p99_latency': m['p99_latency'],
        'max_latency': m['max_latency'],
        'avg_queueing_delay': m['avg_queueing_delay'],
        'avg_service_time': m['avg_service_time'],
        'avg_gpu_utilization': util['avg_utilization'],
        'min_gpu_utilization': util['min_utilization'],
        'max_gpu_utilization': util['max_utilization'],
        'num_batches': batch_stats['num_batches'],
        'avg_batch_size': batch_stats['avg_batch_size'],
        'min_batch_size': batch_stats['min_batch_size'],
        'max_batch_size': batch_stats['max_batch_size'],
        'total_time': m['total_time'],
        'total_tokens': m['total_tokens'],
        'num_completed': len(completed),
        'execution_time_seconds': execution_time,
        'status': 'success'
    }

def main():
    results = []
    
    # Step 1 completion: 1M requests for multi_bin_dynamic (1,2,4 GPUs already done up to 100K)
    # Need to run 1M for all schedulers
    print("\n=== STEP 1 COMPLETION: 1M Request Tests ===\n")
    
    # Run 1M for multi_bin_dynamic with 1, 2, 4 GPUs
    for num_gpus in [1, 2, 4]:
        import time
        start = time.time()
        result = run_test(1000000, num_gpus, 'multi_bin_dynamic', k_bins=8)
        result['execution_time_seconds'] = time.time() - start
        results.append(result)
        print(f"  SLA={result['sla_violation_rate']*100:.2f}%, Latency={result['avg_latency']:.2f}s\n")
    
    # Step 2: GPU Scaling (1M requests, 8-100 GPUs) for multi_bin_dynamic
    print("\n=== STEP 2: GPU SCALING (1M requests) ===\n")
    
    for num_gpus in [8, 16, 32, 64, 100]:
        import time
        start = time.time()
        result = run_test(1000000, num_gpus, 'multi_bin_dynamic', k_bins=8)
        result['execution_time_seconds'] = time.time() - start
        results.append(result)
        print(f"  SLA={result['sla_violation_rate']*100:.2f}%, Latency={result['avg_latency']:.2f}s\n")
    
    # Step 3: K-Bins Sensitivity (best GPU from Step 2, various K values)
    print("\n=== STEP 3: K-BINS SENSITIVITY (best GPU config) ===\n")
    
    # Use 8 GPUs for K-bins sensitivity
    for k_bins in [1, 2, 4, 8, 16, 32]:
        import time
        start = time.time()
        result = run_test(1000000, 8, 'multi_bin_dynamic', k_bins=k_bins)
        result['execution_time_seconds'] = time.time() - start
        results.append(result)
        print(f"  K={k_bins}: SLA={result['sla_violation_rate']*100:.2f}%, Latency={result['avg_latency']:.2f}s\n")
    
    # Save results
    output_file = 'results_1m_extended.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(df[['scheduler_type', 'num_requests', 'num_gpus', 'k_bins', 'sla_violation_rate', 'avg_latency', 'throughput_requests_per_sec']].to_string())

if __name__ == '__main__':
    main()
