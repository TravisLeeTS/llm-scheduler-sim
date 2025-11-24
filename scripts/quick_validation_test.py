#!/usr/bin/env python3
"""
Quick validation test to verify dynamic batching fixes.
Tests 1K requests with all three schedulers to ensure dynamic > static.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics


def test_scheduler(scheduler_type, num_gpus=1):
    """Test a single scheduler configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {scheduler_type} ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")
    print(f"{'='*60}")
    
    cfg = SchedulerConfig(
        NUM_GPUS=num_gpus,
        K_BINS=4,
        NUM_REQUESTS=1000,
        SEED=42,
        D_SLA=1.0,
        DATASET_PATH="data/BurstGPT_sample.csv",
        WORKLOAD_SOURCE="burstgpt_dataset",
        RPS_SCALING=200.0,
        USE_EQUAL_MASS_BINS=True,
        USE_REAL_CALIBRATION=True,
        CALIBRATION_CSV_PATH="data/qwen3_1_7b_latency_grid.csv"
    )
    
    start = time.time()
    requests = generate_workload(cfg)
    simulator = Simulator(cfg, requests, scheduler_type)
    completed = simulator.run()
    metrics = compute_metrics(completed)
    elapsed = time.time() - start
    
    print(f"\n[Results]")
    print(f"  SLA Violations:     {metrics['sla_violation_rate']*100:6.2f}%")
    print(f"  Capacity QPS:       {metrics['capacity_qps_under_sla']:8.2f}")
    print(f"  Avg Latency:        {metrics['avg_latency']:8.3f}s")
    print(f"  P95 Latency:        {metrics['p95_latency']:8.3f}s")
    print(f"  Throughput:         {metrics['throughput_requests_per_sec']:8.2f} req/s")
    print(f"  Completed:          {len(completed):6d}/{len(requests)}")
    print(f"  Execution Time:     {elapsed:8.1f}s")
    
    return {
        'scheduler': scheduler_type,
        'num_gpus': num_gpus,
        'sla_violation_rate': metrics['sla_violation_rate'],
        'capacity_qps': metrics['capacity_qps_under_sla'],
        'avg_latency': metrics['avg_latency'],
        'p95_latency': metrics['p95_latency'],
        'throughput': metrics['throughput_requests_per_sec'],
    }


def main():
    print("="*60)
    print("QUICK VALIDATION TEST - 1K Requests")
    print("Expected: dynamic_no_bins > static_fifo")
    print("         multi_bin_dynamic (4 GPUs) >> both")
    print("="*60)
    
    results = []
    
    # Test baselines (1 GPU each)
    results.append(test_scheduler('static_fifo', num_gpus=1))
    results.append(test_scheduler('dynamic_no_bins', num_gpus=1))
    
    # Test multi-bin (4 GPUs)
    results.append(test_scheduler('multi_bin_dynamic', num_gpus=4))
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    static = results[0]
    dynamic = results[1]
    multibin = results[2]
    
    print(f"\n1. Static FIFO (1 GPU):")
    print(f"   Capacity QPS: {static['capacity_qps']:.2f}")
    print(f"   SLA Violations: {static['sla_violation_rate']*100:.1f}%")
    
    print(f"\n2. Dynamic No-Bins (1 GPU):")
    print(f"   Capacity QPS: {dynamic['capacity_qps']:.2f}")
    print(f"   SLA Violations: {dynamic['sla_violation_rate']*100:.1f}%")
    
    if dynamic['capacity_qps'] > static['capacity_qps']:
        improvement = dynamic['capacity_qps'] / static['capacity_qps']
        print(f"   ✓ {improvement:.2f}x better than static_fifo")
    else:
        ratio = static['capacity_qps'] / dynamic['capacity_qps']
        print(f"   ✗ WORSE than static_fifo by {ratio:.2f}x - PROBLEM!")
    
    print(f"\n3. Multi-Bin Dynamic (4 GPUs):")
    print(f"   Capacity QPS: {multibin['capacity_qps']:.2f}")
    print(f"   SLA Violations: {multibin['sla_violation_rate']*100:.1f}%")
    
    if multibin['capacity_qps'] > dynamic['capacity_qps']:
        improvement = multibin['capacity_qps'] / dynamic['capacity_qps']
        print(f"   ✓ {improvement:.2f}x better than dynamic_no_bins")
    
    print("\n" + "="*60)
    print("VALIDATION:", "PASSED ✓" if dynamic['capacity_qps'] > static['capacity_qps'] else "FAILED ✗")
    print("="*60)


if __name__ == "__main__":
    main()
