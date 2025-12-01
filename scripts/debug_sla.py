#!/usr/bin/env python3
"""Debug SLA calculations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics

D_SLA_TOKEN = 0.010      # 10ms
D_SLA_REQUEST = 20.0     # 20 seconds

def test_sla_calculation(num_gpus, scheduler_type, num_requests=100):
    """Test SLA calculation for given config."""
    cfg = SchedulerConfig(
        NUM_GPUS=num_gpus,
        K_BINS=1,
        NUM_REQUESTS=num_requests,
        EXPERIMENT_MODE=scheduler_type,
        B_MIN=1,
        B_MAX=128,
        D_SLA=D_SLA_TOKEN,
        D_SLA_TOKEN=D_SLA_TOKEN,
        D_SLA_REQUEST=D_SLA_REQUEST,
        USE_REAL_CALIBRATION=True,
        CALIBRATION_CSV_PATH='data/qwen3_1_7b_latency_grid.csv',
        WORKLOAD_SOURCE='burstgpt_dataset',
        DATASET_PATH='data/BurstGPT_sample.csv',
        USE_REAL_TIMESTAMPS=False,
        RPS_SCALING=100.0,
        SEED=42,
    )

    requests = generate_workload(cfg)
    simulator = Simulator(cfg, requests, scheduler_type)
    completed = simulator.run()

    # Analyze requests
    print(f"\n=== {scheduler_type} with {num_gpus} GPU(s) ===")
    print(f"D_SLA_TOKEN = {D_SLA_TOKEN*1000}ms, D_SLA_REQUEST = {D_SLA_REQUEST}s")
    print(f"Completed: {len(completed)} requests")
    
    # Sample requests
    print("\nSample requests:")
    for i, r in enumerate(completed[:3]):
        print(f"  Req {i}: decode_tbt={r.decode_tbt*1000:.2f}ms, latency={r.latency:.2f}s")
        print(f"          deadline={r.deadline*1000}ms, deadline_request={r.deadline_request}s")
        print(f"          violates_sla={r.violates_sla}, violates_sla_request={r.violates_sla_request}")

    # Stats
    latencies = [r.latency for r in completed]
    decode_tbts = [r.decode_tbt*1000 for r in completed if r.decode_tbt >= 0]
    
    print(f"\nLatency distribution:")
    print(f"  Min: {min(latencies):.2f}s, Max: {max(latencies):.2f}s, Avg: {sum(latencies)/len(latencies):.2f}s")
    print(f"  Below {D_SLA_REQUEST}s: {sum(1 for l in latencies if l <= D_SLA_REQUEST)} / {len(latencies)}")
    
    print(f"\nDecode TBT distribution:")
    print(f"  Min: {min(decode_tbts):.2f}ms, Max: {max(decode_tbts):.2f}ms, Avg: {sum(decode_tbts)/len(decode_tbts):.2f}ms")
    print(f"  Below {D_SLA_TOKEN*1000}ms: {sum(1 for t in decode_tbts if t <= D_SLA_TOKEN*1000)} / {len(decode_tbts)}")

    # Compute metrics
    metrics = compute_metrics(completed, D_SLA_TOKEN, D_SLA_REQUEST)
    print(f"\nMetrics:")
    print(f"  Token SLA: {metrics.get('sla_violation_rate_token', 0)*100:.2f}%")
    print(f"  Request SLA: {metrics.get('sla_violation_rate_request', 0)*100:.2f}%")
    print(f"  Avg latency: {metrics.get('avg_latency', 0):.2f}s")
    print(f"  Avg decode_tbt: {metrics.get('avg_decode_tbt', 0)*1000:.2f}ms")


if __name__ == "__main__":
    print("=" * 70)
    print("SLA CALCULATION DEBUG")
    print("=" * 70)
    
    # Test with 1 GPU - should have high request SLA due to queueing
    test_sla_calculation(1, "static_fifo", 100)
    
    # Test with 16 GPUs - should have low request SLA
    test_sla_calculation(16, "multi_bin_dynamic", 100)
