#!/usr/bin/env python
"""Test integration of SLA controller, memory algorithm, and per-bin batch limits."""

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import load_burstgpt_dataset
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics

def main():
    # Get realistic SLA thresholds from config
    cfg_default = SchedulerConfig()
    d_sla_token = cfg_default.D_SLA_TOKEN  # 50ms
    d_sla_request = cfg_default.D_SLA_REQUEST  # 10s
    
    # Load workload with realistic SLA thresholds
    requests = load_burstgpt_dataset(
        'data/BurstGPT_sample.csv', 
        num_requests=500,
        d_sla_token=d_sla_token,
        d_sla_request=d_sla_request,
        use_real_timestamps=True
    )[:500]
    
    print("=" * 60)
    print("Testing SLA + Memory Algorithm Integration")
    print("=" * 60)
    print(f"Requests: {len(requests)}")
    print(f"SLA Thresholds: {d_sla_token*1000:.0f}ms per-token, {d_sla_request:.0f}s per-request")
    
    # Test with different K_BINS values
    for k_bins in [2, 4, 6]:
        print(f"\n{'='*60}")
        print(f"K_BINS = {k_bins}")
        print("=" * 60)
        
        cfg = SchedulerConfig(K_BINS=k_bins)
        sim = Simulator(cfg, requests)
        completed = sim.run()
        
        metrics = compute_metrics(completed, d_sla_token=d_sla_token, d_sla_request=d_sla_request)
        
        print(f"\nBin Configuration:")
        print(f"  Boundaries: {cfg.BIN_BOUNDARIES}")
        print(f"  Per-bin B_MAX: {cfg.BIN_B_MAX}")
        
        print(f"\nResults:")
        print(f"  Completed: {len(completed)}")
        print(f"  SLA Token Violation: {metrics['sla_violation_rate_token']*100:.1f}%")
        print(f"  SLA Request Violation: {metrics['sla_violation_rate_request']*100:.1f}%")
        print(f"  Avg TBT: {metrics['avg_tbt']*1000:.1f}ms")
        print(f"  Avg Latency: {metrics['avg_latency']*1000:.1f}ms")

if __name__ == "__main__":
    main()
