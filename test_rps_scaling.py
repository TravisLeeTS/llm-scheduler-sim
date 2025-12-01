#!/usr/bin/env python
"""Test with RPS scaling to verify batching works."""

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import load_burstgpt_dataset
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics

def main():
    print("=" * 70)
    print("BATCHING TEST WITH RPS SCALING")
    print("=" * 70)
    
    # Test 1: Real timestamps (sparse arrivals)
    print("\n1. REAL TIMESTAMPS (sparse arrivals):")
    cfg = SchedulerConfig(K_BINS=4, NUM_GPUS=4)
    requests = load_burstgpt_dataset('data/BurstGPT_sample.csv', num_requests=200,
                                      d_sla_token=cfg.D_SLA_TOKEN, d_sla_request=cfg.D_SLA_REQUEST,
                                      use_real_timestamps=True)[:200]
    
    print(f"  First 5 arrivals: {[r.arrival_time for r in requests[:5]]}")
    print(f"  Time span: {requests[-1].arrival_time - requests[0].arrival_time:.0f}s")
    
    sim = Simulator(cfg, requests.copy(), scheduler_type='multi_bin_dynamic')
    completed = sim.run()
    metrics = compute_metrics(completed, d_sla_token=cfg.D_SLA_TOKEN, d_sla_request=cfg.D_SLA_REQUEST)
    
    gpu_stats = sim.get_gpu_stats()
    total_batches = sum(g['total_batches'] for g in gpu_stats)
    avg_batch_size = len(completed) / total_batches if total_batches > 0 else 0
    
    print(f"  Total batches: {total_batches}")
    print(f"  Avg batch size: {avg_batch_size:.1f}")
    print(f"  Throughput: {metrics['throughput_requests_per_sec']:.3f} req/s")
    
    # Test 2: RPS scaling (compressed arrivals)
    print("\n2. RPS SCALING (200x compression):")
    requests2 = load_burstgpt_dataset('data/BurstGPT_sample.csv', num_requests=200,
                                       d_sla_token=cfg.D_SLA_TOKEN, d_sla_request=cfg.D_SLA_REQUEST,
                                       use_real_timestamps=False, rps_scaling=200.0)[:200]
    
    print(f"  First 5 arrivals: {[f'{r.arrival_time:.3f}' for r in requests2[:5]]}")
    print(f"  Time span: {requests2[-1].arrival_time - requests2[0].arrival_time:.1f}s")
    
    sim2 = Simulator(cfg, requests2.copy(), scheduler_type='multi_bin_dynamic')
    completed2 = sim2.run()
    metrics2 = compute_metrics(completed2, d_sla_token=cfg.D_SLA_TOKEN, d_sla_request=cfg.D_SLA_REQUEST)
    
    gpu_stats2 = sim2.get_gpu_stats()
    total_batches2 = sum(g['total_batches'] for g in gpu_stats2)
    avg_batch_size2 = len(completed2) / total_batches2 if total_batches2 > 0 else 0
    
    print(f"  Total batches: {total_batches2}")
    print(f"  Avg batch size: {avg_batch_size2:.1f}")
    print(f"  Throughput: {metrics2['throughput_requests_per_sec']:.2f} req/s")
    print(f"  Avg Latency: {metrics2['avg_latency']*1000:.0f}ms")
    print(f"  SLA violation: {metrics2['sla_violation_rate_token']*100:.1f}%")
    
    # Test 3: Compare schedulers with RPS scaling
    print("\n3. SCHEDULER COMPARISON WITH RPS SCALING:")
    print("-" * 50)
    
    for sched_type in ["static_fifo", "dynamic_no_bins", "multi_bin_dynamic"]:
        requests3 = load_burstgpt_dataset('data/BurstGPT_sample.csv', num_requests=500,
                                           d_sla_token=cfg.D_SLA_TOKEN, d_sla_request=cfg.D_SLA_REQUEST,
                                           use_real_timestamps=False, rps_scaling=200.0)[:500]
        
        sim3 = Simulator(cfg, requests3, scheduler_type=sched_type)
        completed3 = sim3.run()
        metrics3 = compute_metrics(completed3, d_sla_token=cfg.D_SLA_TOKEN, d_sla_request=cfg.D_SLA_REQUEST)
        
        gpu_stats3 = sim3.get_gpu_stats()
        total_batches3 = sum(g['total_batches'] for g in gpu_stats3)
        avg_batch_size3 = len(completed3) / total_batches3 if total_batches3 > 0 else 0
        
        print(f"\n  {sched_type}:")
        print(f"    Avg batch size: {avg_batch_size3:.1f}")
        print(f"    Throughput: {metrics3['throughput_requests_per_sec']:.2f} req/s")
        print(f"    Avg Latency: {metrics3['avg_latency']*1000:.0f}ms")
        print(f"    SLA violation: {metrics3['sla_violation_rate_token']*100:.1f}%")


if __name__ == "__main__":
    main()
