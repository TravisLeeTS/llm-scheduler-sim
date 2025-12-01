#!/usr/bin/env python3
"""
Test script for Dual SLA Monitoring.

This script demonstrates the dual SLA monitoring capability:
1. Per-Request SLA (D_SLA_REQUEST = 150ms) - Total response latency for interactive use
2. Per-Token SLA (D_SLA_TOKEN = 5ms) - Time Between Tokens for streaming UX

Reference: Gemini 2.5 Flash-Lite (2025)
- TTFT: 240ms
- Output speed: 410 tokens/sec = 2.44ms/token
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, print_metrics_table, compute_gpu_utilization


def run_dual_sla_test(num_requests=10000, num_gpus=8, use_real_timestamps=False, rps_scaling=200.0):
    """
    Run a test with dual SLA monitoring.
    
    Args:
        num_requests: Number of requests to simulate
        num_gpus: Number of GPUs to use
        use_real_timestamps: Whether to use real timestamps (True) or RPS scaling (False)
        rps_scaling: RPS scaling factor (only used if use_real_timestamps=False)
    """
    print("=" * 80)
    print("DUAL SLA MONITORING TEST")
    print("=" * 80)
    
    # Create configuration with dual SLA thresholds
    cfg = SchedulerConfig(
        NUM_REQUESTS=num_requests,
        NUM_GPUS=num_gpus,
        K_BINS=4,
        USE_REAL_TIMESTAMPS=use_real_timestamps,
        RPS_SCALING=rps_scaling,
    )
    
    print(f"\n[Configuration]")
    print(f"  Requests: {num_requests:,}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Real Timestamps: {use_real_timestamps}")
    print(f"  RPS Scaling: {rps_scaling}x" if not use_real_timestamps else "")
    
    print(f"\n[SLA Thresholds (Calibrated for RTX 4080 + Qwen3 1.7B)]")
    print(f"  Per-Token SLA (D_SLA_TOKEN):     {cfg.D_SLA_TOKEN*1000:.1f}ms ({1/cfg.D_SLA_TOKEN:.0f} tokens/sec)")
    print(f"  Per-Request SLA (D_SLA_REQUEST): {cfg.D_SLA_REQUEST*1000:.0f}ms")
    print(f"\n  Industry Reference: Gemini 2.5 Flash-Lite")
    print(f"    - TTFT: 240ms (cloud-optimized)")
    print(f"    - Output speed: 410 tokens/sec = 2.44ms/token (cloud-optimized)")
    print(f"\n  Recommended Ranges for RTX 4080:")
    print(f"    - Per-Token:   20-100ms (10-50 tokens/sec)")  
    print(f"    - Per-Request: 2-10s (depends on load)")
    print(f"    - Target violation rate: 5-15%")
    
    # Generate workload
    print(f"\n[Loading Workload]")
    workload = generate_workload(cfg)
    
    # Run simulation
    print(f"\n[Running Simulation]")
    sim = Simulator(cfg, workload, scheduler_type='multi_bin_dynamic')
    completed = sim.run()
    
    # Compute metrics with dual SLA thresholds
    metrics = compute_metrics(
        completed, 
        d_sla_token=cfg.D_SLA_TOKEN, 
        d_sla_request=cfg.D_SLA_REQUEST
    )
    
    # Get GPU utilization
    gpu_stats = sim.get_gpu_stats()
    gpu_metrics = compute_gpu_utilization(gpu_stats)
    
    # Print results
    print("\n" + "=" * 80)
    print("DUAL SLA RESULTS")
    print("=" * 80)
    
    print(f"\n[Per-Token SLA: D_SLA_TOKEN = {cfg.D_SLA_TOKEN*1000:.1f}ms]")
    print(f"  Purpose: Streaming UX (smooth token delivery)")
    print(f"  Target:  {1/cfg.D_SLA_TOKEN:.0f} tokens/sec (vs. Gemini Flash-Lite: 410 tokens/sec)")
    print(f"  -" * 40)
    print(f"  Violation Rate:    {metrics['sla_violation_rate_token']*100:.2f}%")
    print(f"  Violations:        {metrics['sla_violations_token']:,} / {metrics['num_requests']:,}")
    print(f"  Avg TBT:           {metrics['avg_tbt']*1000:.3f}ms ({1/metrics['avg_tbt']:.1f} tokens/sec)" if metrics['avg_tbt'] > 0 else "  Avg TBT:           N/A")
    print(f"  P50 TBT:           {metrics['p50_tbt']*1000:.3f}ms" if metrics['p50_tbt'] > 0 else "  P50 TBT:           N/A")
    print(f"  P95 TBT:           {metrics['p95_tbt']*1000:.3f}ms" if metrics['p95_tbt'] > 0 else "  P95 TBT:           N/A")
    print(f"  P99 TBT:           {metrics['p99_tbt']*1000:.3f}ms" if metrics['p99_tbt'] > 0 else "  P99 TBT:           N/A")
    
    print(f"\n[Per-Request SLA: D_SLA_REQUEST = {cfg.D_SLA_REQUEST*1000:.0f}ms]")
    print(f"  Purpose: Interactive response (total request time)")
    print(f"  Target:  {cfg.D_SLA_REQUEST*1000:.0f}ms (vs. Gemini Flash-Lite TTFT: 240ms)")
    print(f"  -" * 40)
    print(f"  Violation Rate:    {metrics['sla_violation_rate_request']*100:.2f}%")
    print(f"  Violations:        {metrics['sla_violations_request']:,} / {metrics['num_requests']:,}")
    print(f"  Avg Latency:       {metrics['avg_latency']*1000:.3f}ms")
    print(f"  P50 Latency:       {metrics['p50_latency']*1000:.3f}ms")
    print(f"  P95 Latency:       {metrics['p95_latency']*1000:.3f}ms")
    print(f"  P99 Latency:       {metrics['p99_latency']*1000:.3f}ms")
    
    print(f"\n[Throughput]")
    print(f"  Requests/sec:      {metrics['throughput_requests_per_sec']:.2f}")
    print(f"  Tokens/sec:        {metrics['throughput_tokens_per_sec']:.2f}")
    
    print(f"\n[GPU Utilization]")
    print(f"  Average:           {gpu_metrics['avg_utilization']*100:.2f}%")
    
    # Summary interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    # Per-token interpretation
    if metrics['sla_violation_rate_token'] < 0.05:
        token_status = "✅ EXCELLENT - Well under 5% target"
    elif metrics['sla_violation_rate_token'] < 0.10:
        token_status = "⚠️ GOOD - Within 10% acceptable range"
    elif metrics['sla_violation_rate_token'] < 0.15:
        token_status = "⚠️ ACCEPTABLE - Within 15% research target"
    else:
        token_status = "❌ HIGH - Consider increasing GPUs or D_SLA_TOKEN"
    
    print(f"\n  Per-Token SLA:     {token_status}")
    
    # Per-request interpretation  
    if metrics['sla_violation_rate_request'] < 0.05:
        request_status = "✅ EXCELLENT - Well under 5% target"
    elif metrics['sla_violation_rate_request'] < 0.10:
        request_status = "⚠️ GOOD - Within 10% acceptable range"
    elif metrics['sla_violation_rate_request'] < 0.15:
        request_status = "⚠️ ACCEPTABLE - Within 15% research target"
    else:
        request_status = "❌ HIGH - Consider increasing GPUs or D_SLA_REQUEST"
    
    print(f"  Per-Request SLA:   {request_status}")
    
    # Overall recommendation
    print("\n[Recommendations]")
    if metrics['sla_violation_rate_token'] > 0.15:
        print(f"  • Increase D_SLA_TOKEN from {cfg.D_SLA_TOKEN*1000:.1f}ms to {cfg.D_SLA_TOKEN*2*1000:.1f}ms for production")
    if metrics['sla_violation_rate_request'] > 0.15:
        print(f"  • Increase D_SLA_REQUEST from {cfg.D_SLA_REQUEST*1000:.0f}ms to {cfg.D_SLA_REQUEST*2*1000:.0f}ms for production")
    if metrics['sla_violation_rate_token'] > 0.15 or metrics['sla_violation_rate_request'] > 0.15:
        print(f"  • Add more GPUs: current {num_gpus} → {num_gpus * 2} for better throughput")
    if metrics['sla_violation_rate_token'] < 0.05 and metrics['sla_violation_rate_request'] < 0.05:
        print(f"  • System performing well! Could potentially reduce GPUs from {num_gpus} to optimize cost")
    
    print("\n" + "=" * 80)
    
    return metrics, gpu_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Dual SLA Monitoring")
    parser.add_argument("--requests", type=int, default=10000, help="Number of requests")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs")
    parser.add_argument("--real-timestamps", action="store_true", help="Use real timestamps")
    parser.add_argument("--rps-scaling", type=float, default=200.0, help="RPS scaling factor")
    
    args = parser.parse_args()
    
    run_dual_sla_test(
        num_requests=args.requests,
        num_gpus=args.gpus,
        use_real_timestamps=args.real_timestamps,
        rps_scaling=args.rps_scaling,
    )
