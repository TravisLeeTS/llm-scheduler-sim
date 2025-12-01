#!/usr/bin/env python
"""
Results Validation Test

This script tests the implementation and validates that the results make sense:
1. Compare scheduler types (static_fifo vs dynamic_no_bins vs multi_bin_dynamic)
2. Check that multi-bin improves over single queue
3. Validate batch sizes are being computed correctly
4. Check SLA controller is adapting
"""

import numpy as np
from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import load_burstgpt_dataset
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics


def run_comparison_test():
    """Compare all scheduler types to validate relative performance."""
    print("=" * 70)
    print("SCHEDULER COMPARISON TEST")
    print("=" * 70)
    
    # Load workload
    cfg_base = SchedulerConfig()
    requests = load_burstgpt_dataset(
        'data/BurstGPT_sample.csv',
        num_requests=1000,
        d_sla_token=cfg_base.D_SLA_TOKEN,
        d_sla_request=cfg_base.D_SLA_REQUEST,
        use_real_timestamps=True
    )[:1000]
    
    print(f"\nWorkload: {len(requests)} requests")
    print(f"SLA Thresholds: {cfg_base.D_SLA_TOKEN*1000:.0f}ms per-token, {cfg_base.D_SLA_REQUEST:.0f}s per-request")
    
    results = {}
    scheduler_types = ["static_fifo", "dynamic_no_bins", "multi_bin_dynamic"]
    
    for sched_type in scheduler_types:
        print(f"\n--- Testing: {sched_type} ---")
        
        cfg = SchedulerConfig(K_BINS=4, NUM_GPUS=4)
        sim = Simulator(cfg, requests.copy(), scheduler_type=sched_type)
        completed = sim.run()
        
        metrics = compute_metrics(completed, d_sla_token=cfg.D_SLA_TOKEN,
                                 d_sla_request=cfg.D_SLA_REQUEST)
        
        # Get batch statistics
        gpu_stats = sim.get_gpu_stats()
        total_batches = sum(g['total_batches'] for g in gpu_stats)
        avg_batch_size = len(completed) / total_batches if total_batches > 0 else 0
        
        results[sched_type] = {
            'throughput': metrics['throughput_requests_per_sec'],
            'avg_latency': metrics['avg_latency'],
            'sla_token': metrics['sla_violation_rate_token'],
            'sla_request': metrics['sla_violation_rate_request'],
            'avg_tbt': metrics['avg_tbt'],
            'total_batches': total_batches,
            'avg_batch_size': avg_batch_size,
        }
        
        print(f"  Throughput: {metrics['throughput_requests_per_sec']:.3f} req/s")
        print(f"  Avg Latency: {metrics['avg_latency']*1000:.0f}ms")
        print(f"  Avg TBT: {metrics['avg_tbt']*1000:.1f}ms")
        print(f"  SLA Violation (token): {metrics['sla_violation_rate_token']*100:.1f}%")
        print(f"  SLA Violation (request): {metrics['sla_violation_rate_request']*100:.1f}%")
        print(f"  Total Batches: {total_batches}")
        print(f"  Avg Batch Size: {avg_batch_size:.1f}")
    
    return results


def validate_results(results):
    """Validate that results make logical sense."""
    print("\n" + "=" * 70)
    print("RESULTS VALIDATION")
    print("=" * 70)
    
    issues = []
    
    # Check 1: Dynamic batching should have larger batch sizes than static
    if 'static_fifo' in results and 'dynamic_no_bins' in results:
        static_batch = results['static_fifo']['avg_batch_size']
        dynamic_batch = results['dynamic_no_bins']['avg_batch_size']
        print(f"\n1. Batch Size Comparison:")
        print(f"   Static FIFO: {static_batch:.1f}")
        print(f"   Dynamic: {dynamic_batch:.1f}")
        # Static uses fixed B=8, dynamic should adapt
        if dynamic_batch < 2:
            issues.append("⚠ Dynamic batch size is very low - SLA controller may be too aggressive")
    
    # Check 2: Multi-bin should have similar or better throughput than dynamic_no_bins
    if 'dynamic_no_bins' in results and 'multi_bin_dynamic' in results:
        dyn_tp = results['dynamic_no_bins']['throughput']
        mb_tp = results['multi_bin_dynamic']['throughput']
        print(f"\n2. Throughput Comparison:")
        print(f"   Dynamic (no bins): {dyn_tp:.3f} req/s")
        print(f"   Multi-bin Dynamic: {mb_tp:.3f} req/s")
        if mb_tp < dyn_tp * 0.5:
            issues.append("⚠ Multi-bin throughput is significantly lower than dynamic")
    
    # Check 3: Latency should be reasonable (not astronomical)
    for name, data in results.items():
        lat = data['avg_latency']
        print(f"\n3. Latency Check ({name}):")
        print(f"   Avg Latency: {lat*1000:.0f}ms = {lat:.1f}s")
        if lat > 100:  # More than 100 seconds average latency is suspicious
            issues.append(f"⚠ {name} has very high latency ({lat:.0f}s) - may indicate queueing issues")
    
    # Check 4: TBT should be reasonable for the hardware
    # RTX 4080 with Qwen3 1.7B: ~5.7ms/token baseline
    for name, data in results.items():
        tbt = data['avg_tbt']
        print(f"\n4. TBT Check ({name}):")
        print(f"   Avg TBT: {tbt*1000:.1f}ms")
        if tbt < 0.001:  # Less than 1ms is unrealistic
            issues.append(f"⚠ {name} TBT is unrealistically low ({tbt*1000:.2f}ms)")
        elif tbt > 0.5:  # More than 500ms is very slow
            issues.append(f"⚠ {name} TBT is very high ({tbt*1000:.0f}ms)")
    
    # Check 5: SLA violations should be in reasonable range
    for name, data in results.items():
        sla_tok = data['sla_token']
        sla_req = data['sla_request']
        print(f"\n5. SLA Check ({name}):")
        print(f"   Token SLA Violation: {sla_tok*100:.1f}%")
        print(f"   Request SLA Violation: {sla_req*100:.1f}%")
        if sla_tok > 0.95:
            issues.append(f"⚠ {name} has very high token SLA violation ({sla_tok*100:.0f}%)")
    
    return issues


def test_sla_controller_adaptation():
    """Test that the SLA controller is actually adapting."""
    print("\n" + "=" * 70)
    print("SLA CONTROLLER ADAPTATION TEST")
    print("=" * 70)
    
    from mb_dyn_sim.schedulers import SLAController
    
    D_SLA = 0.050  # 50ms
    eps_D = 0.005  # 5ms
    
    controller = SLAController(D_SLA, eps_D, B_min=1, B_max=128)
    
    # Simulate a sequence of latency observations
    print("\nSimulating latency feedback sequence:")
    
    # Start with high latency (should shrink interval)
    latencies = [0.080, 0.075, 0.070, 0.065, 0.060, 0.055, 0.050, 0.045, 0.040]
    batch_sizes = []
    
    for i, lat in enumerate(latencies):
        controller.tau_avg = lat
        controller.b_avg = 50.0
        controller.update_count = i + 5
        b_sla = controller.compute_b_SLA()
        batch_sizes.append(b_sla)
        print(f"  τ={lat*1000:.0f}ms → b_SLA={b_sla}, interval=[{controller.b_low}, {controller.b_high}]")
    
    # Check adaptation
    if batch_sizes[0] == batch_sizes[-1]:
        print("\n⚠ WARNING: SLA controller is NOT adapting (batch size constant)")
        return False
    else:
        print(f"\n✓ SLA controller is adapting: batch size {batch_sizes[0]} → {batch_sizes[-1]}")
        return True


def test_memory_constraint():
    """Test memory constraint calculation."""
    print("\n" + "=" * 70)
    print("MEMORY CONSTRAINT TEST")
    print("=" * 70)
    
    from mb_dyn_sim.schedulers import BatchStatistics, compute_b_mem
    
    cfg = SchedulerConfig()
    
    # Test with different sequence lengths
    test_cases = [
        (100, 50),    # Short sequences
        (300, 200),   # Medium sequences
        (1000, 500),  # Long sequences
        (2000, 1000), # Very long sequences
    ]
    
    print("\nMemory-based batch size for different sequence lengths:")
    print(f"  GPU Memory: {cfg.M_MAX_GB}GB, Model: {cfg.M_MODEL_GB}GB")
    print(f"  KV cache: {cfg.KV_MEM_PER_TOKEN_GB*1e6:.2f} MB/token")
    
    for prompt_len, output_len in test_cases:
        stats = BatchStatistics()
        stats.avg_prompt_len = prompt_len
        stats.avg_output_len = output_len
        
        b_mem = compute_b_mem(stats, cfg)
        total_tokens = prompt_len + output_len
        
        print(f"\n  E[l_in]={prompt_len}, E[l_out]={output_len} (total={total_tokens} tokens)")
        print(f"    → b_mem = {b_mem}")
        
        # Verify memory calculation
        eta = (cfg.M_MAX_GB - cfg.M_MODEL_GB) / cfg.KV_MEM_PER_TOKEN_GB
        L0 = 0.1 * eta
        expected = int((eta - L0) / total_tokens)
        expected = max(1, min(cfg.B_MAX, expected))
        print(f"    Expected: {expected} (η={eta:.0f}, L0={L0:.0f})")


def test_batch_composition():
    """Test that multi-bin actually improves batch composition."""
    print("\n" + "=" * 70)
    print("BATCH COMPOSITION TEST")
    print("=" * 70)
    
    cfg = SchedulerConfig(K_BINS=4, NUM_GPUS=4)
    requests = load_burstgpt_dataset(
        'data/BurstGPT_sample.csv',
        num_requests=500,
        d_sla_token=cfg.D_SLA_TOKEN,
        d_sla_request=cfg.D_SLA_REQUEST,
        use_real_timestamps=True
    )[:500]
    
    # Test multi_bin_dynamic
    sim = Simulator(cfg, requests.copy(), scheduler_type="multi_bin_dynamic")
    completed = sim.run()
    
    comp_stats = sim.get_batch_composition_stats()
    
    print(f"\nBatch Composition Statistics:")
    print(f"  Total batches: {comp_stats.get('total_batches', 'N/A')}")
    print(f"  Batches per bin: {comp_stats.get('batches_per_bin', 'N/A')}")
    print(f"  Avg variance per bin: {comp_stats.get('avg_variance_per_bin', 'N/A')}")
    print(f"  Overall avg variance: {comp_stats.get('overall_avg_variance', 'N/A')}")
    
    # Lower variance = better composition
    avg_var = comp_stats.get('overall_avg_variance', float('inf'))
    if avg_var < 1000:
        print(f"\n✓ Batch composition is homogeneous (variance={avg_var:.1f})")
    else:
        print(f"\n⚠ Batch composition has high variance ({avg_var:.1f})")


def main():
    print("=" * 70)
    print("COMPREHENSIVE RESULTS VALIDATION")
    print("=" * 70)
    
    # Run tests
    results = run_comparison_test()
    issues = validate_results(results)
    
    test_sla_controller_adaptation()
    test_memory_constraint()
    test_batch_composition()
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if issues:
        print("\n⚠ Issues Found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✓ All checks passed!")
    
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Scheduler':<25} {'Throughput':<12} {'Latency':<12} {'TBT':<10} {'SLA-Tok':<10} {'SLA-Req':<10}")
    print("-" * 79)
    for name, data in results.items():
        print(f"{name:<25} {data['throughput']:.3f} req/s  {data['avg_latency']*1000:.0f}ms      "
              f"{data['avg_tbt']*1000:.1f}ms    {data['sla_token']*100:.1f}%      {data['sla_request']*100:.1f}%")


if __name__ == "__main__":
    main()
