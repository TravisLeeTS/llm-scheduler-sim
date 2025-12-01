#!/usr/bin/env python
"""
Paper-Faithful Validation Test

This script validates that the implementation follows the algorithms from:
1. SLA-constrained Dynamic Batching (Pang et al.) - Algorithms 1 & 2
2. Multi-Bin Batching (Guldogan et al.) - Algorithm 1

Key Invariants to Verify:
1. Algorithm 1 (Memory): b_mem = floor((η - L₀) / E[l_in + l_out])
2. Algorithm 2 (SLA): Adaptive interval [b_low, b_high] with three cases
3. Final batch size: b* = min(b_mem, b_SLA)
4. Multi-Bin: Requests assigned to bins by predicted length, FIFO within bins
5. Per-GPU state: Each GPU maintains its own controller state
6. Round-robin: Simple rotating assignment of batches to GPUs
"""

import numpy as np
from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import load_burstgpt_dataset, Request
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.schedulers import (
    SLAController, BatchStatistics, compute_b_mem,
    MultiBinScheduler, DynamicBatcher
)
from mb_dyn_sim.metrics import compute_metrics


def test_algorithm_1_memory_constraint():
    """Test Algorithm 1: Memory-constrained batch size computation."""
    print("\n" + "=" * 60)
    print("Test: Algorithm 1 (Memory-Constrained Batch Size)")
    print("=" * 60)
    
    cfg = SchedulerConfig()
    stats = BatchStatistics(bin_idx=-1)
    stats.avg_prompt_len = 300.0
    stats.avg_output_len = 200.0
    
    # Compute expected values manually
    eta = (cfg.M_MAX_GB - cfg.M_MODEL_GB) / cfg.KV_MEM_PER_TOKEN_GB
    E_l_total = stats.avg_prompt_len + stats.avg_output_len
    L0 = 0.1 * eta  # Safety buffer
    expected_b_mem = int((eta - L0) / E_l_total)
    
    # Call the function
    actual_b_mem = compute_b_mem(stats, cfg, bin_idx=-1)
    
    print(f"  η (token capacity): {eta:.0f} tokens")
    print(f"  E[l_in + l_out]: {E_l_total:.0f} tokens")
    print(f"  L₀ (safety buffer): {L0:.0f} tokens")
    print(f"  Expected b_mem: {expected_b_mem}")
    print(f"  Actual b_mem: {actual_b_mem}")
    print(f"  Match: {'✓ PASS' if abs(expected_b_mem - actual_b_mem) <= 1 else '✗ FAIL'}")
    
    # Test per-bin B_MAX constraint
    cfg.BIN_B_MAX = [32, 16, 8, 4]
    for bin_idx, expected_max in enumerate(cfg.BIN_B_MAX):
        b_mem_bin = compute_b_mem(stats, cfg, bin_idx=bin_idx)
        print(f"  Bin {bin_idx}: b_mem={b_mem_bin}, B_MAX[{bin_idx}]={expected_max}, "
              f"Respects limit: {'✓' if b_mem_bin <= expected_max else '✗'}")


def test_algorithm_2_sla_controller():
    """Test Algorithm 2: SLA-constrained adaptive batch sizing."""
    print("\n" + "=" * 60)
    print("Test: Algorithm 2 (SLA-Constrained Adaptive Batch Size)")
    print("=" * 60)
    
    D_SLA = 0.050  # 50ms target TBT
    eps_D = 0.005  # 5ms tolerance
    B_min, B_max = 1, 128
    
    controller = SLAController(D_SLA, eps_D, B_min, B_max)
    
    # Test Case 1: Latency too high (τ_avg > D_SLA + eps_D)
    print("\n  Case 1: Latency too high (shrink interval)")
    controller.tau_avg = 0.080  # 80ms > 55ms threshold
    controller.b_avg = 64.0
    controller.update_count = 10
    old_low, old_high = controller.b_low, controller.b_high
    b_sla = controller.compute_b_SLA()
    print(f"    τ_avg={controller.tau_avg*1000:.0f}ms (> D_SLA+ε={D_SLA*1000 + eps_D*1000:.0f}ms)")
    print(f"    Interval: [{old_low}, {old_high}] → [{controller.b_low}, {controller.b_high}]")
    print(f"    b_SLA = {b_sla}")
    print(f"    Interval shrunk: {'✓ PASS' if controller.b_high <= old_high else '✗ FAIL'}")
    
    # Test Case 2: Latency too low (τ_avg < D_SLA - eps_D)
    print("\n  Case 2: Latency comfortably below SLA (expand interval)")
    controller = SLAController(D_SLA, eps_D, B_min, B_max)
    controller.tau_avg = 0.030  # 30ms < 45ms threshold
    controller.b_avg = 64.0
    controller.update_count = 10
    old_low, old_high = controller.b_low, controller.b_high
    b_sla = controller.compute_b_SLA()
    print(f"    τ_avg={controller.tau_avg*1000:.0f}ms (< D_SLA-ε={D_SLA*1000 - eps_D*1000:.0f}ms)")
    print(f"    Interval: [{old_low}, {old_high}] → [{controller.b_low}, {controller.b_high}]")
    print(f"    b_SLA = {b_sla}")
    print(f"    Interval expanded: {'✓ PASS' if controller.b_low >= old_low else '✗ FAIL'}")
    
    # Test Case 3: Latency within SLA band
    print("\n  Case 3: Latency within SLA band (center interval)")
    controller = SLAController(D_SLA, eps_D, B_min, B_max)
    controller.tau_avg = 0.050  # 50ms = D_SLA exactly
    controller.b_avg = 64.0
    controller.update_count = 10
    b_sla = controller.compute_b_SLA()
    print(f"    τ_avg={controller.tau_avg*1000:.0f}ms (≈ D_SLA={D_SLA*1000:.0f}ms)")
    print(f"    Interval: [{controller.b_low}, {controller.b_high}]")
    print(f"    b_SLA = {b_sla}")
    print(f"    Centered around b_avg: {'✓ PASS' if abs(b_sla - 64) < 10 else '✗ FAIL'}")


def test_final_batch_size():
    """Test: b* = min(b_mem, b_SLA)"""
    print("\n" + "=" * 60)
    print("Test: Final Batch Size b* = min(b_mem, b_SLA)")
    print("=" * 60)
    
    cfg = SchedulerConfig()
    
    def mock_service_time(batch_size, max_seq_len):
        return 0.05 * batch_size + 0.001 * max_seq_len
    
    batcher = DynamicBatcher(cfg, mock_service_time)
    
    # Create test candidates
    candidates = [
        Request(id=i, arrival_time=i*0.1, prompt_len=100, output_len=50,
                predicted_output_len=50, deadline=10.0, deadline_request=10.0)
        for i in range(100)
    ]
    
    batch, service_time = batcher.make_batch(now=0.0, candidates=candidates, bin_idx=-1)
    
    # Get the computed values
    stats = batcher.global_stats
    sla_controller = batcher.global_sla_controller
    b_mem = compute_b_mem(stats, cfg)
    b_sla = sla_controller.compute_b_SLA()
    b_target = min(b_mem, b_sla)
    
    print(f"  b_mem (Algorithm 1): {b_mem}")
    print(f"  b_SLA (Algorithm 2): {b_sla}")
    print(f"  b* = min(b_mem, b_SLA): {b_target}")
    print(f"  Actual batch size: {len(batch)}")
    print(f"  Correct: {'✓ PASS' if len(batch) <= b_target else '✗ FAIL'}")


def test_multi_bin_assignment():
    """Test Multi-Bin paper: requests assigned to bins by predicted length."""
    print("\n" + "=" * 60)
    print("Test: Multi-Bin Request Assignment")
    print("=" * 60)
    
    cfg = SchedulerConfig(K_BINS=4)
    cfg.BIN_BOUNDARIES = [(0, 50), (50, 100), (100, 200), (200, 10000)]
    
    scheduler = MultiBinScheduler(cfg)
    
    # Create requests with different predicted lengths
    test_cases = [
        (10, 0),   # Short → Bin 0
        (75, 1),   # Medium-short → Bin 1
        (150, 2),  # Medium-long → Bin 2
        (500, 3),  # Long → Bin 3
    ]
    
    all_pass = True
    for predicted_len, expected_bin in test_cases:
        req = Request(id=predicted_len, arrival_time=0, prompt_len=100, 
                     output_len=predicted_len, predicted_output_len=predicted_len,
                     deadline=10.0, deadline_request=10.0)
        scheduler.enqueue_request(req)
        actual_bin = scheduler._select_bin(predicted_len)
        
        passed = actual_bin == expected_bin
        all_pass = all_pass and passed
        print(f"  predicted_len={predicted_len}: Bin {actual_bin} "
              f"(expected {expected_bin}) {'✓' if passed else '✗'}")
    
    print(f"\n  Multi-Bin Assignment: {'✓ PASS' if all_pass else '✗ FAIL'}")


def test_per_bin_statistics():
    """Test that each bin maintains its own statistics."""
    print("\n" + "=" * 60)
    print("Test: Per-Bin Statistics (Independent Controllers)")
    print("=" * 60)
    
    cfg = SchedulerConfig(K_BINS=4)
    
    def mock_service_time(batch_size, max_seq_len):
        return 0.01 * max_seq_len
    
    batcher = DynamicBatcher(cfg, mock_service_time)
    
    # Simulate different latencies for different bins
    # Short requests (bin 0) → low latency
    # Long requests (bin 3) → high latency
    
    short_batch = [
        Request(id=i, arrival_time=0, prompt_len=50, output_len=20,
                predicted_output_len=20, deadline=10.0, deadline_request=10.0)
        for i in range(10)
    ]
    long_batch = [
        Request(id=i+100, arrival_time=0, prompt_len=200, output_len=300,
                predicted_output_len=300, deadline=10.0, deadline_request=10.0)
        for i in range(10)
    ]
    
    # Update statistics for bin 0 (short) and bin 3 (long)
    batcher.update_after_batch(short_batch, service_time=0.2, bin_idx=0)
    batcher.update_after_batch(long_batch, service_time=3.0, bin_idx=3)
    
    print(f"  Bin 0 (short) stats:")
    print(f"    avg_prompt_len: {batcher.bin_stats[0].avg_prompt_len:.1f}")
    print(f"    avg_output_len: {batcher.bin_stats[0].avg_output_len:.1f}")
    print(f"    SLA controller τ_avg: {batcher.bin_sla_controllers[0].tau_avg*1000:.1f}ms")
    
    print(f"  Bin 3 (long) stats:")
    print(f"    avg_prompt_len: {batcher.bin_stats[3].avg_prompt_len:.1f}")
    print(f"    avg_output_len: {batcher.bin_stats[3].avg_output_len:.1f}")
    print(f"    SLA controller τ_avg: {batcher.bin_sla_controllers[3].tau_avg*1000:.1f}ms")
    
    # Check independence
    independent = (batcher.bin_stats[0].avg_output_len != batcher.bin_stats[3].avg_output_len)
    print(f"\n  Statistics independent: {'✓ PASS' if independent else '✗ FAIL'}")


def test_fifo_within_bins():
    """Test FIFO ordering within each bin."""
    print("\n" + "=" * 60)
    print("Test: FIFO Ordering Within Bins")
    print("=" * 60)
    
    cfg = SchedulerConfig(K_BINS=2)
    cfg.BIN_BOUNDARIES = [(0, 100), (100, 10000)]
    
    scheduler = MultiBinScheduler(cfg)
    
    # Add requests in specific order
    requests = [
        Request(id=1, arrival_time=0.1, prompt_len=50, output_len=30,
                predicted_output_len=30, deadline=10.0, deadline_request=10.0),
        Request(id=2, arrival_time=0.2, prompt_len=50, output_len=30,
                predicted_output_len=30, deadline=10.0, deadline_request=10.0),
        Request(id=3, arrival_time=0.3, prompt_len=50, output_len=30,
                predicted_output_len=30, deadline=10.0, deadline_request=10.0),
    ]
    
    for req in requests:
        scheduler.enqueue_request(req)
    
    # Get candidates - should be in FIFO order
    candidates, bin_idx = scheduler.get_candidates_for_gpu(0, max_candidates=10)
    
    fifo_correct = all(candidates[i].id < candidates[i+1].id 
                       for i in range(len(candidates)-1))
    
    print(f"  Enqueued order: [1, 2, 3]")
    print(f"  Retrieved order: {[r.id for r in candidates]}")
    print(f"  FIFO preserved: {'✓ PASS' if fifo_correct else '✗ FAIL'}")


def test_round_robin_gpu_assignment():
    """Test round-robin assignment of batches to GPUs."""
    print("\n" + "=" * 60)
    print("Test: Round-Robin GPU Assignment")
    print("=" * 60)
    
    cfg = SchedulerConfig(K_BINS=2, NUM_GPUS=4)
    cfg.BIN_BOUNDARIES = [(0, 100), (100, 10000)]
    
    scheduler = MultiBinScheduler(cfg)
    
    # Add many requests to bin 0
    for i in range(20):
        req = Request(id=i, arrival_time=i*0.1, prompt_len=50, output_len=30,
                     predicted_output_len=30, deadline=10.0, deadline_request=10.0)
        scheduler.enqueue_request(req)
    
    # Track which GPUs get work
    gpu_assignments = []
    for gpu_id in range(4):
        candidates, bin_idx = scheduler.get_candidates_for_gpu(gpu_id, max_candidates=4)
        if candidates:
            gpu_assignments.append(gpu_id)
    
    print(f"  GPUs that received work: {gpu_assignments}")
    # In round-robin, all GPUs should get work if there's enough
    print(f"  Work distributed: {'✓ PASS' if len(set(gpu_assignments)) > 1 else '⚠ Check'}")


def test_full_integration():
    """Full integration test with BurstGPT workload."""
    print("\n" + "=" * 60)
    print("Test: Full Integration (BurstGPT + Multi-Bin + Dynamic Batching)")
    print("=" * 60)
    
    cfg = SchedulerConfig(K_BINS=4, NUM_GPUS=4)
    
    # Load real workload
    requests = load_burstgpt_dataset(
        'data/BurstGPT_sample.csv',
        num_requests=500,
        d_sla_token=cfg.D_SLA_TOKEN,
        d_sla_request=cfg.D_SLA_REQUEST,
        use_real_timestamps=True
    )[:500]
    
    # Run simulation
    sim = Simulator(cfg, requests, scheduler_type="multi_bin_dynamic")
    completed = sim.run()
    
    # Compute metrics
    metrics = compute_metrics(completed, d_sla_token=cfg.D_SLA_TOKEN, 
                             d_sla_request=cfg.D_SLA_REQUEST)
    
    print(f"\n  Workload: {len(requests)} requests")
    print(f"  Completed: {len(completed)} requests")
    print(f"  Throughput: {metrics['throughput_requests_per_sec']:.2f} req/s")
    print(f"  Avg Latency: {metrics['avg_latency']*1000:.0f}ms")
    print(f"  Avg TBT: {metrics['avg_tbt']*1000:.1f}ms")
    print(f"  SLA Violation (per-token): {metrics['sla_violation_rate_token']*100:.1f}%")
    print(f"  SLA Violation (per-request): {metrics['sla_violation_rate_request']*100:.1f}%")
    
    # Get batch composition stats
    comp_stats = sim.get_batch_composition_stats()
    if comp_stats:
        print(f"\n  Batch Composition (Multi-Bin Contribution):")
        print(f"    Total batches: {comp_stats.get('total_batches', 'N/A')}")
        print(f"    Avg variance per bin: {comp_stats.get('overall_avg_variance', 'N/A'):.1f}")
    
    print(f"\n  Integration Test: {'✓ PASS' if len(completed) == len(requests) else '✗ FAIL'}")


if __name__ == "__main__":
    print("=" * 60)
    print("PAPER-FAITHFUL VALIDATION TEST SUITE")
    print("=" * 60)
    print("\nValidating implementation against:")
    print("  1. SLA-constrained Dynamic Batching (Pang et al.)")
    print("  2. Multi-Bin Batching (Guldogan et al.)")
    
    test_algorithm_1_memory_constraint()
    test_algorithm_2_sla_controller()
    test_final_batch_size()
    test_multi_bin_assignment()
    test_per_bin_statistics()
    test_fifo_within_bins()
    test_round_robin_gpu_assignment()
    test_full_integration()
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
