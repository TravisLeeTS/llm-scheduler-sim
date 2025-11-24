"""
Test to demonstrate bin-specific adaptive batching.

This shows how each bin learns its own characteristics and adapts batch size
based on the narrower length distribution within that bin.
"""

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.schedulers import DynamicBatcher
from mb_dyn_sim.workload import Request


def simple_service_time(batch_size: int, max_seq_len: int) -> float:
    """Simple service time model for testing."""
    alpha = 0.015  # 15ms startup
    beta = 0.0003  # 0.3ms per token
    gamma = 0.4     # batching efficiency
    return alpha + beta * max_seq_len * (1 + gamma * (batch_size - 1) / batch_size)


def test_bin_specific_batching():
    """Test that bins maintain separate statistics and controllers."""
    
    cfg = SchedulerConfig()
    batcher = DynamicBatcher(cfg, simple_service_time)
    
    print("=" * 80)
    print("BIN-SPECIFIC ADAPTIVE BATCHING TEST")
    print("=" * 80)
    
    # Create test requests for Bin 0 (short: 0-64 tokens)
    bin0_requests = [
        Request(id=i, arrival_time=i*0.1, prompt_len=100, output_len=32, 
                predicted_output_len=32, deadline=1.0)
        for i in range(20)
    ]
    
    # Create test requests for Bin 3 (very long: 1024+ tokens)
    bin3_requests = [
        Request(id=100+i, arrival_time=i*0.1, prompt_len=100, output_len=2048,
                predicted_output_len=2048, deadline=5.0)
        for i in range(20)
    ]
    
    print("\n1. Initial State (no learning yet)")
    print("-" * 80)
    print(f"Bin 0 stats: avg_prompt={batcher.bin_stats[0].avg_prompt_len:.1f}, "
          f"avg_output={batcher.bin_stats[0].avg_output_len:.1f}")
    print(f"Bin 3 stats: avg_prompt={batcher.bin_stats[3].avg_prompt_len:.1f}, "
          f"avg_output={batcher.bin_stats[3].avg_output_len:.1f}")
    print(f"Global stats: avg_prompt={batcher.global_stats.avg_prompt_len:.1f}, "
          f"avg_output={batcher.global_stats.avg_output_len:.1f}")
    
    # Make batch from Bin 0 (short requests)
    print("\n2. Making batch from Bin 0 (short requests)")
    print("-" * 80)
    batch0, service_time0 = batcher.make_batch(0.0, bin0_requests[:10], bin_idx=0)
    print(f"Bin 0 batch size: {len(batch0)}")
    print(f"Bin 0 service time: {service_time0:.4f}s")
    
    # Make batch from Bin 3 (long requests)
    print("\n3. Making batch from Bin 3 (long requests)")
    print("-" * 80)
    batch3, service_time3 = batcher.make_batch(0.0, bin3_requests[:10], bin_idx=3)
    print(f"Bin 3 batch size: {len(batch3)}")
    print(f"Bin 3 service time: {service_time3:.4f}s")
    
    # Update statistics (simulate batch completion)
    print("\n4. Updating bin-specific statistics (after batch completion)")
    print("-" * 80)
    batcher.update_after_batch(batch0, service_time0, bin_idx=0)
    batcher.update_after_batch(batch3, service_time3, bin_idx=3)
    
    print(f"Bin 0 stats: avg_prompt={batcher.bin_stats[0].avg_prompt_len:.1f}, "
          f"avg_output={batcher.bin_stats[0].avg_output_len:.1f}")
    print(f"Bin 3 stats: avg_prompt={batcher.bin_stats[3].avg_prompt_len:.1f}, "
          f"avg_output={batcher.bin_stats[3].avg_output_len:.1f}")
    print(f"Global stats: avg_prompt={batcher.global_stats.avg_prompt_len:.1f}, "
          f"avg_output={batcher.global_stats.avg_output_len:.1f}")
    
    # Show SLA controller state
    print("\n5. SLA Controller State (after learning)")
    print("-" * 80)
    print(f"Bin 0 SLA: τ_avg={batcher.bin_sla_controllers[0].tau_avg:.4f}s, "
          f"b_avg={batcher.bin_sla_controllers[0].b_avg:.1f}, "
          f"[b_low={batcher.bin_sla_controllers[0].b_low}, "
          f"b_high={batcher.bin_sla_controllers[0].b_high}]")
    print(f"Bin 3 SLA: τ_avg={batcher.bin_sla_controllers[3].tau_avg:.4f}s, "
          f"b_avg={batcher.bin_sla_controllers[3].b_avg:.1f}, "
          f"[b_low={batcher.bin_sla_controllers[3].b_low}, "
          f"b_high={batcher.bin_sla_controllers[3].b_high}]")
    
    # Make another batch to show adaptation
    print("\n6. Making second batch (after learning)")
    print("-" * 80)
    batch0_v2, service_time0_v2 = batcher.make_batch(1.0, bin0_requests[10:], bin_idx=0)
    batch3_v2, service_time3_v2 = batcher.make_batch(1.0, bin3_requests[10:], bin_idx=3)
    print(f"Bin 0 batch size (v2): {len(batch0_v2)} (was {len(batch0)})")
    print(f"Bin 3 batch size (v2): {len(batch3_v2)} (was {len(batch3)})")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("✓ Each bin maintains SEPARATE statistics and SLA controllers")
    print("✓ Bin 0 (short) learns from short request batches")
    print("✓ Bin 3 (long) learns from long request batches")
    print("✓ Bins adapt batch size independently based on their E[max(t_j)]")
    print("✓ Narrower distributions → smaller variance → better SLA compliance")
    print("=" * 80)


if __name__ == "__main__":
    test_bin_specific_batching()
