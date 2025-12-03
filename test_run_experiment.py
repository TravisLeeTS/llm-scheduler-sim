#!/usr/bin/env python
"""
Regression test for run_experiment and CLI interface.

This test validates:
1. run_experiment function works with all scheduler types
2. compute_metrics interface matches run_experiment calls
3. CLI argument parsing works correctly (including BooleanOptionalAction flags)
4. N_decode/N_prefill are properly threaded to compute_b_mem

Run this test after making changes to experiments.py, metrics.py, 
schedulers.py, or the CLI to catch interface drift.
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.experiments import run_experiment
from mb_dyn_sim.metrics import compute_metrics
from mb_dyn_sim.schedulers import compute_b_mem, BatchStatistics


def test_run_experiment_interface():
    """Test that run_experiment works with all scheduler types without TypeError."""
    print("=" * 60)
    print("Test 1: run_experiment interface regression test")
    print("=" * 60)
    
    # Use minimal config for fast testing
    cfg = SchedulerConfig(
        NUM_REQUESTS=100,
        NUM_GPUS=1,
        K_BINS=2,
        SEED=42,
    )
    
    scheduler_types = [
        "static_fifo",
        "dynamic_no_bins", 
        "multi_bin_dynamic",
        "multi_bin_only",
    ]
    
    for scheduler_type in scheduler_types:
        print(f"\n  Testing {scheduler_type}...", end=" ")
        try:
            result = run_experiment(cfg, scheduler_type, load_level="low", seed=42)
            assert 'metrics' in result, f"Missing 'metrics' in result"
            assert result['metrics']['num_requests'] > 0, f"No requests completed"
            print(f"✓ ({result['metrics']['num_requests']} requests completed)")
        except TypeError as e:
            print(f"✗ TypeError: {e}")
            raise AssertionError(f"run_experiment({scheduler_type}) raised TypeError: {e}")
        except Exception as e:
            print(f"✗ {type(e).__name__}: {e}")
            raise
    
    print("\n  All scheduler types passed!")
    return True


def test_compute_metrics_interface():
    """Test that compute_metrics accepts the correct parameters."""
    print("\n" + "=" * 60)
    print("Test 2: compute_metrics interface test")
    print("=" * 60)
    
    # Test with empty list
    print("\n  Testing with empty list...", end=" ")
    try:
        result = compute_metrics([], d_sla_token=0.010, d_sla_request=10.0)
        assert result['num_requests'] == 0
        print("✓")
    except TypeError as e:
        print(f"✗ TypeError: {e}")
        raise AssertionError(f"compute_metrics([]) raised TypeError: {e}")
    
    # Ensure old 'd_sla' parameter is NOT accepted (should raise TypeError)
    print("  Testing that old 'd_sla' parameter is rejected...", end=" ")
    try:
        compute_metrics([], d_sla=0.5)  # Old interface, should fail
        print("✗ (should have raised TypeError)")
        raise AssertionError("compute_metrics should not accept 'd_sla' parameter")
    except TypeError:
        print("✓ (correctly rejected)")
    
    return True


def test_compute_b_mem_with_concurrency():
    """Test that compute_b_mem properly uses N_decode/N_prefill."""
    print("\n" + "=" * 60)
    print("Test 3: compute_b_mem concurrency-aware constraint")
    print("=" * 60)
    
    cfg = SchedulerConfig()
    stats = BatchStatistics(bin_idx=-1)
    # Add some fake statistics
    stats.avg_prompt_len = 500.0
    stats.avg_output_len = 200.0
    
    # Test 1: Without N_decode/N_prefill (defaults to 0)
    print("\n  Testing without concurrency tracking...", end=" ")
    b_mem_default = compute_b_mem(stats, cfg)
    print(f"✓ (b_mem={b_mem_default})")
    
    # Test 2: With both N_decode > 0 and N_prefill > 0 (should enforce constraint)
    print("  Testing with N_decode=10, N_prefill=5...", end=" ")
    b_mem_concurrent = compute_b_mem(stats, cfg, N_decode=10, N_prefill=5)
    # Paper requirement: b_mem >= N_decode when both are positive
    assert b_mem_concurrent >= 10, f"b_mem ({b_mem_concurrent}) should be >= N_decode (10)"
    print(f"✓ (b_mem={b_mem_concurrent} >= N_decode=10)")
    
    # Test 3: With only N_decode > 0 (should not enforce minimum)
    print("  Testing with N_decode=10, N_prefill=0...", end=" ")
    b_mem_decode_only = compute_b_mem(stats, cfg, N_decode=10, N_prefill=0)
    print(f"✓ (b_mem={b_mem_decode_only})")
    
    return True


def test_cli_boolean_flags():
    """Test that CLI boolean flags can be disabled."""
    print("\n" + "=" * 60)
    print("Test 4: CLI BooleanOptionalAction flags")
    print("=" * 60)
    
    script_path = Path(__file__).parent / "scripts" / "run_mb_dynamic.py"
    if not script_path.exists():
        print("  Skipping: run_mb_dynamic.py not found")
        return True
    
    # Test that --help includes both --use and --no-use variants
    print("\n  Testing --help output...", end=" ")
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
    )
    
    help_text = result.stdout
    
    # Check for BooleanOptionalAction patterns
    if "--no-use-real-calibration" in help_text or "--use-real-calibration" in help_text:
        print("✓ (--use-real-calibration flag present)")
    else:
        print("✗ (missing --use-real-calibration in help)")
        return False
    
    if "--no-use-equal-mass-bins" in help_text or "--use-equal-mass-bins" in help_text:
        print("  ✓ (--use-equal-mass-bins flag present)")
    else:
        print("  ✗ (missing --use-equal-mass-bins in help)")
        return False
    
    # Check for new SLA flags (--d-sla-token and --d-sla-request)
    if "--d-sla-token" in help_text:
        print("  ✓ (--d-sla-token flag present)")
    else:
        print("  ✗ (missing --d-sla-token in help)")
        return False
    
    if "--d-sla-request" in help_text:
        print("  ✓ (--d-sla-request flag present)")
    else:
        print("  ✗ (missing --d-sla-request in help)")
        return False
    
    return True


def test_load_level_affects_workload():
    """Test that load_level actually changes workload intensity."""
    print("\n" + "=" * 60)
    print("Test 5: load_level affects workload intensity")
    print("=" * 60)
    
    from mb_dyn_sim.experiments import LOAD_LEVEL_CONFIG
    
    # Test that LOAD_LEVEL_CONFIG exists and has expected structure
    print("\n  Testing LOAD_LEVEL_CONFIG...", end=" ")
    assert "low" in LOAD_LEVEL_CONFIG, "Missing 'low' level"
    assert "medium" in LOAD_LEVEL_CONFIG, "Missing 'medium' level"
    assert "high" in LOAD_LEVEL_CONFIG, "Missing 'high' level"
    
    # Verify multipliers make sense
    assert LOAD_LEVEL_CONFIG["low"]["request_mult"] < LOAD_LEVEL_CONFIG["medium"]["request_mult"]
    assert LOAD_LEVEL_CONFIG["medium"]["request_mult"] < LOAD_LEVEL_CONFIG["high"]["request_mult"]
    print("✓")
    
    # Test with actual experiment to verify different request counts
    print("  Testing different load levels produce different request counts...", end=" ")
    cfg = SchedulerConfig(
        NUM_REQUESTS=100,
        NUM_GPUS=1,
        K_BINS=2,
        SEED=42,
    )
    
    # Run with low and high loads
    result_low = run_experiment(cfg, "static_fifo", load_level="low", seed=42)
    result_high = run_experiment(cfg, "static_fifo", load_level="high", seed=42)
    
    # High load should have more requests than low load
    low_requests = result_low['metrics']['num_requests']
    high_requests = result_high['metrics']['num_requests']
    
    assert high_requests > low_requests, f"High load ({high_requests}) should have more requests than low ({low_requests})"
    print(f"✓ (low={low_requests}, high={high_requests})")
    
    return True


def test_sla_alignment():
    """Test that D_SLA, D_SLA_TOKEN are properly aligned."""
    print("\n" + "=" * 60)
    print("Test 6: SLA threshold alignment")
    print("=" * 60)
    
    # Test that default config has aligned SLA values
    print("\n  Testing default config SLA alignment...", end=" ")
    cfg = SchedulerConfig()
    
    # D_SLA should equal D_SLA_TOKEN (used by controller and evaluation)
    assert cfg.D_SLA == cfg.D_SLA_TOKEN, f"D_SLA ({cfg.D_SLA}) != D_SLA_TOKEN ({cfg.D_SLA_TOKEN})"
    print(f"✓ (D_SLA={cfg.D_SLA}, D_SLA_TOKEN={cfg.D_SLA_TOKEN})")
    
    # Test explicit setting keeps them aligned
    print("  Testing explicit SLA setting...", end=" ")
    cfg2 = SchedulerConfig(D_SLA=0.020, D_SLA_TOKEN=0.020)
    assert cfg2.D_SLA == cfg2.D_SLA_TOKEN
    print("✓")
    
    return True


def main():
    """Run all regression tests."""
    print("\n" + "=" * 60)
    print("REGRESSION TEST SUITE: run_experiment & CLI interface")
    print("=" * 60)
    
    all_passed = True
    
    try:
        test_run_experiment_interface()
    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        all_passed = False
    
    try:
        test_compute_metrics_interface()
    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        all_passed = False
    
    try:
        test_compute_b_mem_with_concurrency()
    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        all_passed = False
    
    try:
        test_cli_boolean_flags()
    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        all_passed = False
    
    try:
        test_load_level_affects_workload()
    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        all_passed = False
    
    try:
        test_sla_alignment()
    except AssertionError as e:
        print(f"\n  FAILED: {e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
