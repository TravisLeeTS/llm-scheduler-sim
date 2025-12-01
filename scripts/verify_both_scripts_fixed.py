#!/usr/bin/env python3
"""
Quick verification that both scripts now handle K-bins correctly.
Tests both the original and optimized versions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("TESTING K-BINS FIX FOR BOTH SCRIPTS")
print("="*80)

# Test 1: Original script
print("\n1. Testing ORIGINAL script (comprehensive_stress_test.py)...")
print("-"*80)

from scripts.comprehensive_stress_test import run_single_test as run_original

result = run_original(
    num_requests=1000,
    num_gpus=2,
    scheduler_type="multi_bin_dynamic",
    use_real_timestamps=False,
    rps_scaling=200.0,
    d_sla=0.05,
    dataset_path="data/BurstGPT_sample.csv",
    calibration_csv="data/qwen3_1_7b_latency_grid.csv",
    k_bins=8  # This was failing before fix
)

if result['status'] == 'success':
    print(f"✓ ORIGINAL script K=8 PASSED")
else:
    print(f"✗ ORIGINAL script K=8 FAILED: {result.get('error', 'Unknown')}")

# Test 2: Optimized script
print("\n2. Testing OPTIMIZED script (comprehensive_stress_test_optimized.py)...")
print("-"*80)

from scripts.comprehensive_stress_test_optimized import run_single_test as run_optimized

result = run_optimized(
    num_requests=1000,
    num_gpus=2,
    scheduler_type="multi_bin_dynamic",
    use_real_timestamps=False,
    rps_scaling=200.0,
    d_sla=0.05,
    dataset_path="data/BurstGPT_sample.csv",
    calibration_csv="data/qwen3_1_7b_latency_grid.csv",
    k_bins=8,  # This was failing before fix
    show_progress=True
)

if result['status'] == 'success':
    print(f"✓ OPTIMIZED script K=8 PASSED")
else:
    print(f"✗ OPTIMIZED script K=8 FAILED: {result.get('error', 'Unknown')}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\nBoth scripts should now handle K=8, 16, 32 correctly!")
print("\nTo run full Step 3 test:")
print("  Original:  python scripts/comprehensive_stress_test.py --step3-only --best-gpu-count 32")
print("  Optimized: python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 32")
