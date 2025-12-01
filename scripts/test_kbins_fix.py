#!/usr/bin/env python3
"""
Quick test to verify K-bins sensitivity works for all K values.
This tests the fix for the bin boundary caching bug.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.comprehensive_stress_test_optimized import run_single_test

# Test all K values with a small workload
print("Testing K-bins sensitivity fix...")
print("="*80)

k_values = [1, 2, 4, 8, 16, 32]
results = []

for k in k_values:
    print(f"\nTesting K={k}...")
    result = run_single_test(
        num_requests=1000,  # Small test
        num_gpus=2,
        scheduler_type="multi_bin_dynamic",
        use_real_timestamps=False,
        rps_scaling=200.0,
        d_sla=0.05,
        dataset_path="data/BurstGPT_sample.csv",
        calibration_csv="data/qwen3_1_7b_latency_grid.csv",
        k_bins=k,
        show_progress=True
    )
    
    if result['status'] == 'success':
        print(f"✓ K={k} PASSED")
        results.append((k, 'PASS'))
    else:
        print(f"✗ K={k} FAILED: {result.get('error', 'Unknown error')}")
        results.append((k, 'FAIL', result.get('error', '')))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
for r in results:
    if len(r) == 2:
        print(f"  K={r[0]:2d}: {r[1]}")
    else:
        print(f"  K={r[0]:2d}: {r[1]} - {r[2]}")

passed = sum(1 for r in results if r[1] == 'PASS')
print(f"\nPassed: {passed}/{len(k_values)}")

if passed == len(k_values):
    print("\n✓ ALL TESTS PASSED - Fix verified!")
else:
    print(f"\n✗ {len(k_values) - passed} tests failed")
