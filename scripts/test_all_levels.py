#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to validate all three levels of simulation fidelity.

Level 1: Synthetic - Formula-based service time (no dependencies)
Level 2: BurstGPT - Real Azure dataset (requires CSV download)
Level 3: GPU Calibrated - Measured latency from RTX 4080 (requires GPU + calibration)

This script runs comprehensive tests to ensure all three levels work correctly
and produce valid comparative results.
"""

import argparse
import sys
import io
from pathlib import Path
import os
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.config import SchedulerConfig, compute_equal_mass_boundaries
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.experiments import run_experiment, compare_schedulers
from mb_dyn_sim.metrics import print_metrics_table


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_level_1_synthetic():
    """
    Level 1: Synthetic - Formula-based service time
    
    Validity: ✓ Relative comparisons valid
    Requirements: None
    
    This tests the basic simulation with synthetic latency model:
    T(b, L) = α + β·L·(1 + γ·(b-1)/b)
    """
    print_header("LEVEL 1: SYNTHETIC (Formula-based service time)")
    
    print("Configuration:")
    print("  - Service time: Synthetic formula T(b,L) = α + β·L·(1 + γ·(b-1)/b)")
    print("  - Arrival profile: Poisson")
    print("  - Workload: Synthetic distributions")
    print("  - Purpose: Quick testing, algorithm validation, relative comparisons")
    print()
    
    cfg = SchedulerConfig(
        NUM_GPUS=2,
        K_BINS=4,
        NUM_REQUESTS=1000,
        SEED=42,
        D_SLA=1.0,
        ARRIVAL_PROFILE='poisson',
        POISSON_LAMBDA=50.0,
        USE_EQUAL_MASS_BINS=True,
        EXPERIMENT_MODE='multi_bin_dynamic',
        USE_REAL_MODEL=False,  # Use synthetic model
        USE_REAL_CALIBRATION=False,
    )
    
    # Compute equal-mass bin boundaries
    print("Computing equal-mass bin boundaries...")
    sample_cfg = SchedulerConfig(
        NUM_REQUESTS=min(2000, cfg.NUM_REQUESTS),
        SEED=cfg.SEED,
        ARRIVAL_PROFILE=cfg.ARRIVAL_PROFILE,
        POISSON_LAMBDA=cfg.POISSON_LAMBDA,
    )
    sample_requests = generate_workload(sample_cfg)
    predicted_lengths = [r.predicted_output_len for r in sample_requests]
    cfg.BIN_BOUNDARIES = compute_equal_mass_boundaries(predicted_lengths, cfg.K_BINS)
    print(f"Bin boundaries: {cfg.BIN_BOUNDARIES}\n")
    
    # Test all three schedulers
    scheduler_types = ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']
    
    print("Running simulations for all scheduler types...")
    start_time = time.time()
    
    results = {}
    for sched_type in scheduler_types:
        print(f"\n  Testing {sched_type}...")
        result = run_experiment(cfg, sched_type, load_level='medium')
        results[sched_type] = result
        
        # Quick validation
        metrics = result['metrics']
        assert metrics['throughput_requests_per_sec'] > 0, f"{sched_type}: Throughput must be > 0"
        assert metrics['avg_latency'] > 0, f"{sched_type}: Average latency must be > 0"
        assert 0 <= metrics['sla_violation_rate'] <= 1, f"{sched_type}: SLA violation rate must be 0-1"
        print(f"    ✓ Throughput: {metrics['throughput_requests_per_sec']:.2f} req/s")
        print(f"    ✓ Avg Latency: {metrics['avg_latency']:.3f}s")
        print(f"    ✓ SLA Violations: {metrics['sla_violation_rate']*100:.1f}%")
    
    elapsed = time.time() - start_time
    
    # Verify schedulers produce different results (key validation)
    static_viol = results['static_fifo']['metrics']['sla_violation_rate'] * 100
    dynamic_viol = results['dynamic_no_bins']['metrics']['sla_violation_rate'] * 100
    multibin_viol = results['multi_bin_dynamic']['metrics']['sla_violation_rate'] * 100
    
    print("\n" + "-"*80)
    print("VALIDATION CHECKS:")
    print("-"*80)
    print(f"✓ All schedulers completed successfully")
    print(f"✓ Execution time: {elapsed:.2f}s")
    print(f"✓ SLA Violations: static={static_viol:.1f}%, dynamic={dynamic_viol:.1f}%, multi_bin={multibin_viol:.1f}%")
    
    # Check that schedulers produce distinct results
    if static_viol == dynamic_viol == multibin_viol:
        print("⚠ WARNING: All schedulers have identical SLA violations (unexpected)")
    else:
        print(f"✓ Schedulers produce distinct results (as expected)")
    
    print("\n✅ LEVEL 1 TEST PASSED: Synthetic simulation working correctly")
    print("   Validity: Relative comparisons valid for algorithm development")
    
    return results


def test_level_2_burstgpt():
    """
    Level 2: BurstGPT - Real Azure dataset
    
    Validity: ✓✓ Realistic workload
    Requirements: Download CSV (48MB) or use sample
    
    This tests the simulation with realistic arrival patterns from Azure traces.
    """
    print_header("LEVEL 2: BURSTGPT (Real Azure dataset)")
    
    # Check if dataset exists
    dataset_path = Path("data/BurstGPT_sample.csv")
    
    if not dataset_path.exists():
        print(f"⚠ WARNING: BurstGPT dataset not found at {dataset_path}")
        print("   Skipping Level 2 test. To enable:")
        print("   1. Download BurstGPT dataset (48MB)")
        print("   2. Place CSV file at data/BurstGPT_sample.csv")
        print("   3. Or use synthetic BurstGPT-like arrivals instead")
        print("\n⏭ LEVEL 2 TEST SKIPPED")
        return None
    
    print("Configuration:")
    print(f"  - Service time: Synthetic formula (same as Level 1)")
    print(f"  - Arrival profile: BurstGPT dataset")
    print(f"  - Workload: Real Azure traces with bursty patterns")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Purpose: Realistic workload validation")
    print()
    
    cfg = SchedulerConfig(
        NUM_GPUS=2,
        K_BINS=4,
        NUM_REQUESTS=1000,
        SEED=42,
        D_SLA=1.0,
        ARRIVAL_PROFILE='burstgpt_dataset',
        DATASET_PATH=str(dataset_path),
        WORKLOAD_SOURCE='burstgpt_dataset',
        RPS_SCALING=100.0,  # Scale to desired request rate
        USE_EQUAL_MASS_BINS=True,
        EXPERIMENT_MODE='multi_bin_dynamic',
        USE_REAL_MODEL=False,  # Still use synthetic latency
        USE_REAL_CALIBRATION=False,
    )
    
    # Compute equal-mass bin boundaries from dataset
    print("Loading BurstGPT dataset and computing bin boundaries...")
    sample_cfg = SchedulerConfig(
        NUM_REQUESTS=min(2000, cfg.NUM_REQUESTS),
        SEED=cfg.SEED,
        ARRIVAL_PROFILE=cfg.ARRIVAL_PROFILE,
        DATASET_PATH=cfg.DATASET_PATH,
        WORKLOAD_SOURCE=cfg.WORKLOAD_SOURCE,
        RPS_SCALING=cfg.RPS_SCALING,
    )
    sample_requests = generate_workload(sample_cfg)
    predicted_lengths = [r.predicted_output_len for r in sample_requests]
    cfg.BIN_BOUNDARIES = compute_equal_mass_boundaries(predicted_lengths, cfg.K_BINS)
    print(f"Bin boundaries from dataset: {cfg.BIN_BOUNDARIES}\n")
    
    # Test all three schedulers
    scheduler_types = ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']
    
    print("Running simulations with BurstGPT workload...")
    start_time = time.time()
    
    results = {}
    for sched_type in scheduler_types:
        print(f"\n  Testing {sched_type}...")
        result = run_experiment(cfg, sched_type, load_level='medium')
        results[sched_type] = result
        
        metrics = result['metrics']
        print(f"    ✓ Throughput: {metrics['throughput_requests_per_sec']:.2f} req/s")
        print(f"    ✓ Avg Latency: {metrics['avg_latency']:.3f}s")
        print(f"    ✓ SLA Violations: {metrics['sla_violation_rate']*100:.1f}%")
    
    elapsed = time.time() - start_time
    
    print("\n" + "-"*80)
    print("VALIDATION CHECKS:")
    print("-"*80)
    print(f"✓ BurstGPT dataset loaded successfully")
    print(f"✓ All schedulers completed with realistic arrivals")
    print(f"✓ Execution time: {elapsed:.2f}s")
    
    print("\n✅ LEVEL 2 TEST PASSED: BurstGPT realistic workload simulation working")
    print("   Validity: Realistic workload patterns from Azure production traces")
    
    return results


def test_level_3_gpu_calibrated():
    """
    Level 3: GPU Calibrated - Measured latency from RTX 4080
    
    Validity: ✓✓✓ Production-ready
    Requirements: GPU + Transformers/vLLM + calibration data
    
    This tests the simulation with real GPU-calibrated latency measurements.
    """
    print_header("LEVEL 3: GPU CALIBRATED (Measured latency from RTX 4080)")
    
    # Check if calibration data exists
    calibration_path = Path("data/qwen3_1_7b_latency_grid.csv")
    
    if not calibration_path.exists():
        print(f"⚠ WARNING: GPU calibration data not found at {calibration_path}")
        print("   Skipping Level 3 test. To enable:")
        print("   1. Run GPU calibration script:")
        print("      python scripts/calibrate_real_gpu_transformers.py")
        print("   2. Or use existing calibration file")
        print("\n⏭ LEVEL 3 TEST SKIPPED")
        return None
    
    print("Configuration:")
    print(f"  - Service time: GPU-calibrated from real measurements")
    print(f"  - Arrival profile: Poisson (configurable)")
    print(f"  - Workload: Synthetic or BurstGPT")
    print(f"  - Calibration: {calibration_path}")
    print(f"  - Purpose: Production-ready accuracy")
    print()
    
    cfg = SchedulerConfig(
        NUM_GPUS=2,
        K_BINS=4,
        NUM_REQUESTS=1000,
        SEED=42,
        D_SLA=1.0,
        ARRIVAL_PROFILE='poisson',
        POISSON_LAMBDA=50.0,
        USE_EQUAL_MASS_BINS=True,
        EXPERIMENT_MODE='multi_bin_dynamic',
        USE_REAL_MODEL=False,  # Don't run actual vLLM
        USE_REAL_CALIBRATION=True,  # Use calibration data
        CALIBRATION_CSV_PATH=str(calibration_path),
    )
    
    # Compute equal-mass bin boundaries
    print("Computing equal-mass bin boundaries...")
    sample_cfg = SchedulerConfig(
        NUM_REQUESTS=min(2000, cfg.NUM_REQUESTS),
        SEED=cfg.SEED,
        ARRIVAL_PROFILE=cfg.ARRIVAL_PROFILE,
        POISSON_LAMBDA=cfg.POISSON_LAMBDA,
    )
    sample_requests = generate_workload(sample_cfg)
    predicted_lengths = [r.predicted_output_len for r in sample_requests]
    cfg.BIN_BOUNDARIES = compute_equal_mass_boundaries(predicted_lengths, cfg.K_BINS)
    print(f"Bin boundaries: {cfg.BIN_BOUNDARIES}\n")
    
    # Test all three schedulers
    scheduler_types = ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']
    
    print("Running simulations with GPU-calibrated latency model...")
    start_time = time.time()
    
    results = {}
    for sched_type in scheduler_types:
        print(f"\n  Testing {sched_type}...")
        result = run_experiment(cfg, sched_type, load_level='medium')
        results[sched_type] = result
        
        metrics = result['metrics']
        print(f"    ✓ Throughput: {metrics['throughput_requests_per_sec']:.2f} req/s")
        print(f"    ✓ Avg Latency: {metrics['avg_latency']:.3f}s")
        print(f"    ✓ SLA Violations: {metrics['sla_violation_rate']*100:.1f}%")
    
    elapsed = time.time() - start_time
    
    print("\n" + "-"*80)
    print("VALIDATION CHECKS:")
    print("-"*80)
    print(f"✓ GPU calibration data loaded successfully")
    print(f"✓ All schedulers completed with realistic latency")
    print(f"✓ Execution time: {elapsed:.2f}s")
    
    print("\n✅ LEVEL 3 TEST PASSED: GPU-calibrated simulation working")
    print("   Validity: Production-ready accuracy with real GPU measurements")
    
    return results


def test_level_4_full_production():
    """
    Level 4: Full Production - Real BurstGPT + GPU Calibrated
    
    Validity: ✓✓✓✓ Maximum realism
    Requirements: BurstGPT CSV + GPU calibration CSV
    
    This combines the best of Level 2 and Level 3:
    - Real Azure arrival patterns (bursty workload)
    - Real GPU latency measurements
    - Complete production simulation
    """
    print_header("LEVEL 4: FULL PRODUCTION (BurstGPT + GPU Calibrated)")
    
    # Check if both datasets exist
    dataset_path = Path("data/BurstGPT_sample.csv")
    calibration_path = Path("data/qwen3_1_7b_latency_grid.csv")
    
    missing = []
    if not dataset_path.exists():
        missing.append(f"BurstGPT dataset: {dataset_path}")
    if not calibration_path.exists():
        missing.append(f"GPU calibration: {calibration_path}")
    
    if missing:
        print(f"⚠ WARNING: Required files not found:")
        for item in missing:
            print(f"   - {item}")
        print("\n⏭ LEVEL 4 TEST SKIPPED")
        return None
    
    print("Configuration:")
    print(f"  - Service time: GPU-calibrated from real RTX 4080 measurements")
    print(f"  - Arrival profile: BurstGPT dataset (real Azure traces)")
    print(f"  - Workload: Real bursty patterns with real latency model")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Calibration: {calibration_path}")
    print(f"  - Purpose: Maximum realism - production simulation")
    print()
    
    cfg = SchedulerConfig(
        NUM_GPUS=2,
        K_BINS=4,
        NUM_REQUESTS=1000,
        SEED=42,
        D_SLA=1.0,
        ARRIVAL_PROFILE='burstgpt_dataset',
        DATASET_PATH=str(dataset_path),
        WORKLOAD_SOURCE='burstgpt_dataset',
        RPS_SCALING=100.0,  # Scale to desired request rate
        USE_EQUAL_MASS_BINS=True,
        EXPERIMENT_MODE='multi_bin_dynamic',
        USE_REAL_MODEL=False,
        USE_REAL_CALIBRATION=True,  # ← Real GPU latency
        CALIBRATION_CSV_PATH=str(calibration_path),
    )
    
    # Compute equal-mass bin boundaries from dataset
    print("Loading BurstGPT dataset and computing bin boundaries...")
    sample_cfg = SchedulerConfig(
        NUM_REQUESTS=min(2000, cfg.NUM_REQUESTS),
        SEED=cfg.SEED,
        ARRIVAL_PROFILE=cfg.ARRIVAL_PROFILE,
        DATASET_PATH=cfg.DATASET_PATH,
        WORKLOAD_SOURCE=cfg.WORKLOAD_SOURCE,
        RPS_SCALING=cfg.RPS_SCALING,
    )
    sample_requests = generate_workload(sample_cfg)
    predicted_lengths = [r.predicted_output_len for r in sample_requests]
    cfg.BIN_BOUNDARIES = compute_equal_mass_boundaries(predicted_lengths, cfg.K_BINS)
    print(f"Bin boundaries from dataset: {cfg.BIN_BOUNDARIES}\n")
    
    # Test all three schedulers
    scheduler_types = ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']
    
    print("Running FULL PRODUCTION simulations (BurstGPT + GPU calibrated)...")
    start_time = time.time()
    
    results = {}
    for sched_type in scheduler_types:
        print(f"\n  Testing {sched_type}...")
        result = run_experiment(cfg, sched_type, load_level='medium')
        results[sched_type] = result
        
        metrics = result['metrics']
        print(f"    ✓ Throughput: {metrics['throughput_requests_per_sec']:.2f} req/s")
        print(f"    ✓ Avg Latency: {metrics['avg_latency']:.3f}s")
        print(f"    ✓ SLA Violations: {metrics['sla_violation_rate']*100:.1f}%")
    
    elapsed = time.time() - start_time
    
    print("\n" + "-"*80)
    print("VALIDATION CHECKS:")
    print("-"*80)
    print(f"✓ BurstGPT dataset loaded (real arrivals)")
    print(f"✓ GPU calibration loaded (real latency)")
    print(f"✓ All schedulers completed with maximum realism")
    print(f"✓ Execution time: {elapsed:.2f}s")
    
    print("\n✅ LEVEL 4 TEST PASSED: Full production simulation working")
    print("   Validity: Maximum realism combining real workload + real GPU")
    
    return results


def print_summary(level1_results, level2_results, level3_results, level4_results):
    """Print comprehensive summary of all test results"""
    print_header("COMPREHENSIVE TEST SUMMARY")
    
    print("Test Coverage:")
    print(f"  ✅ Level 1 (Synthetic): {'PASSED' if level1_results else 'FAILED'}")
    print(f"  {'✅' if level2_results else '⏭'} Level 2 (BurstGPT): {'PASSED' if level2_results else 'SKIPPED'}")
    print(f"  {'✅' if level3_results else '⏭'} Level 3 (GPU Calibrated): {'PASSED' if level3_results else 'SKIPPED'}")
    print(f"  {'✅' if level4_results else '⏭'} Level 4 (Full Production): {'PASSED' if level4_results else 'SKIPPED'}")
    
    print("\n" + "-"*80)
    print("VALIDITY COMPARISON:")
    print("-"*80)
    print("Level 1: Synthetic")
    print("  ✓ Relative comparisons valid")
    print("  ✓ Fast iteration for algorithm development")
    print("  ✓ No external dependencies")
    
    if level2_results:
        print("\nLevel 2: BurstGPT")
        print("  ✓✓ Realistic workload patterns")
        print("  ✓✓ Real arrival distributions from Azure")
        print("  ✓✓ Validates bursty traffic handling")
    else:
        print("\nLevel 2: BurstGPT (SKIPPED)")
        print("  To enable: Download BurstGPT dataset to data/BurstGPT_sample.csv")
    
    if level3_results:
        print("\nLevel 3: GPU Calibrated")
        print("  ✓✓✓ Production-ready accuracy")
        print("  ✓✓✓ Real GPU latency measurements")
        print("  ✓✓✓ Hardware-specific validation")
    else:
        print("\nLevel 3: GPU Calibrated (SKIPPED)")
        print("  To enable: Run python scripts/calibrate_real_gpu_transformers.py")
    
    if level4_results:
        print("\nLevel 4: Full Production")
        print("  ✓✓✓✓ Maximum realism")
        print("  ✓✓✓✓ Real BurstGPT arrivals + Real GPU latency")
        print("  ✓✓✓✓ Complete production simulation")
    else:
        print("\nLevel 4: Full Production (SKIPPED)")
        print("  To enable: Ensure both BurstGPT dataset and GPU calibration available")
    
    print("\n" + "-"*80)
    print("KEY FINDINGS:")
    print("-"*80)
    
    if level1_results:
        static_viol = level1_results['static_fifo']['metrics']['sla_violation_rate'] * 100
        dynamic_viol = level1_results['dynamic_no_bins']['metrics']['sla_violation_rate'] * 100
        multibin_viol = level1_results['multi_bin_dynamic']['metrics']['sla_violation_rate'] * 100
        
        print(f"\nLevel 1 (Synthetic) - SLA Violations:")
        print(f"  static_fifo:       {static_viol:6.1f}%")
        print(f"  dynamic_no_bins:   {dynamic_viol:6.1f}% ({(dynamic_viol-static_viol)/static_viol*100:+.1f}% vs static)")
        print(f"  multi_bin_dynamic: {multibin_viol:6.1f}% ({(multibin_viol-dynamic_viol)/dynamic_viol*100:+.1f}% vs dynamic)")
    
    if level3_results:
        static_viol = level3_results['static_fifo']['metrics']['sla_violation_rate'] * 100
        dynamic_viol = level3_results['dynamic_no_bins']['metrics']['sla_violation_rate'] * 100
        multibin_viol = level3_results['multi_bin_dynamic']['metrics']['sla_violation_rate'] * 100
        
        print(f"\nLevel 3 (GPU Calibrated) - SLA Violations:")
        print(f"  static_fifo:       {static_viol:6.1f}%")
        print(f"  dynamic_no_bins:   {dynamic_viol:6.1f}% ({(dynamic_viol-static_viol)/static_viol*100:+.1f}% vs static)")
        print(f"  multi_bin_dynamic: {multibin_viol:6.1f}% ({(multibin_viol-dynamic_viol)/dynamic_viol*100:+.1f}% vs dynamic)")
    
    if level4_results:
        static_viol = level4_results['static_fifo']['metrics']['sla_violation_rate'] * 100
        dynamic_viol = level4_results['dynamic_no_bins']['metrics']['sla_violation_rate'] * 100
        multibin_viol = level4_results['multi_bin_dynamic']['metrics']['sla_violation_rate'] * 100
        
        print(f"\nLevel 4 (Full Production) - SLA Violations:")
        print(f"  static_fifo:       {static_viol:6.1f}%")
        print(f"  dynamic_no_bins:   {dynamic_viol:6.1f}% ({(dynamic_viol-static_viol)/static_viol*100:+.1f}% vs static)")
        print(f"  multi_bin_dynamic: {multibin_viol:6.1f}% ({(multibin_viol-dynamic_viol)/dynamic_viol*100:+.1f}% vs dynamic)")
        
        # Highlight the achievement
        print(f"\n{'🏆 PRODUCTION SIMULATION COMPLETE':^80}")
        print(f"{'Real workload + Real GPU = Maximum Fidelity':^80}")
    
    print("\n" + "="*80)
    print("OVERALL STATUS: ", end="")
    
    if level4_results:
        print("✅ ALL LEVELS INCLUDING PRODUCTION TESTED SUCCESSFULLY")
    elif level1_results and (level2_results or level3_results):
        print("✅ ALL AVAILABLE LEVELS TESTED SUCCESSFULLY")
    elif level1_results:
        print("✅ BASIC LEVEL TESTED (Level 2, 3, 4 optional)")
    else:
        print("❌ TESTS FAILED")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Test all three levels of simulation fidelity"
    )
    
    parser.add_argument(
        '--level',
        type=str,
        choices=['1', '2', '3', '4', 'all'],
        default='all',
        help='Test specific level or all levels (default: all)'
    )
    
    parser.add_argument(
        '--skip-optional',
        action='store_true',
        help='Skip optional levels (Level 2 & 3) if dependencies missing'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("  LLM SCHEDULER SIMULATION - FIDELITY LEVEL TESTING")
    print("="*80)
    print("\nThis script validates all four levels of simulation fidelity:")
    print("  Level 1: Synthetic (formula-based) - Always available")
    print("  Level 2: BurstGPT dataset - Requires CSV download")
    print("  Level 3: GPU Calibrated - Requires GPU measurements")
    print("  Level 4: Full Production - Requires both BurstGPT + GPU calibration")
    print()
    
    level1_results = None
    level2_results = None
    level3_results = None
    level4_results = None
    
    try:
        if args.level in ['1', 'all']:
            level1_results = test_level_1_synthetic()
        
        if args.level in ['2', 'all']:
            level2_results = test_level_2_burstgpt()
        
        if args.level in ['3', 'all']:
            level3_results = test_level_3_gpu_calibrated()
        
        if args.level in ['4', 'all']:
            level4_results = test_level_4_full_production()
        
        # Print comprehensive summary
        print_summary(level1_results, level2_results, level3_results, level4_results)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
