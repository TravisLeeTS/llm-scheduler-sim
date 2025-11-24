#!/usr/bin/env python3
"""
Stress Testing Suite for Multi-Bin Dynamic Scheduler

Progressive load testing to find system limits:
1. Phase 1: Increase request count (10K → 100K → 1M → full dataset)
2. Phase 2: Scale GPUs when multi-bin saturates (4 → 8 → 16 → ... → 100)

Results saved to Excel for analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.config import SchedulerConfig, compute_equal_mass_boundaries
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics


def run_single_test(
    num_requests: int,
    num_gpus: int,
    scheduler_type: str,
    rps_scaling: float,
    d_sla: float,
    dataset_path: str,
    calibration_csv: str
) -> dict:
    """
    Run a single stress test configuration.
    
    Returns:
        Dictionary with test configuration and results
    """
    print(f"\n{'='*80}")
    print(f"TEST: {scheduler_type} | Requests: {num_requests:,} | GPUs: {num_gpus} | RPS: {rps_scaling}x")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Create configuration
        cfg = SchedulerConfig(
            NUM_GPUS=num_gpus,
            K_BINS=4,
            NUM_REQUESTS=num_requests,
            SEED=42,
            D_SLA=d_sla,
            DATASET_PATH=dataset_path,
            WORKLOAD_SOURCE="burstgpt_dataset",
            RPS_SCALING=rps_scaling,
            USE_EQUAL_MASS_BINS=True,
            USE_REAL_CALIBRATION=True,
            CALIBRATION_CSV_PATH=calibration_csv
        )
        
        # Compute equal-mass bin boundaries
        if cfg.K_BINS > 1 and scheduler_type == "multi_bin_dynamic":
            sample_cfg = SchedulerConfig(
                NUM_REQUESTS=min(5000, num_requests),
                SEED=cfg.SEED,
                DATASET_PATH=cfg.DATASET_PATH,
                WORKLOAD_SOURCE="burstgpt_dataset",
                RPS_SCALING=cfg.RPS_SCALING,
            )
            sample_requests = generate_workload(sample_cfg)
            predicted_lengths = [r.predicted_output_len for r in sample_requests]
            cfg.BIN_BOUNDARIES = compute_equal_mass_boundaries(predicted_lengths, cfg.K_BINS)
        
        # Generate workload
        print(f"Loading {num_requests:,} requests from dataset...")
        requests = generate_workload(cfg)
        
        # Run simulation
        print(f"Running {scheduler_type} simulation...")
        simulator = Simulator(cfg, requests, scheduler_type)
        completed_requests = simulator.run()
        
        # Compute metrics
        from mb_dyn_sim.metrics import compute_gpu_utilization, compute_batch_statistics
        metrics = compute_metrics(completed_requests)
        gpu_stats = simulator.get_gpu_stats()
        gpu_metrics = compute_gpu_utilization(gpu_stats)
        batch_stats = compute_batch_statistics(completed_requests)
        
        # Merge metrics
        metrics.update(gpu_metrics)
        metrics.update(batch_stats)
        
        # Add configuration info
        result = {
            # Configuration
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'rps_scaling': rps_scaling,
            'd_sla': d_sla,
            'k_bins': cfg.K_BINS if scheduler_type == "multi_bin_dynamic" else 1,
            
            # Performance Metrics
            'sla_violation_rate': metrics['sla_violation_rate'],
            'capacity_qps_under_sla': metrics['capacity_qps_under_sla'],
            'throughput_requests_per_sec': metrics['throughput_requests_per_sec'],
            'throughput_tokens_per_sec': metrics['throughput_tokens_per_sec'],
            
            # Latency Metrics
            'avg_latency': metrics['avg_latency'],
            'p50_latency': metrics['p50_latency'],
            'p95_latency': metrics['p95_latency'],
            'p99_latency': metrics['p99_latency'],
            'max_latency': metrics['max_latency'],
            
            # Queue Metrics
            'avg_queueing_delay': metrics['avg_queueing_delay'],
            'avg_service_time': metrics['avg_service_time'],
            
            # GPU Utilization
            'avg_gpu_utilization': metrics['avg_utilization'],
            'min_gpu_utilization': metrics['min_utilization'],
            'max_gpu_utilization': metrics['max_utilization'],
            
            # Batch Statistics
            'num_batches': metrics['num_batches'],
            'avg_batch_size': metrics['avg_batch_size'],
            'min_batch_size': metrics['min_batch_size'],
            'max_batch_size': metrics['max_batch_size'],
            
            # Workload
            'total_time': metrics['total_time'],
            'total_tokens': metrics['total_tokens'],
            'num_completed': metrics['num_requests'],
            
            # Execution Time
            'execution_time_seconds': time.time() - start_time,
            
            # Status
            'status': 'success'
        }
        
        # Print summary
        print(f"\n[SUCCESS] COMPLETED in {result['execution_time_seconds']:.1f}s")
        print(f"  SLA Violations: {result['sla_violation_rate']:.1f}%")
        print(f"  Capacity QPS:   {result['capacity_qps_under_sla']:.2f}")
        print(f"  Avg Latency:    {result['avg_latency']:.3f}s")
        print(f"  GPU Util:       {result['avg_gpu_utilization']:.1f}%")
        
        return result
        
    except Exception as e:
        print(f"\n[FAILED]: {str(e)}")
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'rps_scaling': rps_scaling,
            'd_sla': d_sla,
            'status': f'failed: {str(e)}',
            'execution_time_seconds': time.time() - start_time,
        }


def phase1_load_scaling(
    dataset_path: str,
    calibration_csv: str,
    max_requests: int = 1429738,
    base_rps: float = 200.0,
    d_sla: float = 0.5
) -> pd.DataFrame:
    """
    Phase 1: Scale request count to find breaking point.
    
    Test configurations:
    - 10K, 100K, 1M, full dataset (~1.43M)
    - All three schedulers
    - Fixed 4 GPUs for multi-bin, 1 GPU for baselines
    
    Returns:
        DataFrame with all results
    """
    print("\n" + "="*80)
    print("PHASE 1: LOAD SCALING - Finding Request Volume Breaking Point")
    print("="*80)
    
    results = []
    
    # Request counts to test (10x increments until dataset limit)
    request_counts = [10000]
    current = 10000
    while current * 10 <= max_requests:
        current *= 10
        request_counts.append(current)
    if max_requests not in request_counts:
        request_counts.append(max_requests)
    
    print(f"\nRequest counts to test: {[f'{r:,}' for r in request_counts]}")
    
    for num_requests in request_counts:
        # Test all three schedulers
        for scheduler_type in ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']:
            # Architecturally-appropriate GPU allocation
            if scheduler_type == 'multi_bin_dynamic':
                num_gpus = 4  # Bins enable parallelization
            else:
                num_gpus = 1  # No parallelization mechanism
            
            result = run_single_test(
                num_requests=num_requests,
                num_gpus=num_gpus,
                scheduler_type=scheduler_type,
                rps_scaling=base_rps,
                d_sla=d_sla,
                dataset_path=dataset_path,
                calibration_csv=calibration_csv
            )
            results.append(result)
    
    return pd.DataFrame(results)


def phase2_gpu_scaling(
    dataset_path: str,
    calibration_csv: str,
    saturation_requests: int,
    base_rps: float = 200.0,
    d_sla: float = 0.5,
    max_gpus: int = 100
) -> pd.DataFrame:
    """
    Phase 2: Scale GPU count when multi-bin saturates.
    
    Test multi_bin_dynamic with increasing GPU counts:
    - 4, 8, 16, 32, 64, 100 GPUs
    
    Returns:
        DataFrame with all results
    """
    print("\n" + "="*80)
    print("PHASE 2: GPU SCALING - Finding GPU Resource Limits")
    print("="*80)
    print(f"Using {saturation_requests:,} requests (saturation point from Phase 1)")
    
    results = []
    
    # GPU counts to test (double each time)
    gpu_counts = [4]
    current = 4
    while current * 2 <= max_gpus:
        current *= 2
        gpu_counts.append(current)
    if max_gpus not in gpu_counts and max_gpus > gpu_counts[-1]:
        gpu_counts.append(max_gpus)
    
    print(f"\nGPU counts to test: {gpu_counts}")
    
    for num_gpus in gpu_counts:
        result = run_single_test(
            num_requests=saturation_requests,
            num_gpus=num_gpus,
            scheduler_type='multi_bin_dynamic',
            rps_scaling=base_rps,
            d_sla=d_sla,
            dataset_path=dataset_path,
            calibration_csv=calibration_csv
        )
        results.append(result)
    
    return pd.DataFrame(results)


def save_results_to_excel(
    phase1_df: pd.DataFrame,
    phase2_df: pd.DataFrame,
    output_path: str
):
    """
    Save stress test results to Excel with multiple sheets.
    Falls back to CSV if openpyxl not available.
    """
    print(f"\n{'='*80}")
    print(f"Saving results to: {output_path}")
    print(f"{'='*80}")
    
    try:
        import openpyxl
        use_excel = True
    except ImportError:
        print("Warning: openpyxl not available, saving as CSV instead")
        use_excel = False
        output_path = output_path.replace('.xlsx', '.csv')
    
    if use_excel:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Phase 1: Load scaling
            phase1_df.to_excel(writer, sheet_name='Phase1_Load_Scaling', index=False)
            
            # Phase 2: GPU scaling
            phase2_df.to_excel(writer, sheet_name='Phase2_GPU_Scaling', index=False)
            
            # Summary statistics
            summary_data = {
                'Test Phase': ['Phase 1: Load Scaling', 'Phase 2: GPU Scaling'],
                'Total Tests': [len(phase1_df), len(phase2_df)],
                'Successful': [
                    len(phase1_df[phase1_df['status'] == 'success']),
                    len(phase2_df[phase2_df['status'] == 'success'])
                ],
                'Failed': [
                    len(phase1_df[phase1_df['status'] != 'success']),
                    len(phase2_df[phase2_df['status'] != 'success'])
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Best configurations (lowest SLA violations)
            successful_phase1 = phase1_df[phase1_df['status'] == 'success'].copy()
            if len(successful_phase1) > 0:
                best_phase1 = successful_phase1.nsmallest(10, 'sla_violation_rate')
                best_phase1.to_excel(writer, sheet_name='Best_Phase1_Configs', index=False)
            
            successful_phase2 = phase2_df[phase2_df['status'] == 'success'].copy()
            if len(successful_phase2) > 0:
                best_phase2 = successful_phase2.nsmallest(10, 'sla_violation_rate')
                best_phase2.to_excel(writer, sheet_name='Best_Phase2_Configs', index=False)
            
            # Worst configurations (highest SLA violations)
            if len(successful_phase1) > 0:
                worst_phase1 = successful_phase1.nlargest(10, 'sla_violation_rate')
                worst_phase1.to_excel(writer, sheet_name='Worst_Phase1_Configs', index=False)
    else:
        # Save as CSV files
        phase1_csv = output_path.replace('.csv', '_phase1.csv')
        phase2_csv = output_path.replace('.csv', '_phase2.csv')
        phase1_df.to_csv(phase1_csv, index=False)
        phase2_df.to_csv(phase2_csv, index=False)
        print(f"Saved Phase 1 results to: {phase1_csv}")
        print(f"Saved Phase 2 results to: {phase2_csv}")
    
    print(f"[SUCCESS] Results saved successfully!")
    print(f"\nSheets created:")
    print(f"  1. Phase1_Load_Scaling - Request volume scaling results")
    print(f"  2. Phase2_GPU_Scaling - GPU resource scaling results")
    print(f"  3. Summary - Overall test statistics")
    print(f"  4. Best_Phase1_Configs - Top 10 configurations by SLA")
    print(f"  5. Best_Phase2_Configs - Top 10 GPU configs by SLA")
    print(f"  6. Worst_Phase1_Configs - Bottom 10 configurations")


def main():
    parser = argparse.ArgumentParser(
        description="Stress test Multi-Bin scheduler to find system limits"
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/BurstGPT_sample.csv',
        help='Path to BurstGPT dataset CSV (default: data/BurstGPT_sample.csv)'
    )
    
    parser.add_argument(
        '--calibration-csv',
        type=str,
        default='data/qwen3_1_7b_latency_grid.csv',
        help='Path to GPU calibration CSV (default: data/qwen3_1_7b_latency_grid.csv)'
    )
    
    parser.add_argument(
        '--max-requests',
        type=int,
        default=1429738,
        help='Maximum requests to test (default: 1429738 - full dataset)'
    )
    
    parser.add_argument(
        '--max-gpus',
        type=int,
        default=100,
        help='Maximum GPUs to test in Phase 2 (default: 100)'
    )
    
    parser.add_argument(
        '--rps-scaling',
        type=float,
        default=200.0,
        help='RPS scaling factor (default: 200.0)'
    )
    
    parser.add_argument(
        '--d-sla',
        type=float,
        default=0.5,
        help='SLA deadline in seconds (default: 0.5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='stress_test_results.xlsx',
        help='Output Excel file path (default: stress_test_results.xlsx)'
    )
    
    parser.add_argument(
        '--phase1-only',
        action='store_true',
        help='Run only Phase 1 (load scaling)'
    )
    
    parser.add_argument(
        '--phase2-only',
        action='store_true',
        help='Run only Phase 2 (GPU scaling)'
    )
    
    parser.add_argument(
        '--saturation-requests',
        type=int,
        default=100000,
        help='Request count for Phase 2 testing (default: 100000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MULTI-BIN SCHEDULER STRESS TEST SUITE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset:           {args.dataset_path}")
    print(f"  Calibration:       {args.calibration_csv}")
    print(f"  Max Requests:      {args.max_requests:,}")
    print(f"  Max GPUs:          {args.max_gpus}")
    print(f"  RPS Scaling:       {args.rps_scaling}x")
    print(f"  SLA Deadline:      {args.d_sla}s")
    print(f"  Output:            {args.output}")
    
    start_time = time.time()
    
    # Run Phase 1: Load Scaling
    if not args.phase2_only:
        phase1_df = phase1_load_scaling(
            dataset_path=args.dataset_path,
            calibration_csv=args.calibration_csv,
            max_requests=args.max_requests,
            base_rps=args.rps_scaling,
            d_sla=args.d_sla
        )
        
        # Determine saturation point (where multi-bin SLA violations > 50%)
        multi_bin_results = phase1_df[
            (phase1_df['scheduler_type'] == 'multi_bin_dynamic') &
            (phase1_df['status'] == 'success')
        ]
        if len(multi_bin_results) > 0:
            saturated = multi_bin_results[multi_bin_results['sla_violation_rate'] > 50.0]
            if len(saturated) > 0:
                saturation_requests = saturated.iloc[0]['num_requests']
            else:
                saturation_requests = multi_bin_results.iloc[-1]['num_requests']
        else:
            saturation_requests = args.saturation_requests
    else:
        phase1_df = pd.DataFrame()
        saturation_requests = args.saturation_requests
    
    # Run Phase 2: GPU Scaling
    if not args.phase1_only:
        phase2_df = phase2_gpu_scaling(
            dataset_path=args.dataset_path,
            calibration_csv=args.calibration_csv,
            saturation_requests=saturation_requests,
            base_rps=args.rps_scaling,
            d_sla=args.d_sla,
            max_gpus=args.max_gpus
        )
    else:
        phase2_df = pd.DataFrame()
    
    # Save results
    if len(phase1_df) > 0 or len(phase2_df) > 0:
        save_results_to_excel(phase1_df, phase2_df, args.output)
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"STRESS TEST COMPLETE")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
