#!/usr/bin/env python3
"""
Comprehensive Stress Testing Suite for Research Paper (3-Step Plan)

Step 1: Request Scaling Stress Test (1K -> 1M, 10x each time)
  - static_fifo (1 GPU) - Baseline
  - dynamic_no_bins (1 GPU) - Dynamic batching
  - multi_bin_dynamic (1 GPU, 2 GPUs, 4 GPUs) - Our contribution
  Tests: Algorithm ability to handle increasing request volumes

Step 2: GPU Scaling Stress Test (1M requests, 1-100 GPUs)
  - multi_bin_dynamic only
  - GPU counts: 1, 2, 4, 8, 16, 32, 64, 100
  Tests: Algorithm ability to assign tasks across large GPU pools

Step 3: K-Bins Sensitivity Analysis (Best config from Step 2)
  - multi_bin_dynamic only
  - K values: 1, 2, 4, 8, 16, 32
  - Uses best GPU count from Step 2
  Tests: Optimal bin partitioning strategy

All results saved to CSV/JSON for research paper analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import time
import json
from datetime import datetime
import argparse
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.config import SchedulerConfig, compute_equal_mass_boundaries
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics


def run_single_test(
    num_requests: int,
    num_gpus: int,
    scheduler_type: str,
    use_real_timestamps: bool,
    rps_scaling: float,
    d_sla: float,
    dataset_path: str,
    calibration_csv: str,
    k_bins: int = 4
) -> dict:
    """Run a single stress test configuration."""
    timestamp_mode = "REAL timestamps" if use_real_timestamps else f"{rps_scaling}x RPS"
    k_info = f"K={k_bins}" if scheduler_type == "multi_bin_dynamic" else ""
    print(f"\n{'='*80}")
    print(f"TEST: {scheduler_type} {k_info} | Requests: {num_requests:,} | GPUs: {num_gpus} | {timestamp_mode}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Compute bin boundaries BEFORE creating config to avoid validation error
        # The SchedulerConfig __post_init__ validates K_BINS matches BIN_BOUNDARIES length
        bin_boundaries = None
        if k_bins > 1 and scheduler_type == "multi_bin_dynamic":
            # Create temporary config just for sampling
            sample_cfg = SchedulerConfig(
                NUM_REQUESTS=min(5000, num_requests),
                SEED=42,
                DATASET_PATH=dataset_path,
                WORKLOAD_SOURCE="burstgpt_dataset",
                USE_REAL_TIMESTAMPS=use_real_timestamps,
                RPS_SCALING=rps_scaling,
            )
            sample_requests = generate_workload(sample_cfg)
            predicted_lengths = [r.predicted_output_len for r in sample_requests]
            bin_boundaries = compute_equal_mass_boundaries(predicted_lengths, k_bins)
        
        # Create configuration with correct bin boundaries
        cfg = SchedulerConfig(
            NUM_GPUS=num_gpus,
            K_BINS=k_bins,
            NUM_REQUESTS=num_requests,
            SEED=42,
            D_SLA=d_sla,
            DATASET_PATH=dataset_path,
            WORKLOAD_SOURCE="burstgpt_dataset",
            USE_REAL_TIMESTAMPS=use_real_timestamps,
            RPS_SCALING=rps_scaling,
            USE_EQUAL_MASS_BINS=True,
            USE_REAL_CALIBRATION=True,
            CALIBRATION_CSV_PATH=calibration_csv,
            BIN_BOUNDARIES=bin_boundaries if bin_boundaries else [(1, 27), (27, 101), (101, 188), (188, 10000)]
        )
        
        # Generate workload
        print(f"Loading {num_requests:,} requests from dataset...")
        requests = generate_workload(cfg)
        
        # Run simulation
        print(f"Running {scheduler_type} simulation...")
        simulator = Simulator(cfg, requests, scheduler_type)
        completed_requests = simulator.run()
        
        # Compute metrics (pass d_sla for paper-faithful SLA evaluation)
        metrics = compute_metrics(completed_requests, d_sla=d_sla)
        gpu_stats = simulator.get_gpu_stats()
        gpu_metrics = compute_gpu_utilization(gpu_stats)
        batch_stats = compute_batch_statistics(completed_requests)
        
        # Merge all metrics
        metrics.update(gpu_metrics)
        metrics.update(batch_stats)
        
        execution_time = time.time() - start_time
        
        # Build result dictionary
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'use_real_timestamps': use_real_timestamps,
            'rps_scaling': rps_scaling if not use_real_timestamps else None,
            'd_sla': d_sla,
            'k_bins': cfg.K_BINS if scheduler_type == "multi_bin_dynamic" else 1,
            'sla_violation_rate': metrics['sla_violation_rate'],
            'capacity_qps_under_sla': metrics['capacity_qps_under_sla'],
            'throughput_requests_per_sec': metrics['throughput_requests_per_sec'],
            'throughput_tokens_per_sec': metrics['throughput_tokens_per_sec'],
            'avg_latency': metrics['avg_latency'],
            'p50_latency': metrics['p50_latency'],
            'p95_latency': metrics['p95_latency'],
            'p99_latency': metrics['p99_latency'],
            'max_latency': metrics['max_latency'],
            'avg_queueing_delay': metrics['avg_queueing_delay'],
            'avg_service_time': metrics['avg_service_time'],
            'avg_gpu_utilization': metrics['avg_utilization'],
            'min_gpu_utilization': metrics['min_utilization'],
            'max_gpu_utilization': metrics['max_utilization'],
            'num_batches': metrics['num_batches'],
            'avg_batch_size': metrics['avg_batch_size'],
            'min_batch_size': metrics['min_batch_size'],
            'max_batch_size': metrics['max_batch_size'],
            'total_time': metrics['total_time'],
            'total_tokens': metrics['total_tokens'],
            'num_completed': len(completed_requests),
            'execution_time_seconds': execution_time,
            'status': 'success'
        }
        
        print(f"\n[SUCCESS] COMPLETED in {execution_time:.1f}s")
        print(f"  SLA Violations: {result['sla_violation_rate']*100:.1f}%")
        print(f"  Capacity QPS:   {result['capacity_qps_under_sla']:.2f}")
        print(f"  Avg Latency:    {result['avg_latency']:.3f}s")
        print(f"  GPU Util:       {result['avg_gpu_utilization']*100:.1f}%")
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n[FAILED] Error: {str(e)}")
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'use_real_timestamps': use_real_timestamps,
            'rps_scaling': rps_scaling if not use_real_timestamps else None,
            'd_sla': d_sla,
            'execution_time_seconds': execution_time,
            'status': 'failed',
            'error': str(e)
        }


def step1_request_scaling(args, all_results):
    """
    Step 1: Request Volume Scaling Stress Test (1K -> 1M, 10x each time)
    Tests algorithm ability to handle increasing request volumes:
      - static_fifo: 1 GPU (baseline)
      - dynamic_no_bins: 1 GPU (dynamic batching)
      - multi_bin_dynamic: 1 GPU, 2 GPUs, 4 GPUs (test scalability)
    """
    print("\n" + "="*80)
    print("STEP 1: REQUEST SCALING STRESS TEST (1K -> 1M, 10x increments)")
    print("="*80)
    
    # Generate request counts: 1K -> 10K -> 100K -> 1M (10x scaling)
    request_counts = []
    current = 1_000
    while current <= args.max_requests:
        request_counts.append(current)
        current *= 10
    
    print(f"\nRequest volumes: {[f'{n:,}' for n in request_counts]}")
    print(f"Schedulers: static_fifo (1 GPU), dynamic_no_bins (1 GPU), multi_bin_dynamic (1/2/4 GPUs)")
    print()
    
    for num_requests in request_counts:
        # Baseline schedulers (1 GPU each)
        for scheduler_type in ["static_fifo", "dynamic_no_bins"]:
            result = run_single_test(
                num_requests=num_requests,
                num_gpus=1,
                scheduler_type=scheduler_type,
                use_real_timestamps=args.use_real_timestamps,
                rps_scaling=args.rps_scaling,
                d_sla=args.d_sla,
                dataset_path=args.dataset,
                calibration_csv=args.calibration,
                k_bins=4
            )
            all_results.append(result)
        
        # Multi-bin with different GPU counts
        for num_gpus in [1, 2, 4]:
            result = run_single_test(
                num_requests=num_requests,
                num_gpus=num_gpus,
                scheduler_type="multi_bin_dynamic",
                use_real_timestamps=args.use_real_timestamps,
                rps_scaling=args.rps_scaling,
                d_sla=args.d_sla,
                dataset_path=args.dataset,
                calibration_csv=args.calibration,
                k_bins=4
            )
            all_results.append(result)


def step2_gpu_scaling(args, all_results):
    """
    Step 2: GPU Scaling Stress Test (1M requests, 1-100 GPUs)
    Tests multi_bin_dynamic's ability to assign tasks across large GPU pools.
    GPU progression: 1, 2, 4, 8, 16, 32, 64, 100
    """
    print("\n" + "="*80)
    print("STEP 2: GPU SCALING STRESS TEST (1M requests, 1-100 GPUs)")
    print("="*80)
    print(f"\nScheduler: multi_bin_dynamic only")
    print(f"Fixed: 1,000,000 requests (maximum scale), K=4 bins")
    
    # GPU counts: 1, 2, 4, 8, 16, 32, 64, 100
    gpu_counts = [1, 2, 4, 8, 16, 32, 64, 100]
    gpu_counts = [n for n in gpu_counts if n <= args.max_gpus]
    
    print(f"GPU counts: {gpu_counts}")
    print()
    
    scheduler_type = "multi_bin_dynamic"
    num_requests = 1_000_000
    
    for num_gpus in gpu_counts:
        result = run_single_test(
            num_requests=num_requests,
            num_gpus=num_gpus,
            scheduler_type=scheduler_type,
            use_real_timestamps=args.use_real_timestamps,
            rps_scaling=args.rps_scaling,
            d_sla=args.d_sla,
            dataset_path=args.dataset,
            calibration_csv=args.calibration,
            k_bins=4
        )
        
        all_results.append(result)


def step3_kbins_sensitivity(args, all_results, best_gpu_count):
    """
    Step 3: K-Bins Sensitivity Analysis (Best GPU config from Step 2)
    Tests optimal bin partitioning strategy for multi_bin_dynamic.
    K values: 1, 2, 4, 8, 16, 32
    """
    print("\n" + "="*80)
    print(f"STEP 3: K-BINS SENSITIVITY ANALYSIS (GPU={best_gpu_count}, 1M requests)")
    print("="*80)
    print(f"\nScheduler: multi_bin_dynamic only")
    print(f"Fixed: 1,000,000 requests, {best_gpu_count} GPUs (best from Step 2)")
    
    # K-bins values: 1, 2, 4, 8, 16, 32
    k_values = [1, 2, 4, 8, 16, 32]
    
    print(f"K-bins values to test: {k_values}")
    print()
    
    scheduler_type = "multi_bin_dynamic"
    num_requests = 1_000_000
    
    for k_bins in k_values:
        result = run_single_test(
            num_requests=num_requests,
            num_gpus=best_gpu_count,
            scheduler_type=scheduler_type,
            use_real_timestamps=args.use_real_timestamps,
            rps_scaling=args.rps_scaling,
            d_sla=args.d_sla,
            dataset_path=args.dataset,
            calibration_csv=args.calibration,
            k_bins=k_bins
        )
        
        all_results.append(result)


def create_analysis_summary(all_results, args):
    """Create comprehensive analysis for research paper"""
    print("\n" + "="*80)
    print("CREATING ANALYSIS SUMMARY FOR RESEARCH PAPER")
    print("="*80)
    
    # Separate results by test step
    step1_results = [r for r in all_results 
                     if r['scheduler_type'] in ['static_fifo', 'dynamic_no_bins'] 
                     or (r['scheduler_type'] == 'multi_bin_dynamic' and r['num_requests'] < 1_000_000)]
    
    step2_results = [r for r in all_results 
                     if r['scheduler_type'] == 'multi_bin_dynamic' 
                     and r['num_requests'] == 1_000_000
                     and r.get('k_bins', 4) == 4]  # Exclude Step 3 (k-bins variations)
    
    step3_results = [r for r in all_results 
                     if r['scheduler_type'] == 'multi_bin_dynamic' 
                     and r['num_requests'] == 1_000_000
                     and r.get('k_bins', 4) != 4]  # Only k-bins variations
    
    analysis = {
        'experiment_metadata': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': args.dataset,
            'calibration': args.calibration,
            'rps_scaling': args.rps_scaling,
            'sla_deadline_seconds': args.d_sla,
            'total_tests': len(all_results),
            'step1_tests': len(step1_results),
            'step2_tests': len(step2_results),
            'step3_tests': len(step3_results),
            'successful_tests': len([r for r in all_results if r['status'] == 'success']),
        },
        'step1_request_scaling': analyze_request_scaling(step1_results),
        'step2_gpu_scaling': analyze_gpu_scaling(step2_results),
        'step3_kbins_sensitivity': analyze_kbins_sensitivity(step3_results),
        'scheduler_comparison': compare_schedulers(step1_results),
        'key_findings': generate_key_findings(step1_results, step2_results, step3_results),
    }
    
    # Save to JSON
    analysis_path = args.output.replace('.csv', '_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Analysis summary saved to: {analysis_path}")
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    for finding in analysis['key_findings']:
        print(f"  • {finding}")
    print()


def analyze_request_scaling(results):
    """Analyze how schedulers handle increasing request volumes"""
    analysis = {}
    
    for scheduler in ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']:
        scheduler_results = [r for r in results 
                           if r['scheduler_type'] == scheduler and r['status'] == 'success']
        if not scheduler_results:
            continue
        
        scheduler_results.sort(key=lambda x: x['num_requests'])
        
        analysis[scheduler] = {
            'request_volumes': [r['num_requests'] for r in scheduler_results],
            'sla_violations_pct': [r['sla_violation_rate'] * 100 for r in scheduler_results],
            'capacity_qps': [r['capacity_qps_under_sla'] for r in scheduler_results],
            'avg_latency_sec': [r['avg_latency'] for r in scheduler_results],
            'p95_latency_sec': [r['p95_latency'] for r in scheduler_results],
            'throughput_req_per_sec': [r['throughput_requests_per_sec'] for r in scheduler_results],
            'gpu_utilization_pct': [r['avg_gpu_utilization'] * 100 for r in scheduler_results],
            'num_gpus': scheduler_results[0]['num_gpus'] if scheduler_results else 0,
        }
    
    return analysis


def analyze_gpu_scaling(results):
    """Analyze GPU scaling efficiency"""
    if not results:
        return {}
    
    results = [r for r in results if r['status'] == 'success']
    results.sort(key=lambda x: x['num_gpus'])
    
    if not results:
        return {}
    
    base = results[0]
    
    scaling_efficiency = []
    for curr in results[1:]:
        gpu_ratio = curr['num_gpus'] / base['num_gpus']
        qps_ratio = curr['capacity_qps_under_sla'] / base['capacity_qps_under_sla']
        efficiency = qps_ratio / gpu_ratio
        
        scaling_efficiency.append({
            'num_gpus': curr['num_gpus'],
            'gpu_ratio': gpu_ratio,
            'qps_ratio': qps_ratio,
            'efficiency': efficiency,
            'interpretation': 'superlinear' if efficiency > 1.05 else ('linear' if efficiency > 0.85 else 'sublinear')
        })
    
    # Identify bottleneck
    bottleneck = None
    for i in range(1, len(results)):
        prev, curr = results[i-1], results[i]
        qps_gain = (curr['capacity_qps_under_sla'] - prev['capacity_qps_under_sla']) / prev['capacity_qps_under_sla']
        
        if qps_gain < 0.10:  # Less than 10% improvement
            bottleneck = {
                'gpu_count': curr['num_gpus'],
                'qps_gain_pct': qps_gain * 100,
                'likely_cause': 'arrival_rate_limited' if curr['avg_gpu_utilization'] < 0.3 else 'scheduling_overhead',
            }
            break
    
    return {
        'gpu_counts': [r['num_gpus'] for r in results],
        'capacity_qps': [r['capacity_qps_under_sla'] for r in results],
        'sla_violations_pct': [r['sla_violation_rate'] * 100 for r in results],
        'avg_latency_sec': [r['avg_latency'] for r in results],
        'gpu_utilization_pct': [r['avg_gpu_utilization'] * 100 for r in results],
        'scaling_efficiency': scaling_efficiency,
        'bottleneck_analysis': {
            'detected': bottleneck is not None,
            'details': bottleneck if bottleneck else 'No bottleneck detected in tested range',
            'max_qps': max(r['capacity_qps_under_sla'] for r in results),
            'optimal_gpu_count': max(results, key=lambda x: x['capacity_qps_under_sla'])['num_gpus'],
        }
    }


def compare_schedulers(results):
    """Compare schedulers at each request volume"""
    comparison = {}
    
    request_counts = sorted(set(r['num_requests'] for r in results if r['status'] == 'success'))
    
    for req_count in request_counts:
        req_results = [r for r in results if r['num_requests'] == req_count and r['status'] == 'success']
        
        comp_key = f'{req_count:,}_requests'
        comparison[comp_key] = {}
        
        for scheduler in ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']:
            result = next((r for r in req_results if r['scheduler_type'] == scheduler), None)
            if result:
                comparison[comp_key][scheduler] = {
                    'sla_violations_pct': result['sla_violation_rate'] * 100,
                    'capacity_qps': result['capacity_qps_under_sla'],
                    'avg_latency_sec': result['avg_latency'],
                    'p95_latency_sec': result['p95_latency'],
                    'num_gpus': result['num_gpus'],
                    'gpu_utilization_pct': result['avg_gpu_utilization'] * 100,
                }
        
        # Calculate improvements
        if 'multi_bin_dynamic' in comparison[comp_key]:
            mb = comparison[comp_key]['multi_bin_dynamic']
            
            for baseline in ['static_fifo', 'dynamic_no_bins']:
                if baseline in comparison[comp_key]:
                    bl = comparison[comp_key][baseline]
                    
                    sla_improvement = ((bl['sla_violations_pct'] - mb['sla_violations_pct']) / 
                                     bl['sla_violations_pct'] * 100) if bl['sla_violations_pct'] > 0 else 0
                    
                    qps_improvement = mb['capacity_qps'] / bl['capacity_qps'] if bl['capacity_qps'] > 0 else float('inf')
                    
                    latency_improvement = ((bl['avg_latency_sec'] - mb['avg_latency_sec']) / 
                                         bl['avg_latency_sec'] * 100) if bl['avg_latency_sec'] > 0 else 0
                    
                    comparison[comp_key][f'multi_bin_vs_{baseline}'] = {
                        'sla_improvement_pct': sla_improvement,
                        'qps_improvement_ratio': qps_improvement,
                        'latency_reduction_pct': latency_improvement,
                    }
    
    return comparison


def analyze_kbins_sensitivity(results):
    """Analyze K-bins sensitivity for optimal partitioning"""
    if not results:
        return {}
    
    results = [r for r in results if r['status'] == 'success']
    results.sort(key=lambda x: x.get('k_bins', 4))
    
    if not results:
        return {}
    
    # Find optimal K value
    best_result = min(results, key=lambda x: x['sla_violation_rate'])
    
    return {
        'k_values': [r.get('k_bins', 4) for r in results],
        'sla_violations_pct': [r['sla_violation_rate'] * 100 for r in results],
        'capacity_qps': [r['capacity_qps_under_sla'] for r in results],
        'avg_latency_sec': [r['avg_latency'] for r in results],
        'p95_latency_sec': [r['p95_latency'] for r in results],
        'gpu_utilization_pct': [r['avg_gpu_utilization'] * 100 for r in results],
        'optimal_k': {
            'k_bins': best_result.get('k_bins', 4),
            'sla_violations_pct': best_result['sla_violation_rate'] * 100,
            'capacity_qps': best_result['capacity_qps_under_sla'],
            'avg_latency_sec': best_result['avg_latency'],
            'num_gpus': best_result['num_gpus'],
        },
        'interpretation': {
            'best_k': best_result.get('k_bins', 4),
            'performance_range': {
                'min_qps': min(r['capacity_qps_under_sla'] for r in results),
                'max_qps': max(r['capacity_qps_under_sla'] for r in results),
                'variance_pct': (max(r['capacity_qps_under_sla'] for r in results) - 
                               min(r['capacity_qps_under_sla'] for r in results)) / 
                               max(r['capacity_qps_under_sla'] for r in results) * 100,
            }
        }
    }


def generate_key_findings(step1_results, step2_results, step3_results):
    """Generate key findings for research paper"""
    findings = []
    
    # Step 1 findings
    step1_success = [r for r in step1_results if r['status'] == 'success']
    if step1_success:
        # Find best multi-bin result
        mb_results = [r for r in step1_success if r['scheduler_type'] == 'multi_bin_dynamic']
        if mb_results:
            best_mb = min(mb_results, key=lambda x: x['sla_violation_rate'])
            findings.append(
                f"Multi-bin achieves {best_mb['sla_violation_rate']*100:.1f}% SLA violations "
                f"with {best_mb['num_requests']:,} requests on {best_mb['num_gpus']} GPUs"
            )
        
        # Compare to baselines at largest volume
        max_requests = max(r['num_requests'] for r in step1_success)
        max_req_results = [r for r in step1_success if r['num_requests'] == max_requests]
        
        mb = next((r for r in max_req_results if r['scheduler_type'] == 'multi_bin_dynamic'), None)
        sf = next((r for r in max_req_results if r['scheduler_type'] == 'static_fifo'), None)
        dn = next((r for r in max_req_results if r['scheduler_type'] == 'dynamic_no_bins'), None)
        
        if mb and sf:
            qps_ratio = mb['capacity_qps_under_sla'] / sf['capacity_qps_under_sla']
            findings.append(
                f"At {max_requests:,} requests: Multi-bin achieves {qps_ratio:.2f}x higher QPS than static_fifo"
            )
        
        if mb and dn:
            latency_ratio = dn['avg_latency'] / mb['avg_latency']
            findings.append(
                f"Multi-bin reduces average latency by {latency_ratio:.1f}x compared to dynamic_no_bins"
            )
    
    # Step 2 findings
    step2_success = [r for r in step2_results if r['status'] == 'success']
    if len(step2_success) >= 2:
        step2_success.sort(key=lambda x: x['num_gpus'])
        
        # Find saturation point
        for i in range(1, len(step2_success)):
            prev, curr = step2_success[i-1], step2_success[i]
            qps_gain = (curr['capacity_qps_under_sla'] - prev['capacity_qps_under_sla']) / prev['capacity_qps_under_sla']
            
            if qps_gain < 0.10:
                findings.append(
                    f"GPU scaling saturates at ~{prev['num_gpus']} GPUs "
                    f"({curr['capacity_qps_under_sla']:.2f} QPS with {curr['num_gpus']} GPUs)"
                )
                break
        
        # Overall GPU scaling
        base, peak = step2_success[0], step2_success[-1]
        gpu_ratio = peak['num_gpus'] / base['num_gpus']
        qps_ratio = peak['capacity_qps_under_sla'] / base['capacity_qps_under_sla']
        efficiency = qps_ratio / gpu_ratio
        
        findings.append(
            f"Scaling from {base['num_gpus']} to {peak['num_gpus']} GPUs: "
            f"{qps_ratio:.2f}x QPS improvement ({efficiency*100:.0f}% efficiency)"
        )
    
    # Step 3 findings
    step3_success = [r for r in step3_results if r['status'] == 'success']
    if step3_success:
        best_k = min(step3_success, key=lambda x: x['sla_violation_rate'])
        worst_k = max(step3_success, key=lambda x: x['sla_violation_rate'])
        
        findings.append(
            f"Optimal K-bins: {best_k.get('k_bins', 4)} "
            f"({best_k['capacity_qps_under_sla']:.2f} QPS, {best_k['sla_violation_rate']*100:.1f}% SLA violations)"
        )
        
        if len(step3_success) > 1:
            qps_variance = (max(r['capacity_qps_under_sla'] for r in step3_success) - 
                           min(r['capacity_qps_under_sla'] for r in step3_success)) / \
                           max(r['capacity_qps_under_sla'] for r in step3_success) * 100
            
            findings.append(
                f"K-bins sensitivity: {qps_variance:.1f}% performance variance across K=[1,2,4,8,16,32]"
            )
    
    return findings


def save_results_to_csv(all_results, output_path):
    """Save results to CSV files"""
    print("\n" + "="*80)
    print(f"SAVING RESULTS TO: {output_path}")
    print("="*80)
    
    df = pd.DataFrame(all_results)
    
    # Separate all 3 steps
    step1_df = df[df['scheduler_type'].isin(['static_fifo', 'dynamic_no_bins']) | 
                  ((df['scheduler_type'] == 'multi_bin_dynamic') & (df['num_requests'] < 1_000_000))]
    
    step2_df = df[(df['scheduler_type'] == 'multi_bin_dynamic') & 
                  (df['num_requests'] == 1_000_000) &
                  (df.get('k_bins', 4) == 4)]
    
    step3_df = df[(df['scheduler_type'] == 'multi_bin_dynamic') & 
                  (df['num_requests'] == 1_000_000) &
                  (df.get('k_bins', 4) != 4)]
    
    # Save separate files
    step1_path = output_path.replace('.csv', '_step1_request_scaling.csv')
    step2_path = output_path.replace('.csv', '_step2_gpu_scaling.csv')
    step3_path = output_path.replace('.csv', '_step3_kbins_sensitivity.csv')
    all_path = output_path.replace('.csv', '_all_results.csv')
    
    step1_df.to_csv(step1_path, index=False)
    step2_df.to_csv(step2_path, index=False)
    step3_df.to_csv(step3_path, index=False)
    df.to_csv(all_path, index=False)
    
    print(f"Step 1 (Request Scaling):   {step1_path}")
    print(f"Step 2 (GPU Scaling):       {step2_path}")
    print(f"Step 3 (K-bins Sensitivity): {step3_path}")
    print(f"All Results:                {all_path}")
    print(f"\n[SUCCESS] Results saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Stress Testing Suite')
    
    parser.add_argument('--max-requests', type=int, default=1_000_000,
                       help='Maximum requests for Step 1 (default: 1M)')
    parser.add_argument('--max-gpus', type=int, default=100,
                       help='Maximum GPUs for Step 2 (default: 100)')
    parser.add_argument('--use-real-timestamps', action='store_true', default=False,
                       help='Use real timestamps from dataset (default: False, uses RPS scaling)')
    parser.add_argument('--use-rps-scaling', action='store_true', dest='use_rps_scaling', default=True,
                       help='Use RPS scaling for stress testing (default: True)')
    parser.add_argument('--rps-scaling', type=float, default=200.0,
                       help='RPS scaling factor (default: 200.0 for stress testing)')
    parser.add_argument('--d-sla', type=float, default=0.05,
                       help='Per-token decode latency SLA (TBT) in seconds (default: 0.05 = 50ms)')
    parser.add_argument('--dataset', type=str, default='data/BurstGPT_sample.csv',
                       help='Path to BurstGPT dataset')
    parser.add_argument('--calibration', type=str, default='data/qwen3_1_7b_latency_grid.csv',
                       help='Path to GPU calibration file')
    parser.add_argument('--output', type=str, default='comprehensive_stress_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--step1-only', action='store_true',
                       help='Run only Step 1 (request scaling)')
    parser.add_argument('--step2-only', action='store_true',
                       help='Run only Step 2 (GPU scaling)')
    parser.add_argument('--step3-only', action='store_true',
                       help='Run only Step 3 (K-bins sensitivity)')
    parser.add_argument('--best-gpu-count', type=int, default=None,
                       help='For Step 3: specify best GPU count from Step 2 (if known)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*80)
    print("COMPREHENSIVE STRESS TEST SUITE FOR RESEARCH PAPER (3-STEP PLAN)")
    print("="*80)
    print()
    print("Configuration:")
    print(f"  Dataset:           {args.dataset}")
    print(f"  Calibration:       {args.calibration}")
    print(f"  Max Requests:      {args.max_requests:,}")
    print(f"  Max GPUs:          {args.max_gpus}")
    if args.use_real_timestamps:
        print(f"  Arrival Pattern:   REAL timestamps (realistic benchmarking)")
    else:
        print(f"  Arrival Pattern:   RPS Scaling {args.rps_scaling}x (stress testing)")
    print(f"  SLA Deadline:      {args.d_sla}s")
    print(f"  Output:            {args.output}")
    print()
    
    overall_start = time.time()
    all_results = []
    
    # Step 1: Request volume scaling
    if not args.step2_only and not args.step3_only:
        step1_request_scaling(args, all_results)
    
    # Step 2: GPU scaling
    if not args.step1_only and not args.step3_only:
        step2_gpu_scaling(args, all_results)
    
    # Step 3: K-bins sensitivity analysis
    if not args.step1_only and not args.step2_only:
        # Determine best GPU count from Step 2 results
        step2_results = [r for r in all_results 
                        if r['scheduler_type'] == 'multi_bin_dynamic' 
                        and r['num_requests'] == 1_000_000
                        and r.get('k_bins', 4) == 4
                        and r['status'] == 'success']
        
        if args.best_gpu_count:
            best_gpu_count = args.best_gpu_count
            print(f"\n[INFO] Using specified GPU count: {best_gpu_count}")
        elif step2_results:
            # Find GPU count with best QPS
            best_result = max(step2_results, key=lambda x: x['capacity_qps_under_sla'])
            best_gpu_count = best_result['num_gpus']
            print(f"\n[INFO] Best GPU count from Step 2: {best_gpu_count} GPUs "
                  f"({best_result['capacity_qps_under_sla']:.2f} QPS)")
        else:
            # Default fallback
            best_gpu_count = 32
            print(f"\n[WARNING] No Step 2 results found, using default GPU count: {best_gpu_count}")
        
        step3_kbins_sensitivity(args, all_results, best_gpu_count)
    
    # Save results
    save_results_to_csv(all_results, args.output)
    
    # Create analysis
    create_analysis_summary(all_results, args)
    
    # Summary
    total_time = time.time() - overall_start
    print("="*80)
    print("COMPREHENSIVE STRESS TEST COMPLETE (3-STEP PLAN)")
    print("="*80)
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"Total tests run:      {len(all_results)}")
    print(f"Successful tests:     {len([r for r in all_results if r['status'] == 'success'])}")
    print(f"Results saved to:     {args.output}")
    print()
    
    # Auto-generate graphs
    print("="*80)
    print("GENERATING GRAPHS AND ANALYSIS")
    print("="*80)
    try:
        output_dir = Path(args.output).parent / 'plots'
        plot_script = Path(__file__).parent / 'analyze_and_plot_results.py'
        
        print(f"Running: python {plot_script} --input {args.output} --output-dir {output_dir}")
        print()
        
        result = subprocess.run(
            [sys.executable, str(plot_script), '--input', args.output, '--output-dir', str(output_dir)],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print("✓ GRAPHS GENERATED SUCCESSFULLY")
            print("="*80)
            print(f"Location: {output_dir}/")
            print("\nGenerated visualizations:")
            print("  • request_scaling_capacity.png")
            print("  • request_scaling_latency.png")
            print("  • gpu_scaling_analysis.png")
            print("  • scaling_efficiency.png")
            print("  • scheduler_comparison_heatmap.png")
            print("  • throughput_latency_tradeoff.png")
            print("  • batch_size_analysis.png")
            print("  • summary_table.csv")
            print("  • analysis_report.txt")
            print()
        else:
            print(f"\n[WARNING] Graph generation failed with code {result.returncode}")
            print(f"You can manually generate graphs by running:")
            print(f"  python {plot_script} --input {args.output} --output-dir {output_dir}")
    except Exception as e:
        print(f"\n[WARNING] Could not auto-generate graphs: {e}")
        print(f"You can manually generate graphs by running:")
        print(f"  python scripts/analyze_and_plot_results.py --input {args.output}")
    print()


if __name__ == "__main__":
    main()
