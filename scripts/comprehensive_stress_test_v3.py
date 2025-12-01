#!/usr/bin/env python3
"""
Comprehensive Stress Test v3 - Adaptive RPS Scaling

This script implements the full 2-step stress test with:
1. Adaptive RPS scaling based on GPU count (to maintain effective batching)
2. Grid search for optimal GPU × Bin combinations
3. Method comparison (Static, Dynamic, Multi-Bin)

Key insight: RPS must scale with GPU count to maintain batch efficiency.
Formula: RPS = base_rps × GPU_factor where GPU_factor ensures batches ≥ 10

SLA Configuration (Gemini-Like):
- Token SLA: 30ms (loose, for stress test differentiation)
- Request SLA: 20s (loose, for long generation tolerance)
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# SLA Thresholds (Gemini-Like - LOOSE mode for stress testing)
D_SLA_TOKEN = 0.030      # 30ms per-token (loose)
D_SLA_REQUEST = 20.0     # 20s request latency (loose)

# Fixed RPS scaling for fair comparison across all configurations
# Native BurstGPT RPS: 0.27 req/s → Scaled RPS: 54 req/s (200x scaling)
RPS_SCALING = 200.0


def run_single_experiment(
    scheduler_type: str,
    num_requests: int,
    num_gpus: int,
    k_bins: int,
    rps_scaling: float = RPS_SCALING,
    verbose: bool = True
) -> Dict:
    """Run a single experiment with given configuration."""
    
    start_time = time.time()
    
    try:
        # Use fixed RPS scaling for fair comparison
        
        # Configure scheduler
        cfg = SchedulerConfig(
            NUM_GPUS=num_gpus,
            K_BINS=k_bins,
            NUM_REQUESTS=num_requests,
            EXPERIMENT_MODE=scheduler_type,
            B_MIN=1,
            B_MAX=128,
            D_SLA=D_SLA_TOKEN,
            D_SLA_TOKEN=D_SLA_TOKEN,
            D_SLA_REQUEST=D_SLA_REQUEST,
            LATENCY_EPSILON=0.010,  # 10ms tolerance
            USE_REAL_CALIBRATION=True,
            CALIBRATION_CSV_PATH=CALIBRATION_CSV,
            WORKLOAD_SOURCE="burstgpt_dataset",
            DATASET_PATH=DATASET_PATH,
            USE_REAL_TIMESTAMPS=False,
            RPS_SCALING=rps_scaling,
            SEED=42,
        )
        
        if scheduler_type == 'static_fifo':
            cfg.B_FIXED = 8  # Fixed batch size for static
        
        # Generate workload
        requests = generate_workload(cfg)
        
        # Run simulation (correct interface: cfg, requests, scheduler_type)
        simulator = Simulator(cfg, requests, scheduler_type)
        completed_requests = simulator.run()
        
        # Compute metrics
        metrics = compute_metrics(
            completed_requests, 
            d_sla_token=D_SLA_TOKEN,
            d_sla_request=D_SLA_REQUEST
        )
        
        gpu_stats = simulator.get_gpu_stats()
        gpu_util = compute_gpu_utilization(gpu_stats)
        batch_stats = compute_batch_statistics(completed_requests)
        
        execution_time = time.time() - start_time
        
        # Compute actual RPS achieved
        actual_rps = num_requests / metrics.get('total_time', 1) if metrics.get('total_time', 0) > 0 else 0
        
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'k_bins': k_bins,
            'rps_scaling': rps_scaling,
            'target_rps': rps_scaling * 0.27,
            'actual_rps': actual_rps,
            'd_sla_token': D_SLA_TOKEN * 1000,  # in ms
            'd_sla_request': D_SLA_REQUEST,
            'throughput_tokens_per_sec': metrics.get('throughput_tokens_per_sec', 0),
            'throughput_requests_per_sec': metrics.get('throughput_requests_per_sec', 0),
            'avg_latency': metrics.get('avg_latency', 0),
            'p50_latency': metrics.get('p50_latency', 0),
            'p95_latency': metrics.get('p95_latency', 0),
            'p99_latency': metrics.get('p99_latency', 0),
            'avg_tbt_ms': metrics.get('avg_tbt', 0) * 1000,
            'p95_tbt_ms': metrics.get('p95_tbt', 0) * 1000,
            'sla_violation_rate_token': metrics.get('sla_violation_rate_token', 0),
            'sla_violation_rate_request': metrics.get('sla_violation_rate_request', 0),
            'avg_gpu_utilization': gpu_util.get('avg_utilization', 0),
            'avg_batch_size': batch_stats.get('avg_batch_size', 0),
            'avg_queueing_delay': metrics.get('avg_queueing_delay', 0),
            'avg_service_time': metrics.get('avg_service_time', 0),
            'total_time': metrics.get('total_time', 0),
            'execution_time_seconds': time.time() - start_time,
            'status': 'success'
        }
        
        if verbose:
            token_pass = (1 - result['sla_violation_rate_token']) * 100
            req_pass = (1 - result['sla_violation_rate_request']) * 100
            print(f"  -> Token: {token_pass:.1f}%, Req: {req_pass:.1f}%, "
                  f"Batch: {result['avg_batch_size']:.1f}, GPU: {result['avg_gpu_utilization']*100:.1f}% [{execution_time:.1f}s]")
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        if verbose:
            print(f"  -> FAILED: {str(e)[:80]}")
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'k_bins': k_bins,
            'execution_time_seconds': execution_time,
            'status': 'failed',
            'error': str(e)
        }


def run_step1_grid_search(
    request_counts: List[int],
    gpu_counts: List[int],
    bin_counts: List[int],
    output_file: str
) -> pd.DataFrame:
    """
    Step 1: Grid search for Multi-Bin + Dynamic only.
    
    Find optimal (GPU, K_BINS) combination for each workload size.
    """
    print("="*80)
    print("STEP 1: GRID SEARCH - Multi-Bin + Dynamic")
    print("="*80)
    print(f"Request counts: {request_counts}")
    print(f"GPU counts: {gpu_counts}")
    print(f"Bin counts: {bin_counts}")
    print(f"SLA: Token={D_SLA_TOKEN*1000}ms, Request={D_SLA_REQUEST}s")
    print()
    
    results = []
    total_configs = len(request_counts) * len(gpu_counts) * len(bin_counts)
    config_num = 0
    
    for num_requests in request_counts:
        print(f"\n{'='*60}")
        print(f"WORKLOAD: {num_requests:,} requests")
        print(f"{'='*60}")
        
        for num_gpus in gpu_counts:
            for k_bins in bin_counts:
                config_num += 1
                
                # Use fixed RPS scaling (200x) for fair comparison
                effective_rps = RPS_SCALING * 0.27
                
                # Skip severely overloaded configurations (1-4 GPUs with 100K+ requests)
                # These will have ~100% GPU utilization and 0% request SLA anyway
                if num_requests >= 100000 and num_gpus <= 4:
                    print(f"\n[{config_num}/{total_configs}] GPUs={num_gpus}, K={k_bins} [SKIPPED - overloaded config]")
                    continue
                
                print(f"\n[{config_num}/{total_configs}] GPUs={num_gpus}, K={k_bins}, RPS={effective_rps:.0f}")
                
                result = run_single_experiment(
                    scheduler_type='multi_bin_dynamic',
                    num_requests=num_requests,
                    num_gpus=num_gpus,
                    k_bins=k_bins,
                    verbose=True
                )
                results.append(result)
                
                # Save incrementally
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
    
    print(f"\n[COMPLETE] Step 1 results saved to {output_file}")
    return pd.DataFrame(results)


def analyze_optimal_configs(df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Analyze grid search results to find optimal configurations.
    
    Criteria:
    1. Maximize combined SLA pass rate (token + request)
    2. Consider diminishing returns (extra GPUs/bins not worth it if <5% improvement)
    """
    optimal = {}
    
    for num_requests in df['num_requests'].unique():
        subset = df[(df['num_requests'] == num_requests) & (df['status'] == 'success')].copy()
        
        if subset.empty:
            continue
        
        # Compute combined score (token pass + request pass)
        subset['token_pass'] = (1 - subset['sla_violation_rate_token']) * 100
        subset['req_pass'] = (1 - subset['sla_violation_rate_request']) * 100
        subset['combined_score'] = subset['token_pass'] + subset['req_pass']
        
        # Find best configuration
        best_idx = subset['combined_score'].idxmax()
        best = subset.loc[best_idx]
        
        optimal[num_requests] = {
            'num_gpus': int(best['num_gpus']),
            'k_bins': int(best['k_bins']),
            'token_pass': best['token_pass'],
            'req_pass': best['req_pass'],
            'combined_score': best['combined_score'],
            'avg_batch_size': best['avg_batch_size'],
            'gpu_utilization': best['avg_gpu_utilization']
        }
        
        print(f"\n{num_requests:,} requests: Best = GPUs={int(best['num_gpus'])}, K={int(best['k_bins'])}")
        print(f"  Token: {best['token_pass']:.1f}%, Request: {best['req_pass']:.1f}%")
        print(f"  Batch Size: {best['avg_batch_size']:.1f}, GPU Util: {best['avg_gpu_utilization']*100:.1f}%")
    
    return optimal


def run_step2_comparison(
    request_counts: List[int],
    optimal_configs: Dict[int, Dict],
    output_file: str
) -> pd.DataFrame:
    """
    Step 2: Compare all methods using optimal configurations.
    
    Methods:
    1. Static FIFO (1 GPU, no bins) - baseline
    2. Dynamic No-Bins (1 GPU) - dynamic batching only
    3. Multi-Bin Dynamic (1 GPU) - bins + dynamic batching
    4. Multi-Bin Dynamic (optimal) - best from Step 1
    """
    print("\n" + "="*80)
    print("STEP 2: METHOD COMPARISON")
    print("="*80)
    print(f"SLA: Token={D_SLA_TOKEN*1000}ms, Request={D_SLA_REQUEST}s")
    print()
    
    results = []
    
    for num_requests in request_counts:
        print(f"\n{'='*60}")
        print(f"WORKLOAD: {num_requests:,} requests")
        print(f"{'='*60}")
        
        # Get optimal config for this workload
        opt = optimal_configs.get(num_requests, {'num_gpus': 4, 'k_bins': 8})
        
        # Method 1: Static FIFO (1 GPU, K=1)
        print(f"\n[1/4] Static FIFO (1 GPU)")
        result = run_single_experiment(
            scheduler_type='static_fifo',
            num_requests=num_requests,
            num_gpus=1,
            k_bins=1,
            verbose=True
        )
        result['method'] = 'Static_FIFO_1GPU'
        results.append(result)
        
        # Method 2: Dynamic No-Bins (1 GPU, K=1)
        print(f"\n[2/4] Dynamic No-Bins (1 GPU)")
        result = run_single_experiment(
            scheduler_type='dynamic_no_bins',
            num_requests=num_requests,
            num_gpus=1,
            k_bins=1,
            verbose=True
        )
        result['method'] = 'Dynamic_NoBins_1GPU'
        results.append(result)
        
        # Method 3: Multi-Bin Dynamic (1 GPU, K=8)
        print(f"\n[3/4] Multi-Bin Dynamic (1 GPU, K=8)")
        result = run_single_experiment(
            scheduler_type='multi_bin_dynamic',
            num_requests=num_requests,
            num_gpus=1,
            k_bins=8,
            verbose=True
        )
        result['method'] = 'MultiBin_1GPU_K8'
        results.append(result)
        
        # Method 4: Multi-Bin Dynamic (optimal from Step 1)
        opt_gpus = opt['num_gpus']
        opt_bins = opt['k_bins']
        effective_rps = RPS_SCALING * 0.27
        print(f"\n[4/4] Multi-Bin Optimal ({opt_gpus} GPUs, K={opt_bins}, RPS={effective_rps:.0f})")
        result = run_single_experiment(
            scheduler_type='multi_bin_dynamic',
            num_requests=num_requests,
            num_gpus=opt_gpus,
            k_bins=opt_bins,
            verbose=True
        )
        result['method'] = f'MultiBin_Optimal_{opt_gpus}GPU_K{opt_bins}'
        results.append(result)
        
        # Save incrementally
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
    
    print(f"\n[COMPLETE] Step 2 results saved to {output_file}")
    return pd.DataFrame(results)


def generate_plots(step1_df: pd.DataFrame, step2_df: pd.DataFrame, output_dir: str):
    """Generate analysis plots."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Step 1 Grid Search Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Step 1: Multi-Bin Grid Search Results\n(Token SLA=30ms, Request SLA=20s)', 
                 fontsize=14, fontweight='bold')
    
    for idx, num_requests in enumerate(sorted(step1_df['num_requests'].unique())[:4]):
        ax = axes[idx // 2, idx % 2]
        subset = step1_df[step1_df['num_requests'] == num_requests].copy()
        
        if subset.empty:
            continue
        
        subset['token_pass'] = (1 - subset['sla_violation_rate_token']) * 100
        
        # Create pivot table
        pivot = subset.pivot_table(
            values='token_pass', 
            index='num_gpus', 
            columns='k_bins', 
            aggfunc='mean'
        )
        
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('K Bins')
        ax.set_ylabel('Num GPUs')
        ax.set_title(f'{num_requests:,} Requests')
        
        # Add value annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Token SLA Pass %')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step1_grid_search.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Step 2 Method Comparison
    if not step2_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Step 2: Method Comparison\n(Token SLA=30ms, Request SLA=20s)',
                     fontsize=14, fontweight='bold')
        
        step2_df['token_pass'] = (1 - step2_df['sla_violation_rate_token']) * 100
        step2_df['req_pass'] = (1 - step2_df['sla_violation_rate_request']) * 100
        
        colors = {'Static_FIFO_1GPU': 'red', 'Dynamic_NoBins_1GPU': 'orange'}
        
        for metric, ax, ylabel in [('token_pass', axes[0], 'Token SLA Pass %'),
                                    ('req_pass', axes[1], 'Request SLA Pass %')]:
            
            for num_requests in sorted(step2_df['num_requests'].unique()):
                subset = step2_df[step2_df['num_requests'] == num_requests]
                methods = subset['method'].tolist()
                values = subset[metric].tolist()
                
                x = np.arange(len(methods))
                bars = ax.bar(x + 0.2 * list(step2_df['num_requests'].unique()).index(num_requests),
                             values, width=0.2, label=f'{num_requests:,}')
            
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 105)
            ax.axhline(y=95, color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax.legend(title='Requests')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'step2_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Stress Test v3')
    parser.add_argument('--step', type=int, choices=[1, 2, 3], default=3,
                       help='1=Grid search only, 2=Comparison only, 3=Both')
    parser.add_argument('--request-counts', type=int, nargs='+',
                       default=[1000, 10000, 100000, 1000000],
                       help='Request counts to test')
    parser.add_argument('--gpu-counts', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16, 32, 64, 100],
                       help='GPU counts to test')
    parser.add_argument('--bin-counts', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16, 32],
                       help='Bin counts to test')
    parser.add_argument('--output-dir', type=str, default='stress_test_v3_results',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: reduced configs for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Quick mode for testing
    if args.quick:
        args.request_counts = [1000, 10000]
        args.gpu_counts = [1, 4, 8]
        args.bin_counts = [1, 4, 8]
    
    step1_file = os.path.join(args.output_dir, 'step1_grid_search.csv')
    step2_file = os.path.join(args.output_dir, 'step2_comparison.csv')
    plots_dir = os.path.join(args.output_dir, 'plots')
    
    print("="*80)
    print("COMPREHENSIVE STRESS TEST v3 - Fixed RPS (200x)")
    print("="*80)
    print(f"Token SLA: {D_SLA_TOKEN*1000}ms | Request SLA: {D_SLA_REQUEST}s")
    print(f"RPS Scaling: {RPS_SCALING}x (effective: {RPS_SCALING * 0.27:.1f} req/s)")
    print()
    
    step1_df = None
    step2_df = None
    optimal_configs = {}
    
    # Step 1: Grid Search
    if args.step in [1, 3]:
        step1_df = run_step1_grid_search(
            request_counts=args.request_counts,
            gpu_counts=args.gpu_counts,
            bin_counts=args.bin_counts,
            output_file=step1_file
        )
        
        print("\n" + "="*80)
        print("OPTIMAL CONFIGURATIONS")
        print("="*80)
        optimal_configs = analyze_optimal_configs(step1_df)
    
    # Load Step 1 results if only running Step 2
    if args.step == 2 and os.path.exists(step1_file):
        step1_df = pd.read_csv(step1_file)
        optimal_configs = analyze_optimal_configs(step1_df)
    
    # Step 2: Method Comparison
    if args.step in [2, 3]:
        if not optimal_configs:
            # Default optimal configs if Step 1 not run
            optimal_configs = {
                1000: {'num_gpus': 1, 'k_bins': 8},
                10000: {'num_gpus': 4, 'k_bins': 8},
                100000: {'num_gpus': 16, 'k_bins': 8},
                1000000: {'num_gpus': 64, 'k_bins': 8},
            }
        
        step2_df = run_step2_comparison(
            request_counts=args.request_counts,
            optimal_configs=optimal_configs,
            output_file=step2_file
        )
    
    # Generate plots
    if step1_df is not None or step2_df is not None:
        if step1_df is None:
            step1_df = pd.DataFrame()
        if step2_df is None:
            step2_df = pd.DataFrame()
        generate_plots(step1_df, step2_df, plots_dir)
    
    print("\n" + "="*80)
    print("STRESS TEST COMPLETE")
    print("="*80)
    print(f"Results: {args.output_dir}/")
    print(f"  - Step 1: {step1_file}")
    print(f"  - Step 2: {step2_file}")
    print(f"  - Plots:  {plots_dir}/")


if __name__ == '__main__':
    main()
