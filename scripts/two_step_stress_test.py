#!/usr/bin/env python3
"""
Two-Step Stress Testing Suite for LLM Inference Optimization Study

Step 1: Multi-Bin + Dynamic Grid Search
  - Request volumes: 1K, 10K, 100K, 1M
  - GPU counts: 1, 2, 4, 8, 16, 32, 64, 100
  - Bin counts: 1, 2, 4, 8, 16, 32
  Goal: Find optimal GPU + Bin combinations for each workload size

Step 2: Comparison Study (Local vs Cloud)
  - Request volumes: 1K, 10K, 100K, 1M
  - Schedulers:
    1. static_batch (1 GPU, no bin) - Local baseline
    2. dynamic_no_bin (1 GPU) - Local with dynamic batching
    3. multibin_dynamic (1 GPU) - Local with bins + dynamic batching  
    4. multibin_dynamic (optimal GPUs/bins from Step 1) - Cloud optimized
  Goal: Demonstrate impact of cloud inference + optimization techniques

Output:
  - step1_grid_search_results.csv
  - step2_comparison_results.csv
  - Analysis plots in plots/ directory
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.config import SchedulerConfig, compute_equal_mass_boundaries
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics


# ============================================================================
# CONFIGURATION
# ============================================================================

# RPS Scaling - Based on BurstGPT data analysis
# Original dataset: ~0.027 req/s average
# Scale to stress test different GPU counts
RPS_SCALING = 200.0  # 200x scaling -> ~5.4 req/s base rate

# SLA Configuration (per-token TBT)
D_SLA_TOKEN = 0.050  # 50ms per token target

# Dataset and calibration paths
DATASET_PATH = "data/BurstGPT_sample.csv"
CALIBRATION_CSV = "data/qwen3_1_7b_latency_grid.csv"

# Step 1 Grid Search Configuration
STEP1_REQUEST_COUNTS = [1_000, 10_000, 100_000, 1_000_000]
STEP1_GPU_COUNTS = [1, 2, 4, 8, 16, 32, 64, 100]
STEP1_BIN_COUNTS = [1, 2, 4, 8, 16, 32]

# Step 2 Comparison Configuration
STEP2_REQUEST_COUNTS = [1_000, 10_000, 100_000, 1_000_000]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_bin_boundaries_from_data(k_bins: int, num_samples: int = 10000) -> list:
    """Compute equal-mass bin boundaries from BurstGPT data."""
    try:
        df = pd.read_csv(DATASET_PATH, nrows=num_samples * 2)
        df_valid = df[df['Response tokens'] > 0]
        output_lengths = df_valid['Response tokens'].values[:num_samples]
        
        if k_bins == 1:
            return [(1, 10000)]
        
        quantiles = np.linspace(0, 100, k_bins + 1)
        boundaries = [int(np.percentile(output_lengths, q)) for q in quantiles]
        
        # Create bin tuples
        bin_boundaries = []
        for i in range(k_bins):
            min_val = boundaries[i] if i == 0 else boundaries[i]
            max_val = boundaries[i + 1] if i < k_bins - 1 else 10000
            # Ensure min < max
            if min_val >= max_val:
                max_val = min_val + 1
            bin_boundaries.append((min_val, max_val))
        
        return bin_boundaries
    except Exception as e:
        print(f"Warning: Could not compute bin boundaries: {e}")
        # Fallback boundaries
        return [(1, 27), (27, 101), (101, 188), (188, 10000)][:k_bins]


def run_single_simulation(
    num_requests: int,
    num_gpus: int,
    k_bins: int,
    scheduler_type: str,
    rps_scaling: float = RPS_SCALING,
    verbose: bool = True
) -> dict:
    """Run a single simulation and return metrics."""
    
    start_time = time.time()
    
    try:
        # Compute bin boundaries
        if scheduler_type in ["multi_bin_dynamic", "multibin_dynamic"]:
            bin_boundaries = compute_bin_boundaries_from_data(k_bins)
        else:
            bin_boundaries = [(1, 10000)]  # Single bin for non-multibin
            k_bins = 1
        
        # Adjust scheduler type name
        actual_scheduler = "multi_bin_dynamic" if scheduler_type == "multibin_dynamic" else scheduler_type
        
        # Create configuration
        cfg = SchedulerConfig(
            NUM_GPUS=num_gpus,
            K_BINS=k_bins if actual_scheduler == "multi_bin_dynamic" else 1,
            NUM_REQUESTS=num_requests,
            SEED=42,
            D_SLA=D_SLA_TOKEN,
            DATASET_PATH=DATASET_PATH,
            WORKLOAD_SOURCE="burstgpt_dataset",
            USE_REAL_TIMESTAMPS=False,
            RPS_SCALING=rps_scaling,
            USE_EQUAL_MASS_BINS=True,
            USE_REAL_CALIBRATION=True,
            CALIBRATION_CSV_PATH=CALIBRATION_CSV,
            BIN_BOUNDARIES=bin_boundaries if actual_scheduler == "multi_bin_dynamic" else None,
        )
        
        # Generate workload
        requests = generate_workload(cfg)
        
        # Calculate actual RPS
        time_span = max(r.arrival_time for r in requests) - min(r.arrival_time for r in requests)
        actual_rps = len(requests) / time_span if time_span > 0 else 0
        
        # Run simulation
        simulator = Simulator(cfg, requests, actual_scheduler)
        completed_requests = simulator.run()
        
        # Compute metrics
        metrics = compute_metrics(completed_requests, d_sla_token=D_SLA_TOKEN)
        gpu_stats = simulator.get_gpu_stats()
        gpu_metrics = compute_gpu_utilization(gpu_stats)
        batch_stats = compute_batch_statistics(completed_requests)
        
        # Merge metrics
        metrics.update(gpu_metrics)
        metrics.update(batch_stats)
        
        execution_time = time.time() - start_time
        
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'scheduler_type': scheduler_type,
            'num_requests': num_requests,
            'num_gpus': num_gpus,
            'k_bins': k_bins,
            'actual_rps': actual_rps,
            'rps_scaling': rps_scaling,
            'd_sla': D_SLA_TOKEN,
            
            # Key performance metrics
            'sla_violation_rate': metrics.get('sla_violation_rate', 0),
            'capacity_qps_under_sla': metrics.get('capacity_qps_under_sla', 0),
            'throughput_requests_per_sec': metrics.get('throughput_requests_per_sec', 0),
            'throughput_tokens_per_sec': metrics.get('throughput_tokens_per_sec', 0),
            
            # Latency metrics
            'avg_latency': metrics.get('avg_latency', 0),
            'p50_latency': metrics.get('p50_latency', 0),
            'p95_latency': metrics.get('p95_latency', 0),
            'p99_latency': metrics.get('p99_latency', 0),
            'max_latency': metrics.get('max_latency', 0),
            'avg_queueing_delay': metrics.get('avg_queueing_delay', 0),
            'avg_service_time': metrics.get('avg_service_time', 0),
            
            # GPU metrics
            'avg_gpu_utilization': metrics.get('avg_utilization', 0),
            'min_gpu_utilization': metrics.get('min_utilization', 0),
            'max_gpu_utilization': metrics.get('max_utilization', 0),
            
            # Batch metrics
            'num_batches': metrics.get('num_batches', 0),
            'avg_batch_size': metrics.get('avg_batch_size', 0),
            'min_batch_size': metrics.get('min_batch_size', 0),
            'max_batch_size': metrics.get('max_batch_size', 0),
            
            # Summary
            'total_time': metrics.get('total_time', 0),
            'total_tokens': metrics.get('total_tokens', 0),
            'num_completed': len(completed_requests),
            'execution_time_seconds': execution_time,
            'status': 'success'
        }
        
        if verbose:
            print(f"  -> SLA Viol: {result['sla_violation_rate']*100:.1f}%, "
                  f"Throughput: {result['throughput_tokens_per_sec']:.0f} tok/s, "
                  f"P95 Lat: {result['p95_latency']:.3f}s, "
                  f"GPU Util: {result['avg_gpu_utilization']*100:.1f}% "
                  f"[{execution_time:.1f}s]")
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"  -> FAILED: {str(e)[:100]}")
        
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


# ============================================================================
# STEP 1: GRID SEARCH
# ============================================================================

def run_step1_grid_search(args):
    """
    Step 1: Multi-Bin + Dynamic Grid Search
    
    Test all combinations of:
    - Request counts: 1K, 10K, 100K, 1M
    - GPU counts: 1, 2, 4, 8, 16, 32, 64, 100
    - Bin counts: 1, 2, 4, 8, 16, 32
    """
    print("\n" + "="*80)
    print("STEP 1: MULTI-BIN + DYNAMIC GRID SEARCH")
    print("="*80)
    print(f"Request counts: {[f'{n:,}' for n in args.request_counts]}")
    print(f"GPU counts: {args.gpu_counts}")
    print(f"Bin counts: {args.bin_counts}")
    print(f"Total combinations: {len(args.request_counts) * len(args.gpu_counts) * len(args.bin_counts)}")
    print()
    
    results = []
    total_tests = len(args.request_counts) * len(args.gpu_counts) * len(args.bin_counts)
    test_num = 0
    
    for num_requests in args.request_counts:
        print(f"\n{'='*60}")
        print(f"REQUEST COUNT: {num_requests:,}")
        print(f"{'='*60}")
        
        for num_gpus in args.gpu_counts:
            for k_bins in args.bin_counts:
                test_num += 1
                print(f"\n[{test_num}/{total_tests}] GPUs={num_gpus}, K_BINS={k_bins}")
                
                result = run_single_simulation(
                    num_requests=num_requests,
                    num_gpus=num_gpus,
                    k_bins=k_bins,
                    scheduler_type="multi_bin_dynamic",
                    rps_scaling=args.rps_scaling,
                    verbose=True
                )
                results.append(result)
                
                # Intermediate save
                if test_num % 10 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(args.output_step1, index=False)
                    print(f"  [Saved {len(results)} results to {args.output_step1}]")
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(args.output_step1, index=False)
    print(f"\n[STEP 1 COMPLETE] Results saved to {args.output_step1}")
    
    return results


# ============================================================================
# STEP 1 ANALYSIS: FIND OPTIMAL CONFIGURATIONS
# ============================================================================

def analyze_step1_results(results_file: str) -> dict:
    """
    Analyze Step 1 results to find optimal GPU + Bin combinations.
    
    Criteria for "optimal":
    1. Low SLA violation rate (< 10%)
    2. High throughput
    3. Diminishing returns analysis - don't use more GPUs/bins if improvement < 10%
    """
    print("\n" + "="*80)
    print("STEP 1 ANALYSIS: FINDING OPTIMAL CONFIGURATIONS")
    print("="*80)
    
    df = pd.read_csv(results_file)
    df = df[df['status'] == 'success']
    
    optimal_configs = {}
    
    for num_requests in df['num_requests'].unique():
        df_req = df[df['num_requests'] == num_requests].copy()
        
        print(f"\n--- {num_requests:,} Requests ---")
        
        # Sort by throughput, then by GPU count (prefer fewer GPUs at same throughput)
        df_req = df_req.sort_values(
            ['throughput_tokens_per_sec', 'num_gpus', 'k_bins'],
            ascending=[False, True, True]
        )
        
        # Find configurations with acceptable SLA (< 10% violations)
        df_acceptable = df_req[df_req['sla_violation_rate'] < 0.10]
        
        if df_acceptable.empty:
            # If none meet SLA, pick lowest violation rate
            best_row = df_req.iloc[df_req['sla_violation_rate'].idxmin()]
            print(f"  Warning: No config meets SLA < 10%")
        else:
            # Find the "efficient frontier" - best throughput with diminishing returns check
            # Sort by throughput descending
            df_acceptable = df_acceptable.sort_values('throughput_tokens_per_sec', ascending=False)
            best_row = df_acceptable.iloc[0]
            
            # Check for diminishing returns: is there a smaller config within 10% throughput?
            best_throughput = best_row['throughput_tokens_per_sec']
            threshold = best_throughput * 0.90  # 90% of best throughput
            
            df_efficient = df_acceptable[df_acceptable['throughput_tokens_per_sec'] >= threshold]
            df_efficient = df_efficient.sort_values(['num_gpus', 'k_bins'])  # Prefer smaller
            
            if not df_efficient.empty:
                efficient_row = df_efficient.iloc[0]
                # If efficient config uses fewer resources, prefer it
                if (efficient_row['num_gpus'] < best_row['num_gpus'] or 
                    efficient_row['k_bins'] < best_row['k_bins']):
                    best_row = efficient_row
                    print(f"  Selected efficient config (within 10% of max throughput)")
        
        optimal_configs[num_requests] = {
            'num_gpus': int(best_row['num_gpus']),
            'k_bins': int(best_row['k_bins']),
            'throughput_tokens_per_sec': float(best_row['throughput_tokens_per_sec']),
            'sla_violation_rate': float(best_row['sla_violation_rate']),
            'p95_latency': float(best_row['p95_latency']),
            'avg_gpu_utilization': float(best_row['avg_gpu_utilization']),
        }
        
        print(f"  Optimal: GPUs={best_row['num_gpus']}, K_BINS={best_row['k_bins']}")
        print(f"    Throughput: {best_row['throughput_tokens_per_sec']:.0f} tok/s")
        print(f"    SLA Viol: {best_row['sla_violation_rate']*100:.1f}%")
        print(f"    P95 Latency: {best_row['p95_latency']:.3f}s")
        print(f"    GPU Util: {best_row['avg_gpu_utilization']*100:.1f}%")
    
    # Save optimal configs - convert keys to strings for JSON
    optimal_configs_json = {str(k): v for k, v in optimal_configs.items()}
    with open(results_file.replace('.csv', '_optimal_configs.json'), 'w') as f:
        json.dump(optimal_configs_json, f, indent=2)
    
    print(f"\nOptimal configs saved to {results_file.replace('.csv', '_optimal_configs.json')}")
    
    return optimal_configs


# ============================================================================
# STEP 2: COMPARISON STUDY
# ============================================================================

def run_step2_comparison(args, optimal_configs: dict = None):
    """
    Step 2: Comparison Study (Local vs Cloud)
    
    For each request count, compare:
    1. static_fifo (1 GPU, no bins) - Local baseline
    2. dynamic_no_bins (1 GPU) - Local with dynamic batching
    3. multibin_dynamic (1 GPU) - Local with bins + dynamic
    4. multibin_dynamic (optimal config) - Cloud optimized
    """
    print("\n" + "="*80)
    print("STEP 2: COMPARISON STUDY (LOCAL vs CLOUD)")
    print("="*80)
    
    # Load optimal configs if not provided
    if optimal_configs is None:
        config_file = args.output_step1.replace('.csv', '_optimal_configs.json')
        if Path(config_file).exists():
            with open(config_file) as f:
                optimal_configs = json.load(f)
                # Convert string keys to int
                optimal_configs = {int(k): v for k, v in optimal_configs.items()}
            print(f"Loaded optimal configs from {config_file}")
        else:
            print("Warning: No optimal configs found, using defaults")
            optimal_configs = {}
    
    results = []
    
    for num_requests in args.request_counts:
        print(f"\n{'='*60}")
        print(f"REQUEST COUNT: {num_requests:,}")
        print(f"{'='*60}")
        
        # Get optimal config for this request count
        opt = optimal_configs.get(num_requests, {'num_gpus': 4, 'k_bins': 4})
        
        test_configs = [
            ("static_fifo", 1, 1, "Local - Static Batch"),
            ("dynamic_no_bins", 1, 1, "Local - Dynamic Batching"),
            ("multi_bin_dynamic", 1, 4, "Local - MultiBin + Dynamic"),
            ("multi_bin_dynamic", opt['num_gpus'], opt['k_bins'], 
             f"Cloud - Optimized ({opt['num_gpus']} GPUs, {opt['k_bins']} bins)"),
        ]
        
        for scheduler_type, num_gpus, k_bins, description in test_configs:
            print(f"\n{description}:")
            print(f"  Scheduler: {scheduler_type}, GPUs: {num_gpus}, K_BINS: {k_bins}")
            
            result = run_single_simulation(
                num_requests=num_requests,
                num_gpus=num_gpus,
                k_bins=k_bins,
                scheduler_type=scheduler_type,
                rps_scaling=args.rps_scaling,
                verbose=True
            )
            result['description'] = description
            result['comparison_group'] = 'local' if num_gpus == 1 else 'cloud'
            results.append(result)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output_step2, index=False)
    print(f"\n[STEP 2 COMPLETE] Results saved to {args.output_step2}")
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def generate_plots(args):
    """Generate analysis plots from results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create plots directory
    plots_dir = Path("plots/stress_test")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Step 1 Plots =====
    if Path(args.output_step1).exists():
        print("\nGenerating Step 1 plots...")
        df1 = pd.read_csv(args.output_step1)
        df1 = df1[df1['status'] == 'success']
        
        # Plot 1: Heatmap of throughput by GPU and Bin count for each request count
        for num_requests in df1['num_requests'].unique():
            df_req = df1[df1['num_requests'] == num_requests]
            
            # Create pivot table
            pivot = df_req.pivot_table(
                values='throughput_tokens_per_sec',
                index='num_gpus',
                columns='k_bins',
                aggfunc='mean'
            )
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
            
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel('Number of Bins')
            ax.set_ylabel('Number of GPUs')
            ax.set_title(f'Throughput (tokens/sec) - {num_requests:,} Requests')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Throughput (tokens/sec)')
            
            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'step1_heatmap_{num_requests}.png', dpi=150)
            plt.close()
        
        # Plot 2: GPU Scaling curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, num_requests in enumerate(sorted(df1['num_requests'].unique())):
            if idx >= 4:
                break
            ax = axes[idx]
            df_req = df1[df1['num_requests'] == num_requests]
            
            for k_bins in sorted(df_req['k_bins'].unique()):
                df_kb = df_req[df_req['k_bins'] == k_bins]
                df_kb = df_kb.sort_values('num_gpus')
                ax.plot(df_kb['num_gpus'], df_kb['throughput_tokens_per_sec'], 
                       'o-', label=f'{k_bins} bins')
            
            ax.set_xlabel('Number of GPUs')
            ax.set_ylabel('Throughput (tokens/sec)')
            ax.set_title(f'{num_requests:,} Requests')
            ax.legend(title='Bins', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
        
        plt.suptitle('GPU Scaling Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_dir / 'step1_gpu_scaling.png', dpi=150)
        plt.close()
        
        # Plot 3: Bin effect at fixed GPU counts
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, num_gpus in enumerate([1, 4, 16, 64]):
            if idx >= 4:
                break
            ax = axes[idx]
            df_gpu = df1[df1['num_gpus'] == num_gpus]
            
            for num_requests in sorted(df_gpu['num_requests'].unique()):
                df_nr = df_gpu[df_gpu['num_requests'] == num_requests]
                df_nr = df_nr.sort_values('k_bins')
                ax.plot(df_nr['k_bins'], df_nr['throughput_tokens_per_sec'], 
                       'o-', label=f'{num_requests:,}')
            
            ax.set_xlabel('Number of Bins')
            ax.set_ylabel('Throughput (tokens/sec)')
            ax.set_title(f'{num_gpus} GPUs')
            ax.legend(title='Requests', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
        
        plt.suptitle('Bin Count Effect Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_dir / 'step1_bin_effect.png', dpi=150)
        plt.close()
        
        print(f"  Saved: step1_heatmap_*.png, step1_gpu_scaling.png, step1_bin_effect.png")
    
    # ===== Step 2 Plots =====
    if Path(args.output_step2).exists():
        print("\nGenerating Step 2 plots...")
        df2 = pd.read_csv(args.output_step2)
        df2 = df2[df2['status'] == 'success']
        
        # Plot 4: Comparison bar chart - Throughput
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, num_requests in enumerate(sorted(df2['num_requests'].unique())):
            if idx >= 4:
                break
            ax = axes[idx]
            df_req = df2[df2['num_requests'] == num_requests]
            
            descriptions = df_req['description'].values
            throughputs = df_req['throughput_tokens_per_sec'].values
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
            
            bars = ax.bar(range(len(descriptions)), throughputs, color=colors[:len(descriptions)])
            ax.set_xticks(range(len(descriptions)))
            ax.set_xticklabels([d.split(' - ')[1] if ' - ' in d else d for d in descriptions], 
                              rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('Throughput (tokens/sec)')
            ax.set_title(f'{num_requests:,} Requests')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, throughputs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Throughput Comparison: Local vs Cloud', fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_dir / 'step2_throughput_comparison.png', dpi=150)
        plt.close()
        
        # Plot 5: Comparison - Latency
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, num_requests in enumerate(sorted(df2['num_requests'].unique())):
            if idx >= 4:
                break
            ax = axes[idx]
            df_req = df2[df2['num_requests'] == num_requests]
            
            descriptions = df_req['description'].values
            p95_latencies = df_req['p95_latency'].values
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
            
            bars = ax.bar(range(len(descriptions)), p95_latencies, color=colors[:len(descriptions)])
            ax.set_xticks(range(len(descriptions)))
            ax.set_xticklabels([d.split(' - ')[1] if ' - ' in d else d for d in descriptions], 
                              rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('P95 Latency (seconds)')
            ax.set_title(f'{num_requests:,} Requests')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, p95_latencies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}s', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('P95 Latency Comparison: Local vs Cloud', fontsize=14)
        plt.tight_layout()
        plt.savefig(plots_dir / 'step2_latency_comparison.png', dpi=150)
        plt.close()
        
        # Plot 6: Speedup relative to baseline
        fig, ax = plt.subplots(figsize=(12, 6))
        
        request_counts = sorted(df2['num_requests'].unique())
        width = 0.2
        x = np.arange(len(request_counts))
        
        for i, (idx, row) in enumerate(df2.groupby('num_requests')):
            if i == 0:
                labels = [d.split(' - ')[1] if ' - ' in d else d for d in row['description'].values]
        
        scheduler_throughputs = {}
        for desc in df2['description'].unique():
            throughputs = []
            for nr in request_counts:
                df_match = df2[(df2['num_requests'] == nr) & (df2['description'] == desc)]
                if not df_match.empty:
                    throughputs.append(df_match['throughput_tokens_per_sec'].values[0])
                else:
                    throughputs.append(0)
            scheduler_throughputs[desc] = throughputs
        
        # Calculate speedup relative to static_fifo baseline
        baseline_desc = [d for d in scheduler_throughputs.keys() if 'Static' in d][0]
        baseline = scheduler_throughputs[baseline_desc]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        for i, (desc, throughputs) in enumerate(scheduler_throughputs.items()):
            speedups = [t/b if b > 0 else 0 for t, b in zip(throughputs, baseline)]
            label = desc.split(' - ')[1] if ' - ' in desc else desc
            ax.bar(x + i*width, speedups, width, label=label, color=colors[i % len(colors)])
        
        ax.set_xlabel('Number of Requests')
        ax.set_ylabel('Speedup vs Local Static Batch (×)')
        ax.set_title('Performance Improvement: Local vs Cloud Inference')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'{nr:,}' for nr in request_counts])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'step2_speedup.png', dpi=150)
        plt.close()
        
        print(f"  Saved: step2_throughput_comparison.png, step2_latency_comparison.png, step2_speedup.png")
    
    print(f"\nAll plots saved to {plots_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Two-Step Stress Testing for LLM Inference Optimization"
    )
    
    # Step selection
    parser.add_argument('--step', type=str, default='all', choices=['1', '2', 'all', 'analyze', 'plot'],
                       help='Which step to run: 1, 2, all, analyze, or plot')
    
    # Configuration
    parser.add_argument('--request-counts', type=int, nargs='+', 
                       default=STEP1_REQUEST_COUNTS,
                       help='Request counts to test')
    parser.add_argument('--gpu-counts', type=int, nargs='+',
                       default=STEP1_GPU_COUNTS,
                       help='GPU counts for Step 1')
    parser.add_argument('--bin-counts', type=int, nargs='+',
                       default=STEP1_BIN_COUNTS,
                       help='Bin counts for Step 1')
    parser.add_argument('--rps-scaling', type=float, default=RPS_SCALING,
                       help='RPS scaling factor')
    
    # Output files
    parser.add_argument('--output-step1', type=str, 
                       default='step1_grid_search_results.csv',
                       help='Output CSV for Step 1')
    parser.add_argument('--output-step2', type=str,
                       default='step2_comparison_results.csv',
                       help='Output CSV for Step 2')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TWO-STEP STRESS TESTING FOR LLM INFERENCE OPTIMIZATION")
    print("="*80)
    print(f"RPS Scaling: {args.rps_scaling}x")
    print(f"SLA Target: {D_SLA_TOKEN*1000:.0f}ms per token")
    print()
    
    start_time = time.time()
    
    if args.step in ['1', 'all']:
        run_step1_grid_search(args)
    
    optimal_configs = None
    if args.step in ['analyze', 'all', '2']:
        if Path(args.output_step1).exists():
            optimal_configs = analyze_step1_results(args.output_step1)
    
    if args.step in ['2', 'all']:
        run_step2_comparison(args, optimal_configs)
    
    if args.step in ['plot', 'all']:
        generate_plots(args)
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETE! Total execution time: {total_time/60:.1f} minutes")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
