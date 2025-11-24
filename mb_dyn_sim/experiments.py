"""
Experiment runners and comparisons.
"""

import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import os

from .config import SchedulerConfig
from .workload import generate_workload
from .simulation import Simulator
from .metrics import (
    compute_metrics, 
    compute_gpu_utilization, 
    compute_comparative_metrics,
    compute_batch_statistics,
    estimate_memory_usage,
)


def run_experiment(
    cfg: SchedulerConfig,
    scheduler_type: str,
    load_level: str = "medium",
    seed: int | None = None,
) -> Dict:
    """
    Run a single experiment with specified configuration.
    
    Args:
        cfg: Scheduler configuration
        scheduler_type: Type of scheduler ("static_fifo", "dynamic_no_bins", "multi_bin_dynamic")
        load_level: Load level ("low", "medium", "high")
        seed: Random seed (uses cfg.SEED if None)
    
    Returns:
        Dictionary containing metrics and configuration
    """
    # Adjust configuration based on load level
    if seed is not None:
        cfg.SEED = seed
    
    # Generate workload
    print(f"Generating {load_level} load workload...")
    requests = generate_workload(cfg)
    
    # Run simulation
    print(f"Running {scheduler_type} scheduler...")
    sim = Simulator(cfg, requests, scheduler_type=scheduler_type)
    completed = sim.run()
    
    # Compute metrics
    metrics = compute_metrics(completed)
    gpu_stats = sim.get_gpu_stats()
    gpu_metrics = compute_gpu_utilization(gpu_stats)
    batch_stats = compute_batch_statistics(completed)
    
    # Estimate memory usage based on average batch
    if batch_stats['avg_batch_size'] > 0:
        avg_seq_len = metrics['total_tokens'] / metrics['num_requests'] if metrics['num_requests'] > 0 else 0
        memory_stats = estimate_memory_usage(
            batch_size=int(batch_stats['avg_batch_size']),
            avg_sequence_length=int(avg_seq_len),
            kv_cache_per_token_gb=cfg.KV_MEM_PER_TOKEN_GB,
            model_size_gb=cfg.M_MODEL_GB,
        )
    else:
        memory_stats = {}
    
    return {
        'scheduler_type': scheduler_type,
        'load_level': load_level,
        'num_gpus': cfg.NUM_GPUS,
        'k_bins': cfg.K_BINS,
        'metrics': metrics,
        'gpu_metrics': gpu_metrics,
        'batch_stats': batch_stats,
        'memory_stats': memory_stats,
        'completed_requests': completed,
    }


def compare_schedulers(
    cfg: SchedulerConfig,
    scheduler_types: List[str],
    load_level: str = "medium",
) -> pd.DataFrame:
    """
    Compare multiple schedulers and return results as DataFrame.
    
    Args:
        cfg: Scheduler configuration
        scheduler_types: List of scheduler types to compare
        load_level: Load level for the comparison
    
    Returns:
        DataFrame with comparison results including paper-specific metrics
    """
    results = []
    baseline_metrics = None
    
    for scheduler_type in scheduler_types:
        print(f"\nRunning experiment: {scheduler_type}")
        result = run_experiment(cfg, scheduler_type, load_level)
        
        # Extract key metrics
        m = result['metrics']
        gm = result['gpu_metrics']
        bs = result['batch_stats']
        
        # Store baseline for comparisons
        if scheduler_type == 'static_fifo':
            baseline_metrics = m
        
        # Compute improvements if we have baseline
        improvements = {}
        if baseline_metrics is not None and scheduler_type != 'static_fifo':
            improvements = compute_comparative_metrics(m, baseline_metrics)
        
        result_row = {
            'Scheduler': scheduler_type,
            
            # Core metrics
            'Throughput (req/s)': m['throughput_requests_per_sec'],
            'Throughput (tok/s)': m['throughput_tokens_per_sec'],
            'Avg Latency (s)': m['avg_latency'],
            'P95 Latency (s)': m['p95_latency'],
            'P99 Latency (s)': m['p99_latency'],
            'SLA Violation (%)': m['sla_violation_rate'] * 100,
            'GPU Utilization (%)': gm['avg_utilization'] * 100,
            
            # Multi-Bin Paper metrics
            'Capacity Lambda (req/s)': m['capacity_threshold_lambda'],
            'Sec/Gen Token': m['seconds_per_generated_token'],
            
            # Dynamic Batching Paper metrics
            'Decode Step (ms)': m['decode_step_time_ms'],
            'Capacity QPS': m['capacity_qps_under_sla'],
            
            # Batch statistics
            'Avg Batch Size': bs['avg_batch_size'],
            'Max Batch Size': bs['max_batch_size'],
            
            # Configuration
            'Num GPUs': cfg.NUM_GPUS,
            'K Bins': cfg.K_BINS if scheduler_type == 'multi_bin_dynamic' else 1,
        }
        
        # Add improvement metrics if available
        if improvements:
            result_row['Throughput Δ (%)'] = improvements.get('throughput_improvement_vs_baseline_percent', 0)
            result_row['Capacity Δ (%)'] = improvements.get('capacity_improvement_percent_vs_static', 0)
            result_row['Latency Δ (%)'] = improvements.get('latency_improvement_percent', 0)
            result_row['SLA Reduction (%)'] = improvements.get('sla_violation_reduction_percent', 0)
        else:
            result_row['Throughput Δ (%)'] = 0.0
            result_row['Capacity Δ (%)'] = 0.0
            result_row['Latency Δ (%)'] = 0.0
            result_row['SLA Reduction (%)'] = 0.0
        
        results.append(result_row)
    
    df = pd.DataFrame(results)
    
    # Sanity check: warn if all schedulers give identical results
    if len(df) > 1:
        metrics_to_check = ['Avg Latency (s)', 'P95 Latency (s)', 'SLA Violation (%)']
        all_identical = True
        for metric in metrics_to_check:
            if df[metric].std() > 0.001:  # Allow small numerical differences
                all_identical = False
                break
        
        if all_identical:
            print("\n" + "="*70)
            print("WARNING: All schedulers produced IDENTICAL metrics!")
            print("This suggests:")
            print("  1. Load is too low (GPU not saturated)")
            print("  2. All requests have similar lengths (binning has no effect)")
            print("  3. Scheduler implementations may not be truly distinct")
            print("="*70 + "\n")
    
    # Print comprehensive comparison table
    print("\n" + "="*120)
    print("COMPREHENSIVE SCHEDULER COMPARISON - PAPER-ALIGNED METRICS")
    print("="*120)
    
    # Core performance
    print("\n[CORE PERFORMANCE METRICS]")
    print(df[['Scheduler', 'Throughput (req/s)', 'Throughput (tok/s)', 
              'Avg Latency (s)', 'P95 Latency (s)', 'SLA Violation (%)']].to_string(index=False))
    
    # Paper-specific metrics
    print("\n[MULTI-BIN BATCHING PAPER METRICS]")
    print(df[['Scheduler', 'Capacity Lambda (req/s)', 'Sec/Gen Token', 
              'Avg Batch Size']].to_string(index=False))
    
    print("\n[DYNAMIC BATCHING PAPER METRICS]")
    print(df[['Scheduler', 'Decode Step (ms)', 'Capacity QPS', 
              'GPU Utilization (%)']].to_string(index=False))
    
    # Improvement metrics
    print("\n[IMPROVEMENT VS BASELINE - static_fifo]")
    improvement_df = df[df['Scheduler'] != 'static_fifo'][
        ['Scheduler', 'Throughput Δ (%)', 'Capacity Δ (%)', 
         'Latency Δ (%)', 'SLA Reduction (%)']
    ]
    if not improvement_df.empty:
        print(improvement_df.to_string(index=False))
    else:
        print("  (No comparison - baseline only)")
    
    print("\n" + "="*120 + "\n")
    
    return df


def plot_comparison(
    df: pd.DataFrame,
    output_dir: str = "plots",
) -> None:
    """
    Create comparison plots from experiment results.
    
    Args:
        df: DataFrame with experiment results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Scheduler Comparison', fontsize=16, fontweight='bold')
    
    schedulers = df['Scheduler'].tolist()
    
    # Plot 1: Throughput
    ax = axes[0, 0]
    x_pos = range(len(schedulers))
    ax.bar(x_pos, df['Throughput (req/s)'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Scheduler')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('Request Throughput')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(schedulers, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Latency
    ax = axes[0, 1]
    width = 0.35
    x_pos = range(len(schedulers))
    ax.bar([x - width/2 for x in x_pos], df['Avg Latency (s)'], 
           width, label='Average', color='coral', alpha=0.8)
    ax.bar([x + width/2 for x in x_pos], df['P95 Latency (s)'], 
           width, label='P95', color='tomato', alpha=0.8)
    ax.set_xlabel('Scheduler')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Latency Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(schedulers, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: SLA Violation Rate
    ax = axes[1, 0]
    ax.bar(x_pos, df['SLA Violation (%)'], color='indianred', alpha=0.8)
    ax.set_xlabel('Scheduler')
    ax.set_ylabel('SLA Violation (%)')
    ax.set_title('SLA Violation Rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(schedulers, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: GPU Utilization
    ax = axes[1, 1]
    ax.bar(x_pos, df['GPU Utilization (%)'], color='mediumseagreen', alpha=0.8)
    ax.set_xlabel('Scheduler')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU Utilization')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(schedulers, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'scheduler_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.close()


def plot_k_bins_sensitivity(
    cfg: SchedulerConfig,
    k_bins_values: List[int] = [1, 2, 4, 8],
    output_dir: str = "plots",
) -> None:
    """
    Plot sensitivity to number of bins (K_BINS).
    
    Args:
        cfg: Base scheduler configuration
        k_bins_values: List of K_BINS values to test
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for k_bins in k_bins_values:
        print(f"\nTesting K_BINS = {k_bins}")
        
        # Adjust bin boundaries for different K values
        if k_bins == 1:
            boundaries = [(0, 10000)]
        elif k_bins == 2:
            boundaries = [(0, 256), (256, 10000)]
        elif k_bins == 8:
            boundaries = [
                (0, 32), (32, 64), (64, 128), (128, 256),
                (256, 512), (512, 1024), (1024, 2048), (2048, 10000)
            ]
        else:
            # Default 4 bins
            boundaries = [(0, 64), (64, 256), (256, 1024), (1024, 10000)]
        
        # Update config
        test_cfg = SchedulerConfig(
            NUM_GPUS=cfg.NUM_GPUS,
            K_BINS=k_bins,
            BIN_BOUNDARIES=boundaries,
            NUM_REQUESTS=cfg.NUM_REQUESTS,
            SEED=cfg.SEED,
            D_SLA=cfg.D_SLA,
        )
        
        result = run_experiment(test_cfg, "multi_bin_dynamic", "medium")
        m = result['metrics']
        
        results.append({
            'K_BINS': k_bins,
            'Throughput': m['throughput_requests_per_sec'],
            'P95 Latency': m['p95_latency'],
            'SLA Violation': m['sla_violation_rate'] * 100,
        })
    
    df = pd.DataFrame(results)
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Multi-Bin Sensitivity (K_BINS)', fontsize=14, fontweight='bold')
    
    # Throughput vs K_BINS
    ax = axes[0]
    ax.plot(df['K_BINS'], df['Throughput'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Bins (K)')
    ax.set_ylabel('Throughput (req/s)')
    ax.set_title('Throughput vs K_BINS')
    ax.grid(alpha=0.3)
    
    # P95 Latency vs K_BINS
    ax = axes[1]
    ax.plot(df['K_BINS'], df['P95 Latency'], marker='s', linewidth=2, 
            markersize=8, color='coral')
    ax.set_xlabel('Number of Bins (K)')
    ax.set_ylabel('P95 Latency (s)')
    ax.set_title('P95 Latency vs K_BINS')
    ax.grid(alpha=0.3)
    
    # SLA Violation vs K_BINS
    ax = axes[2]
    ax.plot(df['K_BINS'], df['SLA Violation'], marker='^', linewidth=2, 
            markersize=8, color='indianred')
    ax.set_xlabel('Number of Bins (K)')
    ax.set_ylabel('SLA Violation (%)')
    ax.set_title('SLA Violation vs K_BINS')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'k_bins_sensitivity.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.close()
