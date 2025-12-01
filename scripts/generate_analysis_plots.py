"""
Generate analysis plots for the paper figures.

This script creates:
1. Throughput vs GPU count plots (per scheduler)
2. Request SLA violation vs GPU count (Pareto frontier)
3. P95 latency vs throughput for different K bins
4. Statistical variance analysis with multiple seeds
5. Sensitivity analysis for EMA and step-size parameters
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import load_burstgpt_dataset
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics


def load_all_results():
    """Load results from all three load levels."""
    base_dir = Path(__file__).parent.parent
    
    results = {}
    for load_name, load_dir in [
        ("high_100x", "stress_test_final"),
        ("medium_10x", "stress_test_low_load"),
        ("low_1x", "stress_test_ultra_low_load"),
    ]:
        grid_path = base_dir / load_dir / "step1_grid_search.csv"
        comp_path = base_dir / load_dir / "step2_comparison.csv"
        
        if grid_path.exists():
            results[f"{load_name}_grid"] = pd.read_csv(grid_path)
        if comp_path.exists():
            results[f"{load_name}_comp"] = pd.read_csv(comp_path)
    
    return results


def plot_throughput_vs_gpu(results: dict, output_dir: Path):
    """
    Figure 1: Throughput vs GPU count for representative workloads.
    Shows scaling behavior for each scheduler type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    load_labels = {
        "high_100x": "High Load (100× RPS)",
        "medium_10x": "Medium Load (10× RPS)",
        "low_1x": "Low Load (1× RPS)",
    }
    
    # Use 100K requests as representative workload
    workload_size = 100000
    
    for idx, (load_key, load_label) in enumerate(load_labels.items()):
        ax = axes[idx]
        df = results.get(f"{load_key}_grid")
        
        if df is None:
            continue
        
        # Filter for the representative workload
        df_wl = df[df['num_requests'] == workload_size]
        
        # Plot for different K values (using K=1, 4, 8 as representative)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))
        k_values = [1, 4, 8, 16]
        
        for k_idx, k in enumerate(k_values):
            df_k = df_wl[df_wl['k_bins'] == k]
            if len(df_k) > 0:
                df_k = df_k.sort_values('num_gpus')
                ax.plot(df_k['num_gpus'], df_k['throughput_tok_s'], 
                       marker='o', label=f'K={k}', color=colors[k_idx],
                       linewidth=2, markersize=6)
        
        ax.set_xlabel('Number of GPUs', fontsize=12)
        ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_title(f'{load_label}\n(100K requests)', fontsize=12)
        ax.legend(title='Bin Count', loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 100])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_throughput_vs_gpu.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig1_throughput_vs_gpu.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig1_throughput_vs_gpu.png/pdf")


def plot_sla_violation_pareto(results: dict, output_dir: Path):
    """
    Figure 2: Request SLA violation rate vs GPU count.
    Shows the Pareto frontier between resources and SLA compliance.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    load_labels = {
        "high_100x": "High Load (100× RPS)",
        "medium_10x": "Medium Load (10× RPS)", 
        "low_1x": "Low Load (1× RPS)",
    }
    
    workload_sizes = [10000, 100000, 1000000]
    markers = ['o', 's', '^']
    
    for idx, (load_key, load_label) in enumerate(load_labels.items()):
        ax = axes[idx]
        df = results.get(f"{load_key}_grid")
        
        if df is None:
            continue
        
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(workload_sizes)))
        
        for wl_idx, wl_size in enumerate(workload_sizes):
            # Get best K for each GPU count (Pareto optimal)
            df_wl = df[df['num_requests'] == wl_size]
            
            # Group by GPU count and find minimum SLA violation
            pareto_points = df_wl.groupby('num_gpus').agg({
                'request_sla_pct': 'min',
                'throughput_tok_s': 'max'
            }).reset_index()
            
            pareto_points = pareto_points.sort_values('num_gpus')
            
            label = f'{wl_size//1000}K' if wl_size < 1000000 else '1M'
            ax.plot(pareto_points['num_gpus'], pareto_points['request_sla_pct'],
                   marker=markers[wl_idx], label=label, color=colors[wl_idx],
                   linewidth=2, markersize=8)
        
        # Add SLA threshold lines
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.5, label='5% target')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% target')
        
        ax.set_xlabel('Number of GPUs', fontsize=12)
        ax.set_ylabel('Request SLA Violation (%)', fontsize=12)
        ax.set_title(f'{load_label}', fontsize=12)
        ax.legend(title='Workload', loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 100])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_sla_pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig2_sla_pareto_frontier.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig2_sla_pareto_frontier.png/pdf")


def plot_latency_throughput_tradeoff(results: dict, output_dir: Path):
    """
    Figure 3: P95 latency vs throughput for different K bins.
    Shows the bin count tradeoff visually.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    load_labels = {
        "high_100x": "High Load (100× RPS)",
        "medium_10x": "Medium Load (10× RPS)",
        "low_1x": "Low Load (1× RPS)",
    }
    
    # Use 100K requests as representative
    workload_size = 100000
    k_values = [1, 2, 4, 8, 16, 32]
    
    for idx, (load_key, load_label) in enumerate(load_labels.items()):
        ax = axes[idx]
        df = results.get(f"{load_key}_grid")
        
        if df is None:
            continue
        
        df_wl = df[df['num_requests'] == workload_size]
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(k_values)))
        
        for k_idx, k in enumerate(k_values):
            df_k = df_wl[df_wl['k_bins'] == k]
            if len(df_k) > 0:
                ax.scatter(df_k['throughput_tok_s'], df_k['p95_latency_s'],
                          s=100, label=f'K={k}', color=colors[k_idx],
                          alpha=0.7, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_ylabel('P95 Latency (seconds)', fontsize=12)
        ax.set_title(f'{load_label}\n(100K requests)', fontsize=12)
        ax.legend(title='Bin Count', loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_latency_throughput_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_latency_throughput_tradeoff.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig3_latency_throughput_tradeoff.png/pdf")


def plot_scheduler_comparison_bar(results: dict, output_dir: Path):
    """
    Figure 4: Bar chart comparing schedulers across load levels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    load_configs = [
        ("high_100x", "High Load (100×)"),
        ("medium_10x", "Medium Load (10×)"),
        ("low_1x", "Low Load (1×)"),
    ]
    
    # Use 100K workload
    workload_size = 100000
    
    # Scheduler mapping from step2 comparison files
    scheduler_map = {
        'static_fifo': 'Static FIFO',
        'dynamic_no_bins': 'Dynamic No-Bins', 
        'multi_bin_dynamic': 'Multi-Bin (1 GPU)',
        'multi_bin_optimal': 'Multi-Bin (Optimal)'
    }
    
    x = np.arange(len(load_configs))
    width = 0.2
    
    # Plot 1: SLA Violations
    ax = axes[0]
    for load_idx, (load_key, load_label) in enumerate(load_configs):
        df = results.get(f"{load_key}_comp")
        if df is None:
            continue
        df_wl = df[df['num_requests'] == workload_size]
        
        # Get values for each method
        values = []
        for method_idx, method in enumerate(['1. Static FIFO', '2. Dynamic No-Bins', '3. Multi-Bin Dynamic (1 GPU)', '4. Multi-Bin Dynamic (Optimal']):
            df_m = df_wl[df_wl['method'].str.contains(method.split('(')[0].strip(), case=False, na=False)]
            if len(df_m) > 0:
                values.append(df_m['request_sla_pct'].iloc[0])
            else:
                values.append(0)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (val, color) in enumerate(zip(values, colors)):
            ax.bar(load_idx + i*width, val, width, color=color,
                  label=list(scheduler_map.values())[i] if load_idx == 0 else '')
    
    ax.set_ylabel('Request SLA Violation (%)', fontsize=12)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([l[1] for l in load_configs])
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('100K Requests - SLA Comparison', fontsize=12)

    # Plot 2: Throughput comparison
    ax = axes[1]
    for load_idx, (load_key, load_label) in enumerate(load_configs):
        df = results.get(f"{load_key}_comp")
        if df is None:
            continue
        df_wl = df[df['num_requests'] == workload_size]
        
        values = []
        for method in ['1. Static FIFO', '2. Dynamic No-Bins', '3. Multi-Bin Dynamic (1 GPU)', '4. Multi-Bin Dynamic (Optimal']:
            df_m = df_wl[df_wl['method'].str.contains(method.split('(')[0].strip(), case=False, na=False)]
            if len(df_m) > 0:
                values.append(df_m['throughput_tok_s'].iloc[0])
            else:
                values.append(0)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (val, color) in enumerate(zip(values, colors)):
            ax.bar(load_idx + i*width, val, width, color=color)
    
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([l[1] for l in load_configs])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('100K Requests - Throughput Comparison', fontsize=12)
    
    # Plot 3: P95 Latency comparison (log scale)
    ax = axes[2]
    for load_idx, (load_key, load_label) in enumerate(load_configs):
        df = results.get(f"{load_key}_comp")
        if df is None:
            continue
        df_wl = df[df['num_requests'] == workload_size]
        
        values = []
        for method in ['1. Static FIFO', '2. Dynamic No-Bins', '3. Multi-Bin Dynamic (1 GPU)', '4. Multi-Bin Dynamic (Optimal']:
            df_m = df_wl[df_wl['method'].str.contains(method.split('(')[0].strip(), case=False, na=False)]
            if len(df_m) > 0:
                values.append(df_m['p95_latency_s'].iloc[0])
            else:
                values.append(0.1)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (val, color) in enumerate(zip(values, colors)):
            ax.bar(load_idx + i*width, val, width, color=color)
    
    ax.set_ylabel('P95 Latency (seconds, log scale)', fontsize=12)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([l[1] for l in load_configs])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('100K Requests - Latency Comparison', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_scheduler_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_scheduler_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig4_scheduler_comparison.png/pdf")


def run_sensitivity_analysis(output_dir: Path):
    """
    Run sensitivity analysis for EMA and step-size parameters.
    Tests robustness of controller settings.
    """
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Load a subset of requests for quick analysis
    base_dir = Path(__file__).parent.parent
    
    # Use default config and modify as needed
    base_cfg = SchedulerConfig(
        NUM_GPUS=4,
        K_BINS=4,
        D_SLA_TOKEN=0.010,
        D_SLA_REQUEST=20.0,
        NUM_REQUESTS=10000,
        DATASET_PATH=str(base_dir / "data" / "BurstGPT_sample.csv"),
        USE_REAL_TIMESTAMPS=False,
        RPS_SCALING=10.0,  # Medium load
    )
    
    requests = load_burstgpt_dataset(
        str(base_dir / "data" / "BurstGPT_sample.csv"),
        num_requests=10000,
        d_sla_token=0.010,
        d_sla_request=20.0,
        use_real_timestamps=False,
        rps_scaling=10.0  # Medium load
    )
    
    results_sensitivity = []
    
    # Test different K_BINS values (main sensitivity parameter)
    print("\n--- Testing K_BINS variations ---")
    k_values = [1, 2, 4, 8, 16]
    for k in k_values:
        cfg = SchedulerConfig(
            NUM_GPUS=4,
            K_BINS=k,
            D_SLA_TOKEN=0.010,
            D_SLA_REQUEST=20.0,
            NUM_REQUESTS=10000,
            DATASET_PATH=str(base_dir / "data" / "BurstGPT_sample.csv"),
            USE_REAL_TIMESTAMPS=False,
            RPS_SCALING=10.0,
        )
        
        sim = Simulator(cfg, requests.copy(), scheduler_type="multi_bin_dynamic")
        completed = sim.run()
        metrics = compute_metrics(completed, d_sla_token=0.010, d_sla_request=20.0)
        
        results_sensitivity.append({
            'parameter': 'K_BINS',
            'value': k,
            'request_sla_pct': metrics['sla_violation_rate_request'] * 100,
            'throughput_tok_s': metrics['throughput_tokens_per_sec'],
            'p95_latency_s': metrics['p95_latency'],
            'avg_batch_size': metrics.get('avg_batch_size', 0),
        })
        print(f"  K={k}: SLA={metrics['sla_violation_rate_request']*100:.2f}%, "
              f"Throughput={metrics['throughput_tokens_per_sec']:.1f} tok/s")
    
    # Test different GPU counts (resource sensitivity)
    print("\n--- Testing GPU count variations ---")
    gpu_values = [1, 2, 4, 8, 16]
    for g in gpu_values:
        cfg = SchedulerConfig(
            NUM_GPUS=g,
            K_BINS=4,
            D_SLA_TOKEN=0.010,
            D_SLA_REQUEST=20.0,
            NUM_REQUESTS=10000,
            DATASET_PATH=str(base_dir / "data" / "BurstGPT_sample.csv"),
            USE_REAL_TIMESTAMPS=False,
            RPS_SCALING=10.0,
        )
        
        sim = Simulator(cfg, requests.copy(), scheduler_type="multi_bin_dynamic")
        completed = sim.run()
        metrics = compute_metrics(completed, d_sla_token=0.010, d_sla_request=20.0)
        
        results_sensitivity.append({
            'parameter': 'NUM_GPUS',
            'value': g,
            'request_sla_pct': metrics['sla_violation_rate_request'] * 100,
            'throughput_tok_s': metrics['throughput_tokens_per_sec'],
            'p95_latency_s': metrics['p95_latency'],
            'avg_batch_size': metrics.get('avg_batch_size', 0),
        })
        print(f"  GPUs={g}: SLA={metrics['sla_violation_rate_request']*100:.2f}%, "
              f"Throughput={metrics['throughput_tokens_per_sec']:.1f} tok/s")
    
    # Save results
    df_sensitivity = pd.DataFrame(results_sensitivity)
    df_sensitivity.to_csv(output_dir / 'sensitivity_analysis.csv', index=False)
    print(f"\nSaved: sensitivity_analysis.csv")
    
    return df_sensitivity


def run_multi_seed_analysis(output_dir: Path, num_seeds: int = 5):
    """
    Run experiments with different random seeds to measure variance.
    """
    print("\n" + "="*60)
    print(f"MULTI-SEED VARIANCE ANALYSIS ({num_seeds} seeds)")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    
    # Test configuration: 10K requests, medium load, 4 GPUs, K=4
    results_seeds = []
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        
        requests = load_burstgpt_dataset(
            str(base_dir / "data" / "BurstGPT_sample.csv"),
            num_requests=10000,
            d_sla_token=0.010,
            d_sla_request=20.0,
            use_real_timestamps=False,
            rps_scaling=10.0
        )
        
        cfg = SchedulerConfig(
            NUM_GPUS=4,
            K_BINS=4,
            D_SLA_TOKEN=0.010,
            D_SLA_REQUEST=20.0,
            NUM_REQUESTS=10000,
            DATASET_PATH=str(base_dir / "data" / "BurstGPT_sample.csv"),
            USE_REAL_TIMESTAMPS=False,
            RPS_SCALING=10.0,
            SEED=seed,
        )
        
        sim = Simulator(cfg, requests, scheduler_type="multi_bin_dynamic")
        completed = sim.run()
        metrics = compute_metrics(completed, d_sla_token=0.010, d_sla_request=20.0)
        
        results_seeds.append({
            'seed': seed,
            'request_sla_pct': metrics['sla_violation_rate_request'] * 100,
            'throughput_tok_s': metrics['throughput_tokens_per_sec'],
            'p95_latency_s': metrics['p95_latency'],
            'p99_latency_s': metrics['p99_latency'],
            'avg_latency_s': metrics['avg_latency'],
        })
        print(f"  Seed {seed}: SLA={metrics['sla_violation_rate_request']*100:.2f}%, "
              f"Throughput={metrics['throughput_tokens_per_sec']:.1f} tok/s, "
              f"P95={metrics['p95_latency']:.2f}s")
    
    df_seeds = pd.DataFrame(results_seeds)
    
    # Calculate statistics
    print("\n--- Statistical Summary ---")
    for col in ['request_sla_pct', 'throughput_tok_s', 'p95_latency_s']:
        mean = df_seeds[col].mean()
        std = df_seeds[col].std()
        ci_95 = 1.96 * std / np.sqrt(num_seeds)
        print(f"  {col}: {mean:.2f} ± {ci_95:.2f} (95% CI)")
    
    df_seeds.to_csv(output_dir / 'multi_seed_variance.csv', index=False)
    print(f"\nSaved: multi_seed_variance.csv")
    
    return df_seeds


def plot_sensitivity_results(df_sensitivity: pd.DataFrame, df_seeds: pd.DataFrame, output_dir: Path):
    """
    Figure 5: Sensitivity and variance analysis plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: K_BINS sensitivity
    ax = axes[0]
    k_data = df_sensitivity[df_sensitivity['parameter'] == 'K_BINS']
    if len(k_data) > 0:
        x = range(len(k_data))
        bars = ax.bar(x, k_data['request_sla_pct'], color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels([f'K={int(v)}' for v in k_data['value']])
        # Highlight default K=4
        default_idx = k_data[k_data['value'] == 4].index
        if len(default_idx) > 0:
            idx = list(k_data['value']).index(4)
            bars[idx].set_color('coral')
            bars[idx].set_edgecolor('red')
            bars[idx].set_linewidth(2)
            ax.axhline(y=k_data[k_data['value'] == 4]['request_sla_pct'].values[0], 
                       color='red', linestyle='--', alpha=0.7, label='Default (K=4)')
    ax.set_xlabel('Number of Priority Bins (K)', fontsize=12)
    ax.set_ylabel('Request SLA Violation (%)', fontsize=12)
    ax.set_title('K-Bins Parameter Sensitivity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: GPU count sensitivity
    ax = axes[1]
    gpu_data = df_sensitivity[df_sensitivity['parameter'] == 'NUM_GPUS']
    if len(gpu_data) > 0:
        x = range(len(gpu_data))
        # Plot both SLA and throughput
        ax2 = ax.twinx()
        
        bars1 = ax.bar([i - 0.2 for i in x], gpu_data['request_sla_pct'], 0.4, 
                       color='coral', label='SLA Violation %')
        bars2 = ax2.bar([i + 0.2 for i in x], gpu_data['throughput_tok_s'] / 1000, 0.4, 
                        color='steelblue', label='Throughput (K tok/s)')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(v)} GPUs' for v in gpu_data['value']])
        ax.set_xlabel('Number of GPUs', fontsize=12)
        ax.set_ylabel('SLA Violation (%)', fontsize=12, color='coral')
        ax2.set_ylabel('Throughput (K tokens/sec)', fontsize=12, color='steelblue')
        ax.tick_params(axis='y', labelcolor='coral')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title('GPU Scaling Analysis', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Variance across seeds
    ax = axes[2]
    if df_seeds is not None and len(df_seeds) > 0:
        metrics_to_plot = ['request_sla_pct', 'throughput_tok_s']
        x = np.arange(len(df_seeds))
        width = 0.35
        
        # Normalize for visualization
        sla_vals = df_seeds['request_sla_pct']
        tput_vals = df_seeds['throughput_tok_s'] / df_seeds['throughput_tok_s'].mean() * sla_vals.mean()
        
        ax.bar(x - width/2, sla_vals, width, label='SLA Violation %', color='coral')
        ax.bar(x + width/2, tput_vals, width, label='Throughput (normalized)', color='steelblue')
        
        ax.axhline(y=sla_vals.mean(), color='coral', linestyle='--', alpha=0.7)
        ax.axhline(y=tput_vals.mean(), color='steelblue', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Random Seed', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'Variance Across {len(df_seeds)} Seeds', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Seed {i}' for i in df_seeds['seed']])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add variance annotation
        ax.text(0.02, 0.98, f'SLA CV: {sla_vals.std()/sla_vals.mean()*100:.1f}%\n'
                           f'Tput CV: {df_seeds["throughput_tok_s"].std()/df_seeds["throughput_tok_s"].mean()*100:.1f}%',
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_sensitivity_variance.png', dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_sensitivity_variance.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig5_sensitivity_variance.png/pdf")


def main():
    """Main entry point."""
    print("="*60)
    print("GENERATING ANALYSIS PLOTS AND SENSITIVITY ANALYSIS")
    print("="*60)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Load existing results
    print("\nLoading experimental results...")
    results = load_all_results()
    print(f"Loaded {len(results)} result files")
    
    # Generate plots from existing data
    print("\n--- Generating Figures ---")
    
    plot_throughput_vs_gpu(results, output_dir)
    plot_sla_violation_pareto(results, output_dir)
    plot_latency_throughput_tradeoff(results, output_dir)
    plot_scheduler_comparison_bar(results, output_dir)
    
    # Run sensitivity analysis
    df_sensitivity = run_sensitivity_analysis(output_dir)
    
    # Run multi-seed variance analysis
    df_seeds = run_multi_seed_analysis(output_dir, num_seeds=5)
    
    # Plot sensitivity and variance results
    plot_sensitivity_results(df_sensitivity, df_seeds, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("Generated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
