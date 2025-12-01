#!/usr/bin/env python3
"""
Analyze and visualize stress_test_v2.py results.
Generates publication-quality plots for the paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_results():
    """Load and preprocess results."""
    df = pd.read_csv('results_v2.csv')
    
    # Convert violation rates to pass rates
    df['token_pass_pct'] = (1 - df['sla_violation_rate_token']) * 100
    df['req_pass_pct'] = (1 - df['sla_violation_rate_request']) * 100
    
    return df

def plot_step2_comparison(df):
    """Create Step 2 method comparison bar chart."""
    # Get latest Step 2 results
    step2 = df.tail(12).copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Bin Dynamic Scheduling: Method Comparison\n(Token SLA = 30ms, Request SLA = 20s)', 
                 fontsize=14, fontweight='bold')
    
    colors = {
        'static_fifo': '#d62728',       # Red
        'dynamic_no_bins': '#ff7f0e',   # Orange
        'multi_bin_dynamic': '#2ca02c', # Green
    }
    
    for col_idx, n_req in enumerate([1000, 10000]):
        subset = step2[step2['num_requests'] == n_req].copy()
        
        # Create readable labels
        labels = []
        for _, row in subset.iterrows():
            gpu = int(row['num_gpus'])
            k = int(row['k_bins'])
            if row['scheduler_type'] == 'static_fifo':
                labels.append(f'Static FIFO\n(1 GPU)')
            elif row['scheduler_type'] == 'dynamic_no_bins':
                labels.append(f'Dynamic\n(1 GPU)')
            else:
                labels.append(f'Multi-Bin K={k}\n({gpu} GPU)')
        
        subset['label'] = labels
        bar_colors = [colors.get(s, '#1f77b4') for s in subset['scheduler_type']]
        
        x = np.arange(len(labels))
        width = 0.6
        
        # Token SLA Pass Rate
        ax = axes[0, col_idx]
        bars = ax.bar(x, subset['token_pass_pct'], width, color=bar_colors, edgecolor='black', linewidth=1)
        ax.axhline(y=95, color='green', linestyle='--', linewidth=2, label='95% Target')
        ax.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, label='90% Threshold')
        ax.set_ylabel('Token SLA Pass Rate (%)', fontsize=11)
        ax.set_title(f'{n_req:,} Requests - Token SLA (30ms)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, subset['token_pass_pct']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Request SLA Pass Rate
        ax = axes[1, col_idx]
        bars = ax.bar(x, subset['req_pass_pct'], width, color=bar_colors, edgecolor='black', linewidth=1)
        ax.axhline(y=95, color='green', linestyle='--', linewidth=2, label='95% Target')
        ax.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, label='90% Threshold')
        ax.set_ylabel('Request SLA Pass Rate (%)', fontsize=11)
        ax.set_title(f'{n_req:,} Requests - Request SLA (20s)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, subset['req_pass_pct']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'step2_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()

def plot_tbt_comparison(df):
    """Create TBT (Time Between Tokens) comparison."""
    step2 = df.tail(12).copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Time Between Tokens (TBT) Analysis by Scheduler', fontsize=14, fontweight='bold')
    
    for col_idx, n_req in enumerate([1000, 10000]):
        subset = step2[step2['num_requests'] == n_req].copy()
        
        labels = []
        for _, row in subset.iterrows():
            gpu = int(row['num_gpus'])
            k = int(row['k_bins'])
            if row['scheduler_type'] == 'static_fifo':
                labels.append(f'Static\n(1 GPU)')
            elif row['scheduler_type'] == 'dynamic_no_bins':
                labels.append(f'Dynamic\n(1 GPU)')
            else:
                labels.append(f'MB K={k}\n({gpu} GPU)')
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax = axes[col_idx]
        bars_avg = ax.bar(x - width/2, subset['avg_tbt_ms'], width, label='Avg TBT', color='steelblue', edgecolor='black')
        bars_p95 = ax.bar(x + width/2, subset['p95_tbt_ms'], width, label='P95 TBT', color='coral', edgecolor='black')
        
        ax.axhline(y=30, color='red', linestyle='--', linewidth=2, label='30ms SLA')
        ax.set_ylabel('Time Between Tokens (ms)', fontsize=11)
        ax.set_title(f'{n_req:,} Requests', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Value labels
        for bar in bars_avg:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars_p95:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'tbt_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()

def generate_summary_report(df):
    """Generate text summary report."""
    step2 = df.tail(12).copy()
    step2['token_pass_pct'] = (1 - step2['sla_violation_rate_token']) * 100
    step2['req_pass_pct'] = (1 - step2['sla_violation_rate_request']) * 100
    
    report = []
    report.append("="*80)
    report.append("STRESS TEST v2 RESULTS SUMMARY")
    report.append("="*80)
    report.append("")
    report.append("SLA Configuration (Gemini-Like):")
    report.append(f"  Token SLA:   30ms (33 tok/s minimum)")
    report.append(f"  Request SLA: 20s (total latency)")
    report.append("")
    report.append("Latency Model:")
    report.append("  latency = 59.65ms (base) + 5.74ms × output_tokens")
    report.append("  R² = 0.9995 (fitted from RTX 4080 + Qwen3 1.7B)")
    report.append("")
    
    for n_req in [1000, 10000]:
        report.append("-"*80)
        report.append(f"{n_req:,} REQUESTS")
        report.append("-"*80)
        subset = step2[step2['num_requests'] == n_req]
        
        for _, row in subset.iterrows():
            gpu = int(row['num_gpus'])
            k = int(row['k_bins'])
            name = row['scheduler_type']
            if name == 'static_fifo':
                label = f"Static FIFO (1 GPU)"
            elif name == 'dynamic_no_bins':
                label = f"Dynamic No-Bins (1 GPU)"
            else:
                label = f"Multi-Bin K={k} ({gpu} GPU{'s' if gpu > 1 else ''})"
            
            report.append(f"\n{label}:")
            report.append(f"  Token SLA Pass:   {row['token_pass_pct']:6.1f}%")
            report.append(f"  Request SLA Pass: {row['req_pass_pct']:6.1f}%")
            report.append(f"  Avg TBT:          {row['avg_tbt_ms']:6.1f} ms")
            report.append(f"  P95 TBT:          {row['p95_tbt_ms']:6.1f} ms")
            report.append(f"  GPU Utilization:  {row['avg_gpu_utilization']*100:6.1f}%")
    
    report.append("")
    report.append("="*80)
    report.append("KEY FINDINGS")
    report.append("="*80)
    report.append("")
    report.append("1. TOKEN SLA IMPROVEMENT:")
    
    # Calculate improvements
    for n_req in [1000, 10000]:
        subset = step2[step2['num_requests'] == n_req]
        static = subset[subset['scheduler_type'] == 'static_fifo'].iloc[0]
        mb8_1gpu = subset[(subset['scheduler_type'] == 'multi_bin_dynamic') & (subset['num_gpus'] == 1) & (subset['k_bins'] == 8)].iloc[0]
        
        improvement = mb8_1gpu['token_pass_pct'] / static['token_pass_pct'] if static['token_pass_pct'] > 0 else float('inf')
        report.append(f"   {n_req:,} req: Multi-Bin K=8 ({mb8_1gpu['token_pass_pct']:.1f}%) vs Static ({static['token_pass_pct']:.1f}%)")
        report.append(f"             → {improvement:.2f}x improvement")
    
    report.append("")
    report.append("2. REQUEST SLA IMPROVEMENT:")
    report.append("   Single GPU configurations cannot meet 20s Request SLA at scale.")
    report.append("   Multi-GPU scaling (32 GPUs) achieves 98%+ Request SLA compliance.")
    
    report.append("")
    report.append("3. TBT REDUCTION:")
    for n_req in [1000, 10000]:
        subset = step2[step2['num_requests'] == n_req]
        static = subset[subset['scheduler_type'] == 'static_fifo'].iloc[0]
        mb8_1gpu = subset[(subset['scheduler_type'] == 'multi_bin_dynamic') & (subset['num_gpus'] == 1) & (subset['k_bins'] == 8)].iloc[0]
        
        reduction = (1 - mb8_1gpu['avg_tbt_ms'] / static['avg_tbt_ms']) * 100
        report.append(f"   {n_req:,} req: {static['avg_tbt_ms']:.1f}ms → {mb8_1gpu['avg_tbt_ms']:.1f}ms ({reduction:.1f}% reduction)")
    
    report_text = '\n'.join(report)
    
    output_path = os.path.join(PLOTS_DIR, 'step2_report.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f'Saved: {output_path}')
    
    print("\n" + report_text)

def main():
    print("="*80)
    print("STRESS TEST v2 - ANALYSIS AND VISUALIZATION")
    print("="*80)
    
    df = load_results()
    print(f"Loaded {len(df)} results from results_v2.csv")
    
    print("\nGenerating plots...")
    plot_step2_comparison(df)
    plot_tbt_comparison(df)
    
    print("\nGenerating summary report...")
    generate_summary_report(df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
