#!/usr/bin/env python3
"""
Comprehensive graph analysis for stress test results.

Generates publication-quality plots for:
1. Request scaling analysis (10K -> 100K -> 1M)
2. GPU scaling analysis (4 -> 64 GPUs)
3. Scheduler comparison across all metrics
4. SLA compliance and capacity analysis
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import json

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_results(base_path):
    """Load all result CSV files."""
    results = {}
    
    all_results_path = base_path.replace('.csv', '_all_results.csv')
    step1_path = base_path.replace('.csv', '_step1_request_scaling.csv')
    step2_path = base_path.replace('.csv', '_step2_gpu_scaling.csv')
    
    if Path(all_results_path).exists():
        results['all'] = pd.read_csv(all_results_path)
        print(f"✓ Loaded {len(results['all'])} rows from {all_results_path}")
    
    if Path(step1_path).exists():
        results['step1'] = pd.read_csv(step1_path)
        print(f"✓ Loaded {len(results['step1'])} rows from {step1_path}")
    
    if Path(step2_path).exists():
        results['step2'] = pd.read_csv(step2_path)
        print(f"✓ Loaded {len(results['step2'])} rows from {step2_path}")
    
    return results


def plot_request_scaling_capacity(df, output_dir):
    """Plot capacity QPS vs request volume for each scheduler."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter to request scaling data
    schedulers = ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']
    colors = {'static_fifo': '#e74c3c', 'dynamic_no_bins': '#3498db', 'multi_bin_dynamic': '#2ecc71'}
    markers = {'static_fifo': 'o', 'dynamic_no_bins': 's', 'multi_bin_dynamic': '^'}
    
    # Plot 1: Capacity QPS (log scale)
    for scheduler in schedulers:
        data = df[df['scheduler_type'] == scheduler].sort_values('num_requests')
        if len(data) > 0:
            ax1.plot(data['num_requests'], data['capacity_qps_under_sla'], 
                    marker=markers[scheduler], label=scheduler.replace('_', ' ').title(),
                    linewidth=2, markersize=10, color=colors[scheduler])
    
    ax1.set_xlabel('Number of Requests', fontweight='bold')
    ax1.set_ylabel('Capacity QPS (under SLA)', fontweight='bold')
    ax1.set_title('Request Scaling: Capacity QPS vs Volume', fontweight='bold', pad=20)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for GPU counts
    for scheduler in schedulers:
        data = df[df['scheduler_type'] == scheduler].sort_values('num_requests')
        if len(data) > 0:
            for _, row in data.iterrows():
                ax1.annotate(f"{int(row['num_gpus'])}G", 
                           (row['num_requests'], row['capacity_qps_under_sla']),
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=8, alpha=0.7)
    
    # Plot 2: SLA Violation Rate
    for scheduler in schedulers:
        data = df[df['scheduler_type'] == scheduler].sort_values('num_requests')
        if len(data) > 0:
            ax2.plot(data['num_requests'], data['sla_violation_rate'] * 100, 
                    marker=markers[scheduler], label=scheduler.replace('_', ' ').title(),
                    linewidth=2, markersize=10, color=colors[scheduler])
    
    ax2.set_xlabel('Number of Requests', fontweight='bold')
    ax2.set_ylabel('SLA Violation Rate (%)', fontweight='bold')
    ax2.set_title('Request Scaling: SLA Violations vs Volume', fontweight='bold', pad=20)
    ax2.set_xscale('log')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
    
    plt.tight_layout()
    output_path = output_dir / 'request_scaling_capacity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_request_scaling_latency(df, output_dir):
    """Plot latency metrics vs request volume."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    schedulers = ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']
    colors = {'static_fifo': '#e74c3c', 'dynamic_no_bins': '#3498db', 'multi_bin_dynamic': '#2ecc71'}
    markers = {'static_fifo': 'o', 'dynamic_no_bins': 's', 'multi_bin_dynamic': '^'}
    
    # Plot 1: Average Latency (log scale)
    for scheduler in schedulers:
        data = df[df['scheduler_type'] == scheduler].sort_values('num_requests')
        if len(data) > 0:
            ax1.plot(data['num_requests'], data['avg_latency'], 
                    marker=markers[scheduler], label=scheduler.replace('_', ' ').title(),
                    linewidth=2, markersize=10, color=colors[scheduler])
    
    ax1.set_xlabel('Number of Requests', fontweight='bold')
    ax1.set_ylabel('Average Latency (seconds)', fontweight='bold')
    ax1.set_title('Request Scaling: Average Latency vs Volume', fontweight='bold', pad=20)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1s SLA')
    
    # Plot 2: P95 Latency
    for scheduler in schedulers:
        data = df[df['scheduler_type'] == scheduler].sort_values('num_requests')
        if len(data) > 0:
            ax2.plot(data['num_requests'], data['p95_latency'], 
                    marker=markers[scheduler], label=scheduler.replace('_', ' ').title(),
                    linewidth=2, markersize=10, color=colors[scheduler])
    
    ax2.set_xlabel('Number of Requests', fontweight='bold')
    ax2.set_ylabel('P95 Latency (seconds)', fontweight='bold')
    ax2.set_title('Request Scaling: P95 Latency vs Volume', fontweight='bold', pad=20)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1s SLA')
    
    plt.tight_layout()
    output_path = output_dir / 'request_scaling_latency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_gpu_scaling(df, output_dir):
    """Plot GPU scaling efficiency for multi_bin_dynamic."""
    # Filter to GPU scaling data (1M requests)
    data = df[(df['scheduler_type'] == 'multi_bin_dynamic') & 
              (df['num_requests'] == 1000000)].sort_values('num_gpus')
    
    if len(data) == 0:
        print("⚠ No GPU scaling data found (1M requests with multi_bin_dynamic)")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Capacity QPS vs GPUs
    ax1.plot(data['num_gpus'], data['capacity_qps_under_sla'], 
            marker='o', linewidth=2, markersize=10, color='#2ecc71')
    ax1.set_xlabel('Number of GPUs', fontweight='bold')
    ax1.set_ylabel('Capacity QPS (under SLA)', fontweight='bold')
    ax1.set_title('GPU Scaling: Capacity QPS', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Add ideal linear scaling reference
    if len(data) > 0:
        base_gpus = data.iloc[0]['num_gpus']
        base_qps = data.iloc[0]['capacity_qps_under_sla']
        ideal_qps = [base_qps * (g / base_gpus) for g in data['num_gpus']]
        ax1.plot(data['num_gpus'], ideal_qps, '--', alpha=0.5, color='gray', label='Linear scaling')
        ax1.legend()
    
    # Plot 2: SLA Violation Rate vs GPUs
    ax2.plot(data['num_gpus'], data['sla_violation_rate'] * 100, 
            marker='o', linewidth=2, markersize=10, color='#e74c3c')
    ax2.set_xlabel('Number of GPUs', fontweight='bold')
    ax2.set_ylabel('SLA Violation Rate (%)', fontweight='bold')
    ax2.set_title('GPU Scaling: SLA Violations', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% target')
    ax2.legend()
    
    # Plot 3: Average Latency vs GPUs (log scale)
    ax3.plot(data['num_gpus'], data['avg_latency'], 
            marker='o', linewidth=2, markersize=10, color='#3498db')
    ax3.set_xlabel('Number of GPUs', fontweight='bold')
    ax3.set_ylabel('Average Latency (seconds)', fontweight='bold')
    ax3.set_title('GPU Scaling: Average Latency', fontweight='bold', pad=20)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1s SLA')
    ax3.legend()
    
    # Plot 4: GPU Utilization vs GPUs
    ax4.plot(data['num_gpus'], data['avg_gpu_utilization'] * 100, 
            marker='o', linewidth=2, markersize=10, color='#9b59b6')
    ax4.set_xlabel('Number of GPUs', fontweight='bold')
    ax4.set_ylabel('Average GPU Utilization (%)', fontweight='bold')
    ax4.set_title('GPU Scaling: GPU Utilization', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% balanced')
    ax4.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% saturated')
    ax4.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'gpu_scaling_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_scaling_efficiency(df, output_dir):
    """Plot GPU scaling efficiency metrics."""
    data = df[(df['scheduler_type'] == 'multi_bin_dynamic') & 
              (df['num_requests'] == 1000000)].sort_values('num_gpus')
    
    if len(data) < 2:
        print("⚠ Insufficient data for scaling efficiency plot")
        return
    
    # Calculate scaling efficiency
    base_gpus = data.iloc[0]['num_gpus']
    base_qps = data.iloc[0]['capacity_qps_under_sla']
    
    gpu_ratios = []
    qps_ratios = []
    efficiencies = []
    
    for _, row in data.iterrows():
        gpu_ratio = row['num_gpus'] / base_gpus
        qps_ratio = row['capacity_qps_under_sla'] / base_qps
        efficiency = (qps_ratio / gpu_ratio) * 100  # Percentage
        
        gpu_ratios.append(gpu_ratio)
        qps_ratios.append(qps_ratio)
        efficiencies.append(efficiency)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: QPS Ratio vs GPU Ratio
    ax1.plot(gpu_ratios, qps_ratios, marker='o', linewidth=2, markersize=10, color='#2ecc71')
    ax1.plot(gpu_ratios, gpu_ratios, '--', alpha=0.5, color='gray', label='Linear scaling (ideal)')
    ax1.set_xlabel('GPU Count Ratio (vs baseline)', fontweight='bold')
    ax1.set_ylabel('QPS Ratio (vs baseline)', fontweight='bold')
    ax1.set_title('Scaling Behavior: QPS vs GPU Count', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Annotate GPU counts
    for i, (gpu_r, qps_r, gpus) in enumerate(zip(gpu_ratios, qps_ratios, data['num_gpus'])):
        ax1.annotate(f"{int(gpus)}G", (gpu_r, qps_r), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9)
    
    # Plot 2: Scaling Efficiency
    colors_eff = ['#2ecc71' if e >= 85 else '#f39c12' if e >= 60 else '#e74c3c' for e in efficiencies]
    bars = ax2.bar(range(len(data)), efficiencies, color=colors_eff, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Configuration', fontweight='bold')
    ax2.set_ylabel('Scaling Efficiency (%)', fontweight='bold')
    ax2.set_title('GPU Scaling Efficiency', fontweight='bold', pad=20)
    ax2.set_xticks(range(len(data)))
    ax2.set_xticklabels([f"{int(g)}G" for g in data['num_gpus']])
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% (linear)')
    ax2.axhline(y=85, color='green', linestyle='--', alpha=0.3, label='85% (good)')
    ax2.axhline(y=60, color='orange', linestyle='--', alpha=0.3, label='60% (acceptable)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'scaling_efficiency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_scheduler_comparison_heatmap(df, output_dir):
    """Create heatmap comparing schedulers across all metrics."""
    # Get representative data (10K requests for fair comparison)
    comparison_data = df[df['num_requests'] == 10000].copy()
    
    if len(comparison_data) == 0:
        print("⚠ No 10K request data for heatmap")
        return
    
    # Select key metrics
    metrics = [
        'capacity_qps_under_sla',
        'sla_violation_rate',
        'avg_latency',
        'p95_latency',
        'throughput_requests_per_sec',
        'avg_gpu_utilization',
        'avg_batch_size'
    ]
    
    # Create pivot table
    pivot_data = comparison_data.pivot_table(
        values=metrics,
        index='scheduler_type',
        aggfunc='mean'
    )
    
    # Normalize each metric to 0-1 range for visualization
    normalized = pivot_data.copy()
    for col in metrics:
        if col in ['sla_violation_rate', 'avg_latency', 'p95_latency']:
            # Lower is better - invert
            normalized[col] = 1 - (pivot_data[col] - pivot_data[col].min()) / (pivot_data[col].max() - pivot_data[col].min() + 1e-10)
        else:
            # Higher is better
            normalized[col] = (pivot_data[col] - pivot_data[col].min()) / (pivot_data[col].max() - pivot_data[col].min() + 1e-10)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_labels = [
        'Capacity QPS',
        'SLA Compliance',
        'Avg Latency',
        'P95 Latency',
        'Throughput',
        'GPU Util',
        'Batch Size'
    ]
    
    scheduler_labels = [s.replace('_', ' ').title() for s in normalized.index]
    
    sns.heatmap(normalized, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=metric_labels, yticklabels=scheduler_labels,
                cbar_kws={'label': 'Normalized Score (0-1)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Scheduler Comparison Heatmap (10K Requests)\nGreen = Better Performance', 
                fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = output_dir / 'scheduler_comparison_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_throughput_vs_latency_tradeoff(df, output_dir):
    """Plot throughput vs latency tradeoff for all configurations."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    schedulers = df['scheduler_type'].unique()
    colors = {'static_fifo': '#e74c3c', 'dynamic_no_bins': '#3498db', 'multi_bin_dynamic': '#2ecc71'}
    markers = {'static_fifo': 'o', 'dynamic_no_bins': 's', 'multi_bin_dynamic': '^'}
    
    for scheduler in schedulers:
        data = df[df['scheduler_type'] == scheduler]
        if len(data) > 0:
            ax.scatter(data['avg_latency'], data['throughput_requests_per_sec'],
                      s=data['num_gpus']*30, alpha=0.6, 
                      c=colors.get(scheduler, 'gray'),
                      marker=markers.get(scheduler, 'o'),
                      label=scheduler.replace('_', ' ').title(),
                      edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Average Latency (seconds)', fontweight='bold')
    ax.set_ylabel('Throughput (requests/sec)', fontweight='bold')
    ax.set_title('Throughput vs Latency Tradeoff\n(Bubble size = GPU count)', 
                fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1s SLA')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'throughput_latency_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def plot_batch_size_analysis(df, output_dir):
    """Plot batch size behavior across configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    schedulers = ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']
    colors = {'static_fifo': '#e74c3c', 'dynamic_no_bins': '#3498db', 'multi_bin_dynamic': '#2ecc71'}
    
    # Plot 1: Batch size vs request volume
    for scheduler in schedulers:
        data = df[df['scheduler_type'] == scheduler].sort_values('num_requests')
        if len(data) > 0:
            ax1.plot(data['num_requests'], data['avg_batch_size'], 
                    marker='o', label=scheduler.replace('_', ' ').title(),
                    linewidth=2, markersize=8, color=colors[scheduler])
    
    ax1.set_xlabel('Number of Requests', fontweight='bold')
    ax1.set_ylabel('Average Batch Size', fontweight='bold')
    ax1.set_title('Batch Size vs Request Volume', fontweight='bold', pad=20)
    ax1.set_xscale('log')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Batch size vs GPU count (for multi-bin only)
    data = df[(df['scheduler_type'] == 'multi_bin_dynamic') & 
              (df['num_requests'] == 1000000)].sort_values('num_gpus')
    if len(data) > 0:
        ax2.plot(data['num_gpus'], data['avg_batch_size'], 
                marker='o', linewidth=2, markersize=10, color='#2ecc71')
        ax2.set_xlabel('Number of GPUs', fontweight='bold')
        ax2.set_ylabel('Average Batch Size', fontweight='bold')
        ax2.set_title('Multi-Bin: Batch Size vs GPU Count (1M requests)', 
                     fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No GPU scaling data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    
    plt.tight_layout()
    output_path = output_dir / 'batch_size_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_path}")
    plt.close()


def create_summary_table(df, output_dir):
    """Create summary statistics table."""
    summary_data = []
    
    # Group by scheduler and request count
    for scheduler in df['scheduler_type'].unique():
        for num_reqs in sorted(df['num_requests'].unique()):
            data = df[(df['scheduler_type'] == scheduler) & 
                     (df['num_requests'] == num_reqs)]
            
            if len(data) > 0:
                row = data.iloc[0]
                summary_data.append({
                    'Scheduler': scheduler,
                    'Requests': f"{num_reqs:,}",
                    'GPUs': int(row['num_gpus']),
                    'Capacity QPS': f"{row['capacity_qps_under_sla']:.2f}",
                    'SLA Violations': f"{row['sla_violation_rate']*100:.1f}%",
                    'Avg Latency': f"{row['avg_latency']:.2f}s",
                    'P95 Latency': f"{row['p95_latency']:.2f}s",
                    'Throughput': f"{row['throughput_requests_per_sec']:.2f}",
                    'GPU Util': f"{row['avg_gpu_utilization']*100:.1f}%",
                    'Avg Batch': f"{row['avg_batch_size']:.1f}"
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_path = output_dir / 'summary_table.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Saved {output_path}")
    
    # Create formatted text table
    output_txt = output_dir / 'summary_table.txt'
    with open(output_txt, 'w') as f:
        f.write("="*120 + "\n")
        f.write("COMPREHENSIVE STRESS TEST RESULTS SUMMARY\n")
        f.write("="*120 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n" + "="*120 + "\n")
    
    print(f"✓ Saved {output_txt}")
    
    return summary_df


def generate_analysis_report(df, analysis_json_path, output_dir):
    """Generate comprehensive analysis report."""
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE STRESS TEST ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total configurations tested: {len(df)}\n\n")
        
        # Load JSON analysis if available
        if Path(analysis_json_path).exists():
            with open(analysis_json_path, 'r') as jf:
                analysis = json.load(jf)
            
            f.write("\nKEY FINDINGS:\n")
            f.write("-" * 80 + "\n")
            for finding in analysis.get('key_findings', []):
                f.write(f"  • {finding}\n")
        
        f.write("\n\nSCHEDULER COMPARISON (10K Requests):\n")
        f.write("-" * 80 + "\n")
        
        for scheduler in df['scheduler_type'].unique():
            data = df[(df['scheduler_type'] == scheduler) & 
                     (df['num_requests'] == 10000)]
            if len(data) > 0:
                row = data.iloc[0]
                f.write(f"\n{scheduler.upper().replace('_', ' ')}:\n")
                f.write(f"  GPUs: {int(row['num_gpus'])}\n")
                f.write(f"  Capacity QPS: {row['capacity_qps_under_sla']:.2f}\n")
                f.write(f"  SLA Violations: {row['sla_violation_rate']*100:.1f}%\n")
                f.write(f"  Avg Latency: {row['avg_latency']:.3f}s\n")
                f.write(f"  GPU Utilization: {row['avg_gpu_utilization']*100:.1f}%\n")
        
        # GPU Scaling Analysis
        gpu_data = df[(df['scheduler_type'] == 'multi_bin_dynamic') & 
                     (df['num_requests'] == 1000000)].sort_values('num_gpus')
        
        if len(gpu_data) >= 2:
            f.write("\n\nGPU SCALING ANALYSIS (1M Requests, Multi-Bin):\n")
            f.write("-" * 80 + "\n")
            
            base = gpu_data.iloc[0]
            peak = gpu_data.iloc[-1]
            
            f.write(f"\nBaseline ({int(base['num_gpus'])} GPUs):\n")
            f.write(f"  Capacity QPS: {base['capacity_qps_under_sla']:.2f}\n")
            f.write(f"  SLA Violations: {base['sla_violation_rate']*100:.1f}%\n")
            
            f.write(f"\nPeak ({int(peak['num_gpus'])} GPUs):\n")
            f.write(f"  Capacity QPS: {peak['capacity_qps_under_sla']:.2f}\n")
            f.write(f"  SLA Violations: {peak['sla_violation_rate']*100:.1f}%\n")
            
            gpu_ratio = peak['num_gpus'] / base['num_gpus']
            qps_ratio = peak['capacity_qps_under_sla'] / base['capacity_qps_under_sla']
            efficiency = (qps_ratio / gpu_ratio) * 100
            
            f.write(f"\nScaling Efficiency:\n")
            f.write(f"  GPU Ratio: {gpu_ratio:.1f}x\n")
            f.write(f"  QPS Ratio: {qps_ratio:.2f}x\n")
            f.write(f"  Efficiency: {efficiency:.0f}%\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Saved {report_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze and plot comprehensive stress test results')
    parser.add_argument('--input', type=str, default='comprehensive_research_results_fixed.csv',
                       help='Base path for result CSV files')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE STRESS TEST GRAPH ANALYSIS")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}/")
    print()
    
    # Load results
    print("Loading results...")
    results = load_results(args.input)
    
    if 'all' not in results:
        print("❌ Error: Could not find all_results.csv file")
        return
    
    df = results['all']
    print(f"✓ Loaded {len(df)} test configurations\n")
    
    # Generate all plots
    print("Generating plots...")
    print("-" * 80)
    
    if 'step1' in results and len(results['step1']) > 0:
        plot_request_scaling_capacity(results['step1'], output_dir)
        plot_request_scaling_latency(results['step1'], output_dir)
    
    if 'step2' in results and len(results['step2']) > 0:
        plot_gpu_scaling(results['step2'], output_dir)
        plot_scaling_efficiency(results['step2'], output_dir)
    
    plot_scheduler_comparison_heatmap(df, output_dir)
    plot_throughput_vs_latency_tradeoff(df, output_dir)
    plot_batch_size_analysis(df, output_dir)
    
    print("-" * 80)
    
    # Generate summary table
    print("\nGenerating summary tables...")
    create_summary_table(df, output_dir)
    
    # Generate analysis report
    print("\nGenerating analysis report...")
    analysis_json = args.input.replace('.csv', '_analysis.json')
    generate_analysis_report(df, analysis_json, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All plots and reports saved to: {output_dir}/")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  • {file.name}")
    print()


if __name__ == "__main__":
    main()
