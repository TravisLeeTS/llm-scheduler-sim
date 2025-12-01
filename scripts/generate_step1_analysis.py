#!/usr/bin/env python3
"""
Generate comprehensive analysis and plots for Step 1 Grid Search results.
"""

import pandas as pd
import numpy as np
import os

def load_all_results():
    """Load and merge all Step 1 result files."""
    csv_files = [f for f in os.listdir('.') if f.startswith('step1') and f.endswith('.csv') 
                 and 'merged' not in f and 'analysis' not in f]
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"Loaded {f}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not dfs:
        return None
    
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate by key columns
    df_all = df_all.drop_duplicates(subset=['num_requests', 'num_gpus', 'k_bins'], keep='last')
    
    return df_all

def analyze_optimal_configs(df):
    """Analyze and find optimal configurations with diminishing returns."""
    
    recommendations = {}
    
    for req in sorted(df['num_requests'].unique()):
        df_req = df[df['num_requests'] == req].copy()
        
        # Calculate efficiency metric
        df_req['efficiency'] = df_req['throughput_tokens_per_sec'] / df_req['num_gpus']
        df_req['latency_per_token_ms'] = df_req['p95_latency'] / df_req['num_requests'] * 1000
        
        # Find configuration with best balance (considering diminishing returns)
        # Sort by GPUs to find where improvements become marginal
        df_sorted = df_req.sort_values(['k_bins', 'num_gpus'])
        
        best_configs = []
        
        for bins in df_req['k_bins'].unique():
            df_bins = df_req[df_req['k_bins'] == bins].sort_values('num_gpus')
            
            prev_lat = None
            prev_gpus = None
            prev_tput = None
            
            for _, row in df_bins.iterrows():
                gpus = int(row['num_gpus'])
                lat = row['p95_latency']
                tput = row['throughput_tokens_per_sec']
                
                if prev_lat is not None:
                    lat_improvement = (prev_lat - lat) / prev_lat * 100 if prev_lat > 0 else 0
                    tput_improvement = (tput - prev_tput) / prev_tput * 100 if prev_tput > 0 else 0
                    gpu_increase = gpus / prev_gpus
                    
                    # If doubling GPUs gives less than 20% latency improvement, prev config is recommended
                    if gpu_increase >= 2 and lat_improvement < 20 and prev_gpus is not None:
                        best_configs.append({
                            'bins': bins,
                            'gpus': prev_gpus,
                            'latency': prev_lat,
                            'throughput': prev_tput,
                            'reason': 'diminishing_returns'
                        })
                        break
                
                prev_lat = lat
                prev_gpus = gpus
                prev_tput = tput
        
        # Find absolute best
        best_lat_idx = df_req['p95_latency'].idxmin()
        best_lat_config = df_req.loc[best_lat_idx]
        
        best_tput_idx = df_req['throughput_tokens_per_sec'].idxmax()
        best_tput_config = df_req.loc[best_tput_idx]
        
        best_eff_idx = df_req['efficiency'].idxmax()
        best_eff_config = df_req.loc[best_eff_idx]
        
        recommendations[req] = {
            'best_latency': {
                'gpus': int(best_lat_config['num_gpus']),
                'bins': int(best_lat_config['k_bins']),
                'p95_latency': best_lat_config['p95_latency'],
                'throughput': best_lat_config['throughput_tokens_per_sec']
            },
            'best_throughput': {
                'gpus': int(best_tput_config['num_gpus']),
                'bins': int(best_tput_config['k_bins']),
                'p95_latency': best_tput_config['p95_latency'],
                'throughput': best_tput_config['throughput_tokens_per_sec']
            },
            'best_efficiency': {
                'gpus': int(best_eff_config['num_gpus']),
                'bins': int(best_eff_config['k_bins']),
                'efficiency': best_eff_config['efficiency'],
                'throughput': best_eff_config['throughput_tokens_per_sec']
            },
            'diminishing_returns_configs': best_configs
        }
    
    return recommendations

def generate_summary_table(df, recommendations):
    """Generate summary table for each workload size."""
    
    summary_rows = []
    
    for req in sorted(df['num_requests'].unique()):
        rec = recommendations[req]
        
        summary_rows.append({
            'Workload (Requests)': f"{req:,}",
            'Best Latency GPU/Bins': f"{rec['best_latency']['gpus']}/{rec['best_latency']['bins']}",
            'Best Latency P95 (s)': f"{rec['best_latency']['p95_latency']:.3f}",
            'Best Throughput GPU/Bins': f"{rec['best_throughput']['gpus']}/{rec['best_throughput']['bins']}",
            'Best Throughput (tok/s)': f"{rec['best_throughput']['throughput']:.0f}",
            'Best Efficiency GPU/Bins': f"{rec['best_efficiency']['gpus']}/{rec['best_efficiency']['bins']}",
            'Efficiency (tok/s/GPU)': f"{rec['best_efficiency']['efficiency']:.0f}"
        })
    
    return pd.DataFrame(summary_rows)

def generate_scaling_analysis(df):
    """Generate GPU scaling analysis for each workload."""
    
    scaling_data = []
    
    for req in sorted(df['num_requests'].unique()):
        df_req = df[df['num_requests'] == req]
        
        # Use K_BINS=2 as reference
        df_bins2 = df_req[df_req['k_bins'] == 2].sort_values('num_gpus')
        
        for _, row in df_bins2.iterrows():
            scaling_data.append({
                'workload': req,
                'gpus': int(row['num_gpus']),
                'bins': 2,
                'p95_latency': row['p95_latency'],
                'throughput': row['throughput_tokens_per_sec'],
                'gpu_util': row['avg_gpu_utilization']
            })
    
    return pd.DataFrame(scaling_data)

def generate_heatmap_data(df):
    """Generate data for GPU vs Bins heatmap for each workload."""
    
    heatmaps = {}
    
    for req in sorted(df['num_requests'].unique()):
        df_req = df[df['num_requests'] == req]
        
        # Create pivot tables
        lat_pivot = df_req.pivot_table(
            values='p95_latency', 
            index='num_gpus', 
            columns='k_bins',
            aggfunc='mean'
        )
        
        tput_pivot = df_req.pivot_table(
            values='throughput_tokens_per_sec',
            index='num_gpus',
            columns='k_bins',
            aggfunc='mean'
        )
        
        heatmaps[req] = {
            'latency': lat_pivot,
            'throughput': tput_pivot
        }
    
    return heatmaps

def print_analysis_report(df, recommendations, scaling_df, summary_df):
    """Print comprehensive analysis report."""
    
    print("\n" + "="*80)
    print("STEP 1 GRID SEARCH ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"Workload sizes: {sorted(df['num_requests'].unique())}")
    print(f"GPU counts: {sorted(df['num_gpus'].unique())}")
    print(f"Bin counts: {sorted(df['k_bins'].unique())}")
    
    print("\n" + "-"*80)
    print("SUMMARY TABLE")
    print("-"*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print("DETAILED RECOMMENDATIONS")
    print("-"*80)
    
    for req in sorted(recommendations.keys()):
        rec = recommendations[req]
        print(f"\n=== {req:,} REQUESTS ===")
        
        print(f"\n  BEST LATENCY:")
        print(f"    Configuration: {rec['best_latency']['gpus']} GPUs, {rec['best_latency']['bins']} bins")
        print(f"    P95 Latency: {rec['best_latency']['p95_latency']:.3f}s")
        print(f"    Throughput: {rec['best_latency']['throughput']:.0f} tok/s")
        
        print(f"\n  BEST THROUGHPUT:")
        print(f"    Configuration: {rec['best_throughput']['gpus']} GPUs, {rec['best_throughput']['bins']} bins")
        print(f"    Throughput: {rec['best_throughput']['throughput']:.0f} tok/s")
        print(f"    P95 Latency: {rec['best_throughput']['p95_latency']:.3f}s")
        
        print(f"\n  BEST EFFICIENCY (Cost-Optimal):")
        print(f"    Configuration: {rec['best_efficiency']['gpus']} GPUs, {rec['best_efficiency']['bins']} bins")
        print(f"    Efficiency: {rec['best_efficiency']['efficiency']:.0f} tok/s per GPU")
        print(f"    Total Throughput: {rec['best_efficiency']['throughput']:.0f} tok/s")
        
        if rec['diminishing_returns_configs']:
            print(f"\n  DIMINISHING RETURNS POINTS:")
            for cfg in rec['diminishing_returns_configs']:
                print(f"    - Bins={cfg['bins']}: {cfg['gpus']} GPUs recommended")
                print(f"      (Adding more GPUs provides <20% latency improvement)")
    
    print("\n" + "-"*80)
    print("GPU SCALING ANALYSIS (K_BINS=2)")
    print("-"*80)
    
    for req in sorted(df['num_requests'].unique()):
        df_req = scaling_df[scaling_df['workload'] == req].sort_values('gpus')
        print(f"\n{req:,} REQUESTS:")
        
        prev_lat = None
        for _, row in df_req.iterrows():
            gpus = int(row['gpus'])
            lat = row['p95_latency']
            tput = row['throughput']
            
            if prev_lat is not None:
                improvement = (prev_lat - lat) / prev_lat * 100
                status = "✓ Significant" if improvement > 20 else "⚠ Marginal" if improvement > 5 else "✗ Negligible"
                print(f"  {gpus:3d} GPUs: {lat:10.3f}s latency, {tput:8.0f} tok/s ({improvement:+.1f}% lat reduction) {status}")
            else:
                print(f"  {gpus:3d} GPUs: {lat:10.3f}s latency, {tput:8.0f} tok/s (baseline)")
            
            prev_lat = lat

def main():
    print("Loading Step 1 results...")
    df = load_all_results()
    
    if df is None or len(df) == 0:
        print("ERROR: No results found!")
        return
    
    print(f"\nLoaded {len(df)} unique configurations")
    
    # Generate analysis
    print("\nAnalyzing optimal configurations...")
    recommendations = analyze_optimal_configs(df)
    
    print("Generating summary table...")
    summary_df = generate_summary_table(df, recommendations)
    
    print("Analyzing GPU scaling...")
    scaling_df = generate_scaling_analysis(df)
    
    # Print report
    print_analysis_report(df, recommendations, scaling_df, summary_df)
    
    # Save outputs
    summary_df.to_csv('step1_summary_table.csv', index=False)
    scaling_df.to_csv('step1_scaling_analysis.csv', index=False)
    df.to_csv('step1_all_results.csv', index=False)
    
    print("\n" + "="*80)
    print("OUTPUT FILES SAVED:")
    print("  - step1_summary_table.csv")
    print("  - step1_scaling_analysis.csv")
    print("  - step1_all_results.csv")
    print("="*80)

if __name__ == '__main__':
    main()
