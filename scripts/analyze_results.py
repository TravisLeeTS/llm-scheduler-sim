#!/usr/bin/env python3
"""Analyze Step 1 grid search results."""

import pandas as pd

df = pd.read_csv('stress_test_v3_results/step1_grid_search.csv')

print('='*80)
print('STEP 1 GRID SEARCH RESULTS SUMMARY')
print('='*80)
print(f'Total configurations: {len(df)}')
print()
print('Configurations by workload:')
print(df.groupby('num_requests').size())
print()

# Compute pass rates
df['token_pass'] = (1 - df['sla_violation_rate_token']) * 100
df['req_pass'] = (1 - df['sla_violation_rate_request']) * 100
df['combined'] = df['token_pass'] + df['req_pass']

# Token SLA analysis
print('='*80)
print('TOKEN SLA PASS RATE (%) by GPUs and num_requests')
print('(Higher is better)')
print('='*80)
pivot_token = df.pivot_table(values='token_pass', index='num_gpus', columns='num_requests', aggfunc='mean')
print(pivot_token.round(1))

# Request SLA analysis
print()
print('='*80)
print('REQUEST SLA PASS RATE (%) by GPUs and num_requests')
print('(Higher is better)')
print('='*80)
pivot_req = df.pivot_table(values='req_pass', index='num_gpus', columns='num_requests', aggfunc='mean')
print(pivot_req.round(1))

# Combined score analysis
print()
print('='*80)
print('COMBINED SCORE (Token + Request) by GPUs and num_requests')
print('='*80)
pivot_combined = df.pivot_table(values='combined', index='num_gpus', columns='num_requests', aggfunc='mean')
print(pivot_combined.round(1))

# Optimal configurations
print()
print('='*80)
print('OPTIMAL CONFIGURATIONS per workload (by Combined Score)')
print('='*80)
for req in sorted(df['num_requests'].unique()):
    subset = df[df['num_requests'] == req]
    best = subset.loc[subset['combined'].idxmax()]
    print(f"\n{req:,} requests:")
    print(f"  Best: GPUs={int(best['num_gpus']):3d}, K={int(best['k_bins']):2d}")
    print(f"  Token: {best['token_pass']:.1f}%, Request: {best['req_pass']:.1f}%, Combined: {best['combined']:.1f}%")
    print(f"  Batch: {best['avg_batch_size']:.1f}, GPU Util: {best['avg_gpu_utilization']*100:.1f}%")

# K_BINS analysis
print()
print('='*80)
print('K_BINS IMPACT (averaged across all GPUs)')
print('='*80)
k_impact = df.groupby('k_bins').agg({
    'token_pass': 'mean',
    'req_pass': 'mean',
    'avg_batch_size': 'mean',
    'combined': 'mean'
}).round(1)
print(k_impact)

# GPU count impact
print()
print('='*80)
print('GPU COUNT IMPACT (averaged across all K_BINS)')
print('='*80)
gpu_impact = df.groupby('num_gpus').agg({
    'token_pass': 'mean',
    'req_pass': 'mean',
    'avg_batch_size': 'mean',
    'avg_gpu_utilization': lambda x: x.mean() * 100,
    'combined': 'mean'
}).round(1)
gpu_impact.columns = ['Token%', 'Req%', 'AvgBatch', 'GPU%', 'Combined']
print(gpu_impact)
