#!/usr/bin/env python3
"""
Production-Realistic Stress Testing for LLM Inference Optimization (v2)

GEMINI-LIKE SLA CONFIGURATION:
Based on production cloud providers (Gemini Flash Lite, Claude, etc.):
- Output speeds: 100-500+ tokens/sec → per-token latency: 2-10ms
- Time to First Token (TTFT): 0.3s - 1s for fast models
- Typical response: 200-800 tokens

SLA Thresholds (Realistic Production):

  TOKEN SLA (Per-Token Decode Latency / TBT):
    - Strict: 10ms (100 tok/s minimum, Gemini-like)
    - Loose:  30ms (for stress test differentiation)
    
  REQUEST SLA (End-to-End Response Latency):
    - Strict:  5s  (tight production)
    - Default: 10s (realistic user-facing)
    - Loose:  20s  (long generation tolerance)
    
    Rationale: 200-800 tokens @ 10ms/tok = 2-8s decode + 0.3-1s TTFT ≈ 10s p95

Author: Stress Test Script v2 (Gemini-Like Production SLA)
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mb_dyn_sim.config import SchedulerConfig
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.simulation import Simulator
from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics

# ============================================================================
# PRODUCTION SLA CONFIGURATION (Based on Artificial Analysis benchmarks)
# ============================================================================

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

# ============================================================================
# GEMINI-LIKE SLA CONFIGURATION
# ============================================================================
# Latency model: latency = 60ms base + 5.74ms * output_tokens
# At steady state: ~6ms per token decode
#
# TOKEN SLA OPTIONS:
#   Strict: 10ms (100 tok/s) - Production cloud target
#   Loose:  30ms (33 tok/s)  - Stress test differentiation
#
# REQUEST SLA OPTIONS (tied to typical 200-800 token responses + TTFT):
#   Strict:  5s  - Tight production
#   Default: 10s - Realistic user-facing (200tok@10ms + TTFT = ~3s, 800tok = ~9s)
#   Loose:  20s  - Long generation tolerance
#
# Set your test mode:
SLA_MODE = "loose"  # Options: "strict", "default", "loose"

if SLA_MODE == "strict":
    D_SLA_TOKEN = 0.010    # 10ms per token (100 tok/s minimum)
    D_SLA_REQUEST = 5.0    # 5s total request latency
elif SLA_MODE == "loose":
    D_SLA_TOKEN = 0.030    # 30ms per token (stress test mode)
    D_SLA_REQUEST = 20.0   # 20s for long generations
else:  # default
    D_SLA_TOKEN = 0.010    # 10ms per token (Gemini-like)
    D_SLA_REQUEST = 10.0   # 10s total request latency (realistic)

# Simulation scaling
RPS_SCALING = 200.0    # Scale BurstGPT timestamps for stress testing

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results_v2')


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
        
        bin_boundaries = []
        for i in range(k_bins):
            min_val = boundaries[i] if i == 0 else boundaries[i]
            max_val = boundaries[i + 1] if i < k_bins - 1 else 10000
            if min_val >= max_val:
                max_val = min_val + 1
            bin_boundaries.append((min_val, max_val))
        return bin_boundaries
    except Exception as e:
        return [(1, 27), (27, 101), (101, 188), (188, 10000)][:k_bins]


def run_single_simulation(
    num_requests: int,
    num_gpus: int,
    k_bins: int,
    scheduler_type: str,
    rps_scaling: float = RPS_SCALING,
    verbose: bool = False
) -> dict:
    """Run a single simulation and return metrics with dual SLA."""
    
    start_time = time.time()
    
    try:
        # Compute bin boundaries
        if scheduler_type in ["multi_bin_dynamic", "multibin_dynamic"]:
            bin_boundaries = compute_bin_boundaries_from_data(k_bins)
        else:
            bin_boundaries = [(1, 10000)]
            k_bins = 1
        
        actual_scheduler = "multi_bin_dynamic" if scheduler_type == "multibin_dynamic" else scheduler_type
        
        # Create configuration
        cfg = SchedulerConfig(
            NUM_GPUS=num_gpus,
            K_BINS=k_bins if actual_scheduler == "multi_bin_dynamic" else 1,
            NUM_REQUESTS=num_requests,
            SEED=42,
            D_SLA=D_SLA_TOKEN,
            D_SLA_TOKEN=D_SLA_TOKEN,
            D_SLA_REQUEST=D_SLA_REQUEST,
            DATASET_PATH=DATASET_PATH,
            WORKLOAD_SOURCE="burstgpt_dataset",
            USE_REAL_TIMESTAMPS=False,
            RPS_SCALING=rps_scaling,
            USE_EQUAL_MASS_BINS=True,
            USE_REAL_CALIBRATION=True,
            CALIBRATION_CSV_PATH=CALIBRATION_CSV,
            BIN_BOUNDARIES=bin_boundaries if actual_scheduler == "multi_bin_dynamic" else None,
        )
        
        # Generate workload (suppress output for speed)
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            requests = generate_workload(cfg)
        
        time_span = max(r.arrival_time for r in requests) - min(r.arrival_time for r in requests)
        actual_rps = len(requests) / time_span if time_span > 0 else 0
        
        # Run simulation
        simulator = Simulator(cfg, requests, actual_scheduler)
        completed_requests = simulator.run()
        
        # Compute metrics with dual SLA
        metrics = compute_metrics(completed_requests, d_sla_token=D_SLA_TOKEN, d_sla_request=D_SLA_REQUEST)
        gpu_stats = simulator.get_gpu_stats()
        gpu_metrics = compute_gpu_utilization(gpu_stats)
        batch_stats = compute_batch_statistics(completed_requests)
        
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
            
            # Dual SLA thresholds
            'd_sla_token': D_SLA_TOKEN,
            'd_sla_request': D_SLA_REQUEST,
            
            # ===== DUAL SLA METRICS =====
            'sla_violation_rate_token': metrics.get('sla_violation_rate_token', 0),
            'sla_violation_rate_request': metrics.get('sla_violation_rate_request', 0),
            'sla_violations_token': metrics.get('sla_violations_token', 0),
            'sla_violations_request': metrics.get('sla_violations_request', 0),
            
            # Legacy (backward compatibility)
            'sla_violation_rate': metrics.get('sla_violation_rate', 0),
            
            # Throughput
            'throughput_requests_per_sec': metrics.get('throughput_requests_per_sec', 0),
            'throughput_tokens_per_sec': metrics.get('throughput_tokens_per_sec', 0),
            'capacity_qps_under_sla': metrics.get('capacity_qps_under_sla', 0),
            
            # Per-token TBT metrics
            'avg_tbt_ms': metrics.get('avg_tbt', 0) * 1000,
            'p95_tbt_ms': metrics.get('p95_tbt', 0) * 1000,
            'p99_tbt_ms': metrics.get('p99_tbt', 0) * 1000,
            'max_tbt_ms': metrics.get('max_tbt', 0) * 1000,
            
            # Latency metrics (seconds)
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
            token_pass = (1 - result['sla_violation_rate_token']) * 100
            req_pass = (1 - result['sla_violation_rate_request']) * 100
            print(f"  -> Token PASS: {token_pass:.2f}%, "
                  f"Req PASS: {req_pass:.2f}%, "
                  f"GPU: {result['avg_gpu_utilization']*100:.1f}% [{execution_time:.1f}s]")
        
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


def run_step1_grid_search(args):
    """Step 1: Multi-Bin + Dynamic Grid Search with dual SLA tracking."""
    
    print("\n" + "="*80)
    print("STEP 1: MULTI-BIN + DYNAMIC GRID SEARCH (v2)")
    print("="*80)
    print(f"SLA Thresholds: Token={D_SLA_TOKEN*1000:.0f}ms, Request={D_SLA_REQUEST:.1f}s")
    
    request_counts = [int(x) for x in args.request_counts.split(',')]
    gpu_counts = [int(x) for x in args.gpu_counts.split(',')]
    bin_counts = [int(x) for x in args.bin_counts.split(',')]
    
    total_configs = len(request_counts) * len(gpu_counts) * len(bin_counts)
    print(f"Total configurations: {total_configs}")
    
    results = []
    config_num = 0
    
    for num_requests in request_counts:
        print(f"\n--- {num_requests:,} Requests ---")
        
        for num_gpus in gpu_counts:
            for k_bins in bin_counts:
                config_num += 1
                print(f"[{config_num}/{total_configs}] GPUs={num_gpus}, K={k_bins}", end=" ")
                
                result = run_single_simulation(
                    num_requests=num_requests,
                    num_gpus=num_gpus,
                    k_bins=k_bins,
                    scheduler_type="multi_bin_dynamic",
                    verbose=True
                )
                results.append(result)
                
                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False)
    
    print(f"\n[COMPLETE] Results saved to {args.output}")
    return results


def run_step2_comparison(args):
    """Step 2: Method comparison with dual SLA tracking.
    
    Expected differentiation with production SLA thresholds:
    - Static FIFO: Highest SLA violations (no batching optimization, poor GPU use)
    - Dynamic No-Bins: Moderate violations (batching helps, but no output-length grouping)
    - Multi-Bin Dynamic: Lowest violations (grouping similar outputs reduces variance)
    
    The 3ms per-token SLA creates meaningful differentiation because:
    - Inefficient scheduling causes queueing delays → higher TBT variance
    - Multi-bin grouping reduces output variance within batches → more predictable TBT
    """
    
    print("\n" + "="*80)
    print("STEP 2: METHOD COMPARISON (v2 - Production SLA)")
    print("="*80)
    print(f"SLA Thresholds: Token={D_SLA_TOKEN*1000:.0f}ms, Request={D_SLA_REQUEST:.1f}s")
    print("\nExpected Differentiation:")
    print("  - Static FIFO: HIGH violations (no batching, queue delays)")
    print("  - Dynamic No-Bins: MODERATE violations (batching, but high variance)")  
    print("  - Multi-Bin Dynamic: LOW violations (grouping reduces variance)")
    
    request_counts = [int(x) for x in args.request_counts.split(',')]
    
    # Methods to compare - comprehensive list
    methods = [
        ("static_fifo", 1, 1, "Static_FIFO_1GPU"),
        ("dynamic_no_bins", 1, 1, "Dynamic_NoBins_1GPU"),
        ("multi_bin_dynamic", 1, 4, "MultiBin_Dynamic_1GPU_K4"),
        ("multi_bin_dynamic", 1, 8, "MultiBin_Dynamic_1GPU_K8"),
    ]
    
    # Add optimal config if provided
    if hasattr(args, 'optimal_gpus') and args.optimal_gpus:
        methods.append(("multi_bin_dynamic", args.optimal_gpus, args.optimal_bins, 
                       f"MultiBin_Optimal_{args.optimal_gpus}GPU_K{args.optimal_bins}"))
    else:
        # Default: add multi-GPU configs for comparison
        methods.extend([
            ("dynamic_no_bins", 4, 1, "Dynamic_NoBins_4GPU"),
            ("multi_bin_dynamic", 4, 4, "MultiBin_Dynamic_4GPU_K4"),
            ("multi_bin_dynamic", 8, 4, "MultiBin_Dynamic_8GPU_K4"),
        ])
    
    results = []
    
    for num_requests in request_counts:
        print(f"\n--- {num_requests:,} Requests ---")
        
        for scheduler_type, num_gpus, k_bins, label in methods:
            print(f"  {label}:", end=" ")
            
            result = run_single_simulation(
                num_requests=num_requests,
                num_gpus=num_gpus,
                k_bins=k_bins,
                scheduler_type=scheduler_type,
                verbose=True
            )
            result['method_label'] = label
            results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n[COMPLETE] Results saved to {args.output}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Optimized Stress Testing v2')
    parser.add_argument('--step', type=int, choices=[1, 2], required=True)
    parser.add_argument('--request-counts', type=str, default='1000,10000')
    parser.add_argument('--gpu-counts', type=str, default='1,2,4')
    parser.add_argument('--bin-counts', type=str, default='1,2,4')
    parser.add_argument('--output', type=str, default='results_v2.csv')
    parser.add_argument('--optimal-gpus', type=int, default=None)
    parser.add_argument('--optimal-bins', type=int, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("STRESS TESTING v2 - DUAL SLA TRACKING")
    print("="*80)
    print(f"Token SLA: {D_SLA_TOKEN*1000:.0f}ms | Request SLA: {D_SLA_REQUEST:.1f}s")
    
    if args.step == 1:
        run_step1_grid_search(args)
    else:
        run_step2_comparison(args)


if __name__ == "__main__":
    main()
