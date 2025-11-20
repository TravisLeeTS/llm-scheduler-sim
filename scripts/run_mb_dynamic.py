#!/usr/bin/env python3
"""
Run multi-bin dynamic scheduler experiments (paper-faithful).

This script supports:
- Three experiment modes: multi_bin_only, dynamic_only, multi_bin_dynamic
- Equal-mass bin boundaries (paper requirement)
- BurstGPT dataset loading
- vLLM calibration (optional)
- Poisson and BurstGPT arrival patterns
"""

import argparse
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.config import SchedulerConfig, compute_equal_mass_boundaries
from mb_dyn_sim.workload import generate_workload
from mb_dyn_sim.experiments import run_experiment, compare_schedulers, plot_comparison, plot_k_bins_sensitivity
from mb_dyn_sim.metrics import print_metrics_table


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM inference scheduling experiments"
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to simulate (default: 1)'
    )
    
    parser.add_argument(
        '--k-bins',
        type=int,
        default=4,
        help='Number of bins for multi-bin scheduler (default: 4)'
    )
    
    parser.add_argument(
        '--scheduler',
        type=str,
        choices=['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic', 'multi_bin_only', 'all'],
        default='all',
        help='Scheduler type to use (default: all)'
    )
    
    parser.add_argument(
        '--experiment-mode',
        type=str,
        choices=['multi_bin_only', 'dynamic_only', 'multi_bin_dynamic'],
        default=None,
        help='Experiment mode (paper-faithful): multi_bin_only, dynamic_only, or multi_bin_dynamic'
    )
    
    parser.add_argument(
        '--arrival-profile',
        type=str,
        choices=['poisson', 'burstgpt_like', 'burstgpt_dataset'],
        default='burstgpt_like',
        help='Arrival pattern (default: burstgpt_like)'
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='',
        help='Path to BurstGPT dataset CSV file'
    )
    
    parser.add_argument(
        '--use-vllm',
        action='store_true',
        help='Use vLLM calibration for realistic latency (requires vLLM installed)'
    )
    
    parser.add_argument(
        '--use-real-calibration',
        action='store_true',
        help='Use real GPU calibration data (requires calibration CSV)'
    )
    
    parser.add_argument(
        '--calibration-csv',
        type=str,
        default='data/qwen3_1_7b_latency_grid.csv',
        help='Path to GPU calibration CSV (default: data/qwen3_1_7b_latency_grid.csv)'
    )
    
    parser.add_argument(
        '--rps-scaling',
        type=float,
        default=1.0,
        help='RPS scaling factor for BurstGPT dataset (default: 1.0)'
    )
    
    parser.add_argument(
        '--use-equal-mass-bins',
        action='store_true',
        default=True,
        help='Use equal-mass bin boundaries (paper requirement, default: True)'
    )
    
    parser.add_argument(
        '--poisson-lambda',
        type=float,
        default=50.0,
        help='Poisson arrival rate (requests/second) for poisson mode (default: 50.0)'
    )
    
    parser.add_argument(
        '--b-fixed',
        type=int,
        default=32,
        help='Fixed batch size for multi_bin_only mode (default: 32)'
    )
    
    parser.add_argument(
        '--load-level',
        type=str,
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Load level (default: medium)'
    )
    
    parser.add_argument(
        '--num-requests',
        type=int,
        default=10000,
        help='Number of requests to simulate (default: 10000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--d-sla',
        type=float,
        default=1.0,
        help='SLA deadline in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all schedulers and generate plots'
    )
    
    parser.add_argument(
        '--k-bins-sensitivity',
        action='store_true',
        help='Test sensitivity to K_BINS parameter'
    )
    
    args = parser.parse_args()
    
    # Set vLLM environment variable if requested
    if args.use_vllm:
        os.environ['USE_VLLM'] = 'true'
    
    # Determine workload source
    workload_source = "synthetic"
    if args.arrival_profile == "burstgpt_dataset":
        workload_source = "burstgpt_dataset"
        if not args.dataset_path:
            print("ERROR: --dataset-path required when using burstgpt_dataset")
            return
    
    # Create configuration
    cfg = SchedulerConfig(
        NUM_GPUS=args.num_gpus,
        K_BINS=args.k_bins,
        NUM_REQUESTS=args.num_requests,
        SEED=args.seed,
        D_SLA=args.d_sla,
        ARRIVAL_PROFILE=args.arrival_profile,
        POISSON_LAMBDA=args.poisson_lambda,
        DATASET_PATH=args.dataset_path,
        WORKLOAD_SOURCE=workload_source,
        RPS_SCALING=args.rps_scaling,
        USE_EQUAL_MASS_BINS=args.use_equal_mass_bins,
        EXPERIMENT_MODE=args.experiment_mode or "multi_bin_dynamic",
        B_FIXED=args.b_fixed,
        USE_REAL_MODEL=args.use_vllm,
        USE_REAL_CALIBRATION=args.use_real_calibration,
        CALIBRATION_CSV_PATH=args.calibration_csv if args.use_real_calibration else "",
    )
    
    # Compute equal-mass bin boundaries if enabled
    if cfg.USE_EQUAL_MASS_BINS and cfg.K_BINS > 1:
        print("Computing equal-mass bin boundaries (paper requirement)...")
        # Generate small sample to estimate boundaries
        sample_cfg = SchedulerConfig(
            NUM_REQUESTS=min(5000, cfg.NUM_REQUESTS),
            SEED=cfg.SEED,
            ARRIVAL_PROFILE=cfg.ARRIVAL_PROFILE,
            POISSON_LAMBDA=cfg.POISSON_LAMBDA,
            DATASET_PATH=cfg.DATASET_PATH,
            WORKLOAD_SOURCE=cfg.WORKLOAD_SOURCE,
        )
        sample_requests = generate_workload(sample_cfg)
        predicted_lengths = [r.predicted_output_len for r in sample_requests]
        cfg.BIN_BOUNDARIES = compute_equal_mass_boundaries(predicted_lengths, cfg.K_BINS)
        print(f"Equal-mass boundaries: {cfg.BIN_BOUNDARIES}")
    
    print("="*70)
    print("Multi-Bin Dynamic Scheduler Simulation (Paper-Faithful)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Experiment mode:   {cfg.EXPERIMENT_MODE}")
    print(f"  NUM_GPUS:          {cfg.NUM_GPUS}")
    print(f"  K_BINS:            {cfg.K_BINS}")
    print(f"  NUM_REQUESTS:      {cfg.NUM_REQUESTS}")
    print(f"  D_SLA:             {cfg.D_SLA}s")
    print(f"  Arrival profile:   {cfg.ARRIVAL_PROFILE}")
    if cfg.ARRIVAL_PROFILE == "poisson":
        print(f"  Poisson λ:         {cfg.POISSON_LAMBDA} req/s")
    if cfg.WORKLOAD_SOURCE == "burstgpt_dataset":
        print(f"  Dataset:           {cfg.DATASET_PATH}")
    print(f"  Equal-mass bins:   {cfg.USE_EQUAL_MASS_BINS}")
    if cfg.EXPERIMENT_MODE == "multi_bin_only":
        print(f"  Fixed batch size:  {cfg.B_FIXED}")
    print(f"  Use vLLM:          {args.use_vllm}")
    print(f"  Load level:        {args.load_level}")
    print()
    
    if args.k_bins_sensitivity:
        # Run K_BINS sensitivity analysis
        print("\nRunning K_BINS sensitivity analysis...")
        plot_k_bins_sensitivity(cfg, k_bins_values=[1, 2, 4, 8])
        
    elif args.compare or args.scheduler == 'all':
        # Compare all schedulers
        print("\nComparing all schedulers...")
        scheduler_types = ['static_fifo', 'dynamic_no_bins', 'multi_bin_dynamic']
        
        df = compare_schedulers(cfg, scheduler_types, args.load_level)
        
        # Print comparison table
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(df.to_string(index=False))
        print()
        
        # Generate plots
        if args.compare:
            plot_comparison(df, output_dir="mb_dyn_sim/plots")
    
    else:
        # Run single scheduler
        result = run_experiment(cfg, args.scheduler, args.load_level)
        
        # Print metrics
        print_metrics_table(
            result['metrics'],
            result['gpu_metrics'],
            args.scheduler
        )


if __name__ == '__main__':
    main()
