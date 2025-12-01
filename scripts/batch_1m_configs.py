#!/usr/bin/env python3
"""
Batch run all missing 1M configurations, one at a time.
Uses multiprocessing to isolate each run from interrupts.
"""

import os
import sys
import time
import signal
from datetime import datetime
import pandas as pd
import numpy as np
import traceback
from multiprocessing import Process, Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'stress_test_v3_results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'step1_grid_search.csv')
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'BurstGPT_sample.csv')
CALIBRATION_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'qwen3_1_7b_latency_grid.csv')

D_SLA_TOKEN = 0.030
D_SLA_REQUEST = 20.0
RPS_SCALING = 200.0
NUM_REQUESTS = 1_000_000


def ignore_interrupt(signum, frame):
    """Ignore SIGINT in worker process."""
    pass


def worker_run_experiment(num_gpus, k_bins, result_queue):
    """Worker function to run in subprocess."""
    # Ignore interrupts
    signal.signal(signal.SIGINT, ignore_interrupt)
    
    from mb_dyn_sim.config import SchedulerConfig
    from mb_dyn_sim.workload import generate_workload
    from mb_dyn_sim.simulation import Simulator
    from mb_dyn_sim.metrics import compute_metrics, compute_gpu_utilization, compute_batch_statistics
    
    start_time = time.time()
    
    try:
        cfg = SchedulerConfig(
            NUM_GPUS=num_gpus,
            K_BINS=k_bins,
            NUM_REQUESTS=NUM_REQUESTS,
            EXPERIMENT_MODE="multi_bin_dynamic",
            B_MIN=1,
            B_MAX=128,
            D_SLA=D_SLA_TOKEN,
            D_SLA_TOKEN=D_SLA_TOKEN,
            D_SLA_REQUEST=D_SLA_REQUEST,
            LATENCY_EPSILON=0.010,
            USE_REAL_CALIBRATION=True,
            CALIBRATION_CSV_PATH=CALIBRATION_CSV,
            WORKLOAD_SOURCE="burstgpt_dataset",
            DATASET_PATH=DATASET_PATH,
            USE_REAL_TIMESTAMPS=False,
            RPS_SCALING=RPS_SCALING,
            SEED=42,
        )
        
        print(f"Loading {NUM_REQUESTS} requests...", flush=True)
        requests = generate_workload(cfg)
        print(f"Running simulation with {num_gpus} GPUs, K={k_bins}...", flush=True)
        
        simulator = Simulator(cfg, requests, "multi_bin_dynamic")
        completed_requests = simulator.run()
        
        metrics = compute_metrics(
            completed_requests, 
            d_sla_token=D_SLA_TOKEN,
            d_sla_request=D_SLA_REQUEST
        )
        
        gpu_stats = simulator.get_gpu_stats()
        gpu_util = compute_gpu_utilization(gpu_stats)
        batch_stats = compute_batch_statistics(completed_requests)
        
        execution_time = time.time() - start_time
        actual_rps = NUM_REQUESTS / metrics.get('total_time', 1) if metrics.get('total_time', 0) > 0 else 0
        
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_requests': NUM_REQUESTS,
            'num_gpus': num_gpus,
            'k_bins': k_bins,
            'scheduler': "multi_bin_dynamic",
            'token_sla_pct': round(metrics.get('token_sla_pct', 0), 3),
            'request_sla_pct': round(metrics.get('request_sla_pct', 0), 3),
            'avg_batch_size': round(batch_stats.get('avg_batch_size', 0), 1),
            'gpu_utilization': round(gpu_util * 100, 1),
            'rps_scaling': RPS_SCALING,
            'actual_rps': round(actual_rps, 1),
            'execution_time': round(execution_time, 1),
            'completed_requests': len(completed_requests),
            'avg_latency_ms': round(metrics.get('avg_latency', 0) * 1000, 2),
            'p50_latency_ms': round(metrics.get('p50_latency', 0) * 1000, 2),
            'p99_latency_ms': round(metrics.get('p99_latency', 0) * 1000, 2),
        }
        
        result_queue.put(('success', result))
        
    except Exception as e:
        result_queue.put(('error', str(e) + '\n' + traceback.format_exc()))


def run_experiment_in_subprocess(num_gpus, k_bins, timeout=600):
    """Run experiment in a subprocess with timeout."""
    result_queue = Queue()
    
    p = Process(target=worker_run_experiment, args=(num_gpus, k_bins, result_queue))
    p.start()
    p.join(timeout=timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return None, "Timeout"
    
    if result_queue.empty():
        return None, "No result returned"
    
    status, data = result_queue.get()
    if status == 'success':
        return data, None
    else:
        return None, data


def save_result(result):
    """Append result to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    df_new = pd.DataFrame([result])
    
    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_csv(RESULTS_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(RESULTS_FILE, index=False)
    return len(df_combined)


def get_missing_configs():
    """Find all missing 1M configurations."""
    gpu_counts = [1, 2, 4, 8, 16, 32, 64, 100]
    bin_counts = [1, 2, 4, 8, 16, 32]
    
    missing = []
    
    if not os.path.exists(RESULTS_FILE):
        for g in gpu_counts:
            for k in bin_counts:
                missing.append((g, k))
        return missing
    
    df = pd.read_csv(RESULTS_FILE)
    existing_1m = df[df['num_requests'] == NUM_REQUESTS][['num_gpus', 'k_bins']].values.tolist()
    existing_set = set([(int(r[0]), int(r[1])) for r in existing_1m])
    
    for g in gpu_counts:
        for k in bin_counts:
            if (g, k) not in existing_set:
                missing.append((g, k))
    
    return missing


def main():
    print("="*70)
    print("BATCH 1M CONFIGURATION RUNNER")
    print("="*70)
    
    missing = get_missing_configs()
    total = len(missing)
    
    if total == 0:
        print("\n✓ All 1M configurations complete!")
        return
    
    print(f"\nMissing configurations: {total}")
    print(f"Estimated time: {total * 90 / 60:.0f} - {total * 120 / 60:.0f} minutes")
    print()
    
    completed = 0
    failed = 0
    
    for i, (num_gpus, k_bins) in enumerate(missing):
        print(f"\n[{i+1}/{total}] Running: GPUs={num_gpus}, K={k_bins}")
        print("-" * 50)
        
        result, error = run_experiment_in_subprocess(num_gpus, k_bins, timeout=600)
        
        if result:
            count = save_result(result)
            completed += 1
            print(f"[OK] Token={result['token_sla_pct']:.1f}%, Req={result['request_sla_pct']:.1f}%, "
                  f"Batch={result['avg_batch_size']:.1f}, Time={result['execution_time']:.0f}s")
            print(f"  Total saved: {count}/192")
        else:
            failed += 1
            print(f"[FAILED] {error[:100] if error else 'Unknown error'}")
        
        # Small delay between runs
        time.sleep(1)
    
    print("\n" + "="*70)
    print(f"COMPLETED: {completed}/{total}")
    if failed > 0:
        print(f"FAILED: {failed}")
    print("="*70)


if __name__ == "__main__":
    main()
