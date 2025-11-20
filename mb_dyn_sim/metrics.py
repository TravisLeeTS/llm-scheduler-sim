"""
Metrics computation and analysis.
"""

import numpy as np
from typing import List, Dict
from .workload import Request


def compute_metrics(requests: List[Request]) -> Dict:
    """
    Compute comprehensive metrics for completed requests.
    
    Args:
        requests: List of completed requests with timing information
    
    Returns:
        Dictionary of metrics including throughput, latency, SLA violations, etc.
    """
    if not requests:
        return {
            'num_requests': 0,
            'throughput_tokens_per_sec': 0.0,
            'throughput_requests_per_sec': 0.0,
            'avg_latency': 0.0,
            'p50_latency': 0.0,
            'p95_latency': 0.0,
            'p99_latency': 0.0,
            'max_latency': 0.0,
            'sla_violation_rate': 0.0,
            'avg_queueing_delay': 0.0,
            'avg_service_time': 0.0,
            'total_tokens': 0,
        }
    
    # Filter only completed requests
    completed = [r for r in requests if r.completion_time >= 0]
    
    if not completed:
        return compute_metrics([])  # Return empty metrics
    
    # Calculate basic stats
    num_requests = len(completed)
    total_tokens = sum(r.prompt_len + r.output_len for r in completed)
    
    # Time span
    min_arrival = min(r.arrival_time for r in completed)
    max_completion = max(r.completion_time for r in completed)
    total_time = max_completion - min_arrival
    
    # Throughput
    throughput_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0
    throughput_requests_per_sec = num_requests / total_time if total_time > 0 else 0.0
    
    # Latency statistics
    latencies = [r.latency for r in completed]
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    max_latency = np.max(latencies)
    
    # SLA violations
    violations = sum(1 for r in completed if r.violates_sla)
    sla_violation_rate = violations / num_requests
    
    # Queueing and service time
    queueing_delays = [r.queueing_delay for r in completed if r.queueing_delay >= 0]
    service_times = [r.service_time for r in completed if r.service_time >= 0]
    
    avg_queueing_delay = np.mean(queueing_delays) if queueing_delays else 0.0
    avg_service_time = np.mean(service_times) if service_times else 0.0
    
    return {
        'num_requests': num_requests,
        'throughput_tokens_per_sec': throughput_tokens_per_sec,
        'throughput_requests_per_sec': throughput_requests_per_sec,
        'avg_latency': avg_latency,
        'p50_latency': p50_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency,
        'max_latency': max_latency,
        'sla_violation_rate': sla_violation_rate,
        'avg_queueing_delay': avg_queueing_delay,
        'avg_service_time': avg_service_time,
        'total_tokens': total_tokens,
        'total_time': total_time,
    }


def compute_gpu_utilization(gpu_stats: List[Dict]) -> Dict:
    """
    Compute GPU utilization metrics.
    
    Args:
        gpu_stats: List of per-GPU statistics from simulator
    
    Returns:
        Dictionary with GPU utilization metrics
    """
    if not gpu_stats:
        return {
            'num_gpus': 0,
            'avg_utilization': 0.0,
            'min_utilization': 0.0,
            'max_utilization': 0.0,
        }
    
    utilizations = [g['utilization'] for g in gpu_stats]
    
    return {
        'num_gpus': len(gpu_stats),
        'avg_utilization': np.mean(utilizations),
        'min_utilization': np.min(utilizations),
        'max_utilization': np.max(utilizations),
        'per_gpu_utilization': utilizations,
    }


def print_metrics_table(metrics: Dict, gpu_metrics: Dict, scheduler_name: str) -> None:
    """
    Print a formatted table of metrics.
    
    Args:
        metrics: Request metrics dictionary
        gpu_metrics: GPU utilization metrics dictionary
        scheduler_name: Name of the scheduler for display
    """
    print(f"\n{'='*70}")
    print(f"Scheduler: {scheduler_name}")
    print(f"{'='*70}")
    
    print(f"\nThroughput:")
    print(f"  Requests/sec:  {metrics['throughput_requests_per_sec']:.2f}")
    print(f"  Tokens/sec:    {metrics['throughput_tokens_per_sec']:.2f}")
    
    print(f"\nLatency (seconds):")
    print(f"  Average:       {metrics['avg_latency']:.4f}")
    print(f"  P50:           {metrics['p50_latency']:.4f}")
    print(f"  P95:           {metrics['p95_latency']:.4f}")
    print(f"  P99:           {metrics['p99_latency']:.4f}")
    print(f"  Max:           {metrics['max_latency']:.4f}")
    
    print(f"\nSLA Performance:")
    print(f"  Violation rate: {metrics['sla_violation_rate']*100:.2f}%")
    
    print(f"\nQueue Statistics:")
    print(f"  Avg queueing delay: {metrics['avg_queueing_delay']:.4f}s")
    print(f"  Avg service time:   {metrics['avg_service_time']:.4f}s")
    
    print(f"\nGPU Utilization:")
    print(f"  Num GPUs:      {gpu_metrics['num_gpus']}")
    print(f"  Average:       {gpu_metrics['avg_utilization']*100:.2f}%")
    if gpu_metrics['num_gpus'] > 1:
        print(f"  Min:           {gpu_metrics['min_utilization']*100:.2f}%")
        print(f"  Max:           {gpu_metrics['max_utilization']*100:.2f}%")
    
    print(f"\nWorkload:")
    print(f"  Total requests: {metrics['num_requests']}")
    print(f"  Total tokens:   {metrics['total_tokens']}")
    print(f"  Simulation time: {metrics['total_time']:.2f}s")
    
    print(f"{'='*70}\n")
