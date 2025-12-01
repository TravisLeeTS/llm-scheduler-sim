"""
Metrics computation and analysis.

Paper-specific metrics:
- Multi-Bin Batching Paper: tokens_per_second, capacity_threshold_lambda, throughput_improvement
- Dynamic Batching Paper: decode_step_time, request_rate_qps, capacity_qps_under_SLA

Dual SLA Definition (production-grade + paper-faithful):

1. Per-Request SLA (D_SLA_REQUEST = 150ms):
   - Total request latency from arrival to completion
   - Targets interactive use cases (user perceives response time)
   - Industry standard: 100-200ms for "snappy" feel
   - Reference: Gemini 2.5 Flash-Lite achieves ~240ms TTFT

2. Per-Token SLA (D_SLA_TOKEN = 5ms):
   - Time Between Tokens (TBT) for streaming output
   - Targets smooth streaming experience (200 tokens/sec)
   - Reference: Gemini 2.5 Flash-Lite achieves 410 tokens/sec (~2.44ms/token)
   - Paper definition: SLA-constrained dynamic batching

Both metrics are tracked independently to support different use cases:
- Per-request: Critical for non-streaming, batch processing
- Per-token: Critical for streaming UX, reading speed matching
"""

import numpy as np
from typing import List, Dict, Optional
from .workload import Request


def compute_metrics(requests: List[Request], d_sla_token: float = 0.005, d_sla_request: float = 0.150) -> Dict:
    """
    Compute comprehensive metrics for completed requests.
    
    Dual SLA evaluation:
    1. Per-token SLA (TBT): Based on per-token decode latency
       - D_SLA_TOKEN = 5ms (200 tokens/sec target)
       - Gemini 2.5 Flash-Lite reference: 2.44ms/token (410 tokens/sec)
    
    2. Per-request SLA: Based on total request latency (arrival to completion)
       - D_SLA_REQUEST = 150ms (interactive response target)
       - Gemini 2.5 Flash-Lite reference: 240ms TTFT
    
    Args:
        requests: List of completed requests with timing information
        d_sla_token: D_SLA_TOKEN threshold for per-token TBT (default 5ms)
        d_sla_request: D_SLA_REQUEST threshold for total latency (default 150ms)
    
    Returns:
        Dictionary of metrics including throughput, latency, dual SLA violations, etc.
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
    
    # Per-token TBT (Time Between Tokens) statistics - legacy (includes TTFT)
    # TBT = service_time / max_output_len_in_batch (computed in simulation)
    tbt_values = [r.per_token_tbt for r in completed if hasattr(r, 'per_token_tbt') and r.per_token_tbt >= 0]
    if tbt_values:
        avg_tbt = np.mean(tbt_values)
        p50_tbt = np.percentile(tbt_values, 50)
        p95_tbt = np.percentile(tbt_values, 95)
        p99_tbt = np.percentile(tbt_values, 99)
        max_tbt = np.max(tbt_values)
    else:
        avg_tbt = p50_tbt = p95_tbt = p99_tbt = max_tbt = 0.0
    
    # ===== v2 SLA Model: TTFT and Decode TBT Separation =====
    # TTFT = Time To First Token (prefill latency, α ≈ 60ms)
    # Decode TBT = Per-token decode time (β * h(b) ≈ 5.74ms)
    
    ttft_values = [r.ttft for r in completed if hasattr(r, 'ttft') and r.ttft >= 0]
    if ttft_values:
        avg_ttft = np.mean(ttft_values)
        p50_ttft = np.percentile(ttft_values, 50)
        p99_ttft = np.percentile(ttft_values, 99)
    else:
        avg_ttft = p50_ttft = p99_ttft = 0.0
    
    decode_tbt_values = [r.decode_tbt for r in completed if hasattr(r, 'decode_tbt') and r.decode_tbt >= 0]
    if decode_tbt_values:
        avg_decode_tbt = np.mean(decode_tbt_values)
        p50_decode_tbt = np.percentile(decode_tbt_values, 50)
        p99_decode_tbt = np.percentile(decode_tbt_values, 99)
    else:
        avg_decode_tbt = p50_decode_tbt = p99_decode_tbt = 0.0
    
    # ===== DUAL SLA VIOLATIONS =====
    
    # 1. Per-token SLA violations (v2: uses decode TBT only, not TTFT)
    # A request violates per-token SLA if decode_tbt exceeds D_SLA_TOKEN
    violations_token = sum(1 for r in completed if r.violates_sla)
    sla_violation_rate_token = violations_token / num_requests
    
    # 2. Per-request SLA violations (production-grade definition)
    # A request violates per-request SLA if total latency exceeds D_SLA_REQUEST
    violations_request = sum(1 for r in completed if r.violates_sla_request)
    sla_violation_rate_request = violations_request / num_requests
    
    # Legacy alias (backward compatibility - uses per-token)
    violations = violations_token
    sla_violation_rate = sla_violation_rate_token
    
    # Queueing and service time
    queueing_delays = [r.queueing_delay for r in completed if r.queueing_delay >= 0]
    service_times = [r.service_time for r in completed if r.service_time >= 0]
    
    avg_queueing_delay = np.mean(queueing_delays) if queueing_delays else 0.0
    avg_service_time = np.mean(service_times) if service_times else 0.0
    
    # Calculate additional paper-specific metrics
    total_output_tokens = sum(r.output_len for r in completed)
    avg_output_tokens = total_output_tokens / num_requests if num_requests > 0 else 0.0
    
    # Seconds per generated token (Dynamic Batching paper metric)
    seconds_per_generated_token = total_time / total_output_tokens if total_output_tokens > 0 else 0.0
    
    # Decode step time (average service time per token in batch)
    # Approximation: avg_service_time / avg_output_tokens
    decode_step_time_ms = (avg_service_time / avg_output_tokens * 1000) if avg_output_tokens > 0 else 0.0
    
    # Request rate (QPS) = requests per second
    request_rate_qps = throughput_requests_per_sec
    
    # Capacity under SLA: requests/sec that meet SLA
    requests_meeting_sla = num_requests - violations
    capacity_qps_under_sla = requests_meeting_sla / total_time if total_time > 0 else 0.0
    
    # Capacity threshold lambda (Multi-Bin paper metric)
    # Maximum sustainable arrival rate before SLA violations exceed threshold
    # Approximation: current request rate adjusted by SLA violation headroom
    sla_headroom = max(0.05 - sla_violation_rate, 0.0)  # Assume 5% violation threshold
    capacity_threshold_lambda = throughput_requests_per_sec * (1 + sla_headroom / 0.05) if sla_violation_rate < 0.05 else throughput_requests_per_sec
    
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
        
        # Multi-Bin Batching Paper Metrics
        'tokens_per_second': throughput_tokens_per_sec,  # Alias for clarity
        'requests_per_second': throughput_requests_per_sec,  # Alias for clarity
        'average_latency_seconds': avg_latency,  # Alias for clarity
        'capacity_threshold_lambda': capacity_threshold_lambda,
        'seconds_per_generated_token': seconds_per_generated_token,
        
        # Dynamic Batching Paper Metrics
        'decode_step_time_ms': decode_step_time_ms,
        'request_rate_qps': request_rate_qps,
        'capacity_qps_under_sla': capacity_qps_under_sla,
        'total_output_tokens': total_output_tokens,
        'avg_output_tokens_per_request': avg_output_tokens,
        
        # Per-token TBT (legacy - includes TTFT)
        'avg_tbt': avg_tbt,           # Average per-token decode latency
        'p50_tbt': p50_tbt,           # Median per-token decode latency
        'p95_tbt': p95_tbt,           # 95th percentile TBT
        'p99_tbt': p99_tbt,           # 99th percentile TBT
        'max_tbt': max_tbt,           # Maximum TBT observed
        'tbt_sla_threshold': d_sla_token,   # D_SLA_TOKEN threshold used
        
        # ===== v2 SLA Model: TTFT/TBT Separation =====
        # TTFT = Time To First Token (prefill, α ≈ 60ms)
        'avg_ttft': avg_ttft,
        'p50_ttft': p50_ttft,
        'p99_ttft': p99_ttft,
        
        # Decode TBT = Per-token decode time (β * h(b) ≈ 5.74ms)
        'avg_decode_tbt': avg_decode_tbt,
        'p50_decode_tbt': p50_decode_tbt,
        'p99_decode_tbt': p99_decode_tbt,
        
        # ===== DUAL SLA METRICS (v2) =====
        # Per-token SLA (D_SLA_TOKEN, for streaming UX) - uses decode_tbt only
        'sla_violation_rate_token': sla_violation_rate_token,  # % violating per-token SLA
        'sla_violations_token': violations_token,              # Count of per-token violations
        'd_sla_token': d_sla_token,                            # Per-token threshold
        
        # Per-request SLA (D_SLA_REQUEST, for interactive response)
        'sla_violation_rate_request': sla_violation_rate_request,  # % violating per-request SLA
        'sla_violations_request': violations_request,               # Count of per-request violations
        'd_sla_request': d_sla_request,                             # Per-request threshold
        
        # Token SLA semantic: decode_tbt vs d_sla_token (not total TBT)
        # This eliminates structural violations where TTFT/L dominates
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


def compute_comparative_metrics(
    test_metrics: Dict,
    baseline_metrics: Dict,
) -> Dict:
    """
    Compute improvement metrics vs baseline (paper-specific).
    
    For Multi-Bin Paper:
    - throughput_improvement_vs_baseline_percent
    
    For Dynamic Batching Paper:
    - throughput_improvement_percent_vs_static
    - capacity_improvement_percent_vs_static
    
    Args:
        test_metrics: Metrics from test scheduler (e.g., multi_bin_dynamic or dynamic_no_bins)
        baseline_metrics: Metrics from baseline scheduler (e.g., static_fifo)
    
    Returns:
        Dictionary with improvement percentages
    """
    improvements = {}
    
    # Throughput improvement
    if baseline_metrics['throughput_tokens_per_sec'] > 0:
        throughput_improvement = (
            (test_metrics['throughput_tokens_per_sec'] - baseline_metrics['throughput_tokens_per_sec']) 
            / baseline_metrics['throughput_tokens_per_sec'] * 100
        )
        improvements['throughput_improvement_vs_baseline_percent'] = throughput_improvement
        improvements['throughput_improvement_percent_vs_static'] = throughput_improvement  # Alias
    
    # Capacity improvement (requests meeting SLA)
    if baseline_metrics['capacity_qps_under_sla'] > 0:
        capacity_improvement = (
            (test_metrics['capacity_qps_under_sla'] - baseline_metrics['capacity_qps_under_sla'])
            / baseline_metrics['capacity_qps_under_sla'] * 100
        )
        improvements['capacity_improvement_percent_vs_static'] = capacity_improvement
    
    # Latency improvement (lower is better, so invert)
    if baseline_metrics['avg_latency'] > 0:
        latency_improvement = (
            (baseline_metrics['avg_latency'] - test_metrics['avg_latency'])
            / baseline_metrics['avg_latency'] * 100
        )
        improvements['latency_improvement_percent'] = latency_improvement
    
    # SLA violation reduction (lower is better)
    if baseline_metrics['sla_violation_rate'] > 0:
        sla_reduction = (
            (baseline_metrics['sla_violation_rate'] - test_metrics['sla_violation_rate'])
            / baseline_metrics['sla_violation_rate'] * 100
        )
        improvements['sla_violation_reduction_percent'] = sla_reduction
    
    return improvements


def estimate_memory_usage(
    batch_size: int,
    avg_sequence_length: int,
    kv_cache_per_token_gb: float = 5e-6,
    model_size_gb: float = 2.0,
) -> Dict:
    """
    Estimate memory usage for a batch (Dynamic Batching paper metric).
    
    Args:
        batch_size: Number of requests in batch
        avg_sequence_length: Average total sequence length (prompt + output)
        kv_cache_per_token_gb: KV cache memory per token in GB
        model_size_gb: Model size in GB
    
    Returns:
        Dictionary with memory estimates
    """
    # KV cache memory for batch
    total_tokens = batch_size * avg_sequence_length
    kv_cache_gb = total_tokens * kv_cache_per_token_gb
    
    # Total memory
    total_memory_gb = model_size_gb + kv_cache_gb
    
    return {
        'batch_size': batch_size,
        'total_tokens_in_batch': total_tokens,
        'kv_cache_memory_gb': kv_cache_gb,
        'model_memory_gb': model_size_gb,
        'total_memory_gb': total_memory_gb,
        'memory_tokens_used': total_tokens,  # For paper metric naming
    }


def compute_batch_statistics(requests: List[Request]) -> Dict:
    """
    Compute batch-level statistics from completed requests.
    
    This extracts information about how requests were batched together,
    useful for analyzing batch composition efficiency.
    
    Args:
        requests: List of completed requests
    
    Returns:
        Dictionary with batch statistics
    """
    if not requests:
        return {
            'num_batches': 0,
            'avg_batch_size': 0.0,
            'min_batch_size': 0,
            'max_batch_size': 0,
            'std_batch_size': 0.0,
        }
    
    # Group requests by their service start time (same time = same batch)
    from collections import defaultdict
    batches = defaultdict(list)
    
    for req in requests:
        if req.start_service_time >= 0:
            # Use GPU and start time to uniquely identify batch
            batch_key = (req.assigned_gpu, round(req.start_service_time, 6))
            batches[batch_key].append(req)
    
    batch_sizes = [len(batch) for batch in batches.values()]
    
    if not batch_sizes:
        return {
            'num_batches': 0,
            'avg_batch_size': 0.0,
            'min_batch_size': 0,
            'max_batch_size': 0,
            'std_batch_size': 0.0,
        }
    
    return {
        'num_batches': len(batch_sizes),
        'avg_batch_size': np.mean(batch_sizes),
        'min_batch_size': np.min(batch_sizes),
        'max_batch_size': np.max(batch_sizes),
        'std_batch_size': np.std(batch_sizes),
    }


def print_metrics_table(
    metrics: Dict, 
    gpu_metrics: Dict, 
    scheduler_name: str,
    baseline_metrics: Optional[Dict] = None,
    show_paper_metrics: bool = True,
) -> None:
    """
    Print a formatted table of metrics with paper-specific metrics.
    
    Args:
        metrics: Request metrics dictionary
        gpu_metrics: GPU utilization metrics dictionary
        scheduler_name: Name of the scheduler for display
        baseline_metrics: Optional baseline metrics for computing improvements
        show_paper_metrics: Whether to show paper-specific metrics
    """
    print(f"\n{'='*80}")
    print(f"Scheduler: {scheduler_name}")
    print(f"{'='*80}")
    
    print(f"\n[Core Performance Metrics]")
    print(f"  Requests/sec:        {metrics['throughput_requests_per_sec']:.2f}")
    print(f"  Tokens/sec:          {metrics['throughput_tokens_per_sec']:.2f}")
    
    print(f"\n[Latency (seconds)]")
    print(f"  Average:             {metrics['avg_latency']:.4f}")
    print(f"  P50:                 {metrics['p50_latency']:.4f}")
    print(f"  P95:                 {metrics['p95_latency']:.4f}")
    print(f"  P99:                 {metrics['p99_latency']:.4f}")
    print(f"  Max:                 {metrics['max_latency']:.4f}")
    
    print(f"\n[SLA Performance - Dual Mode]")
    # Per-token SLA (streaming UX)
    d_sla_token = metrics.get('d_sla_token', 0.005)
    sla_viol_token = metrics.get('sla_violation_rate_token', metrics['sla_violation_rate'])
    print(f"  Per-Token SLA (D={d_sla_token*1000:.1f}ms):")
    print(f"    Violation rate:    {sla_viol_token*100:.2f}%")
    print(f"    Requests meeting:  {metrics['num_requests'] - int(metrics['num_requests'] * sla_viol_token)}")
    print(f"    Avg TBT:           {metrics.get('avg_tbt', 0)*1000:.2f}ms")
    print(f"    P95 TBT:           {metrics.get('p95_tbt', 0)*1000:.2f}ms")
    
    # Per-request SLA (interactive response)
    d_sla_request = metrics.get('d_sla_request', 0.150)
    sla_viol_request = metrics.get('sla_violation_rate_request', 0)
    print(f"  Per-Request SLA (D={d_sla_request*1000:.0f}ms):")
    print(f"    Violation rate:    {sla_viol_request*100:.2f}%")
    print(f"    Requests meeting:  {metrics['num_requests'] - int(metrics['num_requests'] * sla_viol_request)}")
    print(f"    Avg Latency:       {metrics['avg_latency']*1000:.2f}ms")
    print(f"    P95 Latency:       {metrics['p95_latency']*1000:.2f}ms")
    
    print(f"\n[Queue Statistics]")
    print(f"  Avg queueing delay:  {metrics['avg_queueing_delay']:.4f}s")
    print(f"  Avg service time:    {metrics['avg_service_time']:.4f}s")
    
    print(f"\n[GPU Utilization]")
    print(f"  Num GPUs:            {gpu_metrics['num_gpus']}")
    print(f"  Average:             {gpu_metrics['avg_utilization']*100:.2f}%")
    if gpu_metrics['num_gpus'] > 1:
        print(f"  Min:                 {gpu_metrics['min_utilization']*100:.2f}%")
        print(f"  Max:                 {gpu_metrics['max_utilization']*100:.2f}%")
    
    print(f"\n[Workload]")
    print(f"  Total requests:      {metrics['num_requests']}")
    print(f"  Total tokens:        {metrics['total_tokens']:,}")
    print(f"  Output tokens:       {metrics['total_output_tokens']:,}")
    print(f"  Simulation time:     {metrics['total_time']:.2f}s")
    
    if show_paper_metrics:
        print(f"\n[Multi-Bin Batching Paper Metrics]")
        print(f"  Tokens/second:                   {metrics['tokens_per_second']:.2f}")
        print(f"  Requests/second:                 {metrics['requests_per_second']:.2f}")
        print(f"  Average latency (seconds):       {metrics['average_latency_seconds']:.4f}")
        print(f"  Capacity threshold lambda:       {metrics['capacity_threshold_lambda']:.2f} req/s")
        print(f"  Seconds per generated token:     {metrics['seconds_per_generated_token']:.6f}")
        
        print(f"\n[Dynamic Batching Paper Metrics]")
        print(f"  Decode step time:                {metrics['decode_step_time_ms']:.2f} ms")
        print(f"  Request rate (QPS):              {metrics['request_rate_qps']:.2f}")
        print(f"  Capacity QPS under SLA:          {metrics['capacity_qps_under_sla']:.2f}")
        print(f"  Avg output tokens/request:       {metrics['avg_output_tokens_per_request']:.1f}")
    
    if baseline_metrics is not None:
        improvements = compute_comparative_metrics(metrics, baseline_metrics)
        print(f"\n[Improvement vs Baseline - static_fifo]")
        print(f"  Throughput improvement:          {improvements.get('throughput_improvement_vs_baseline_percent', 0):.2f}%")
        print(f"  Capacity improvement:            {improvements.get('capacity_improvement_percent_vs_static', 0):.2f}%")
        print(f"  Latency improvement:             {improvements.get('latency_improvement_percent', 0):.2f}%")
        print(f"  SLA violation reduction:         {improvements.get('sla_violation_reduction_percent', 0):.2f}%")
    
    print(f"{'='*80}\n")
