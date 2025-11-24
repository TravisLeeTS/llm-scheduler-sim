# Performance Metrics Guide

## Overview

This simulator now provides **paper-aligned metrics** for direct comparison with both research papers:
1. **Multi-Bin Batching for LLM Inference Throughput Optimization**
2. **Memory-Aware and SLA-Constrained Dynamic Batching for LLM Inference**

---

## Multi-Bin Batching Paper Metrics

### Primary Metrics

| Metric | Description | Formula | Paper Section |
|--------|-------------|---------|---------------|
| **tokens_per_second** | Token throughput | `total_tokens / simulation_time` | Results, Fig. 4 |
| **requests_per_second** | Request throughput | `num_requests / simulation_time` | Results, Fig. 3 |
| **average_latency_seconds** | Mean end-to-end latency | `mean(completion_time - arrival_time)` | Results, Table 2 |
| **capacity_threshold_lambda** | Max sustainable arrival rate under SLA | `request_rate × (1 + sla_headroom)` | Analysis |
| **seconds_per_generated_token** | Time cost per output token | `simulation_time / total_output_tokens` | Performance |
| **throughput_improvement_vs_baseline_percent** | Improvement over static FIFO | `(test - baseline) / baseline × 100%` | Comparison |

### Example Output

```
📄 Multi-Bin Batching Paper Metrics:
  Tokens/second:                   2847.35
  Requests/second:                 10.23
  Average latency (seconds):       0.2871
  Capacity threshold λ:            12.45 req/s
  Seconds per generated token:     0.000351
```

---

## Dynamic Batching Paper Metrics

### Primary Metrics

| Metric | Description | Formula | Paper Section |
|--------|-------------|---------|---------------|
| **throughput_tokens_per_second** | Token generation rate | `total_tokens / simulation_time` | Results, Fig. 5 |
| **decode_step_time_ms** | Time per decoding step | `avg_service_time / avg_output_tokens × 1000` | Latency Breakdown |
| **request_rate_qps** | Queries per second | `num_requests / simulation_time` | Load Testing |
| **capacity_qps_under_sla** | Sustainable QPS meeting SLA | `requests_meeting_sla / simulation_time` | Capacity Analysis |
| **throughput_improvement_percent_vs_static** | Improvement over fixed batching | `(dynamic - static) / static × 100%` | Comparison |
| **capacity_improvement_percent_vs_static** | SLA capacity improvement | `(dynamic_capacity - static_capacity) / static_capacity × 100%` | Comparison |
| **memory_tokens_used_vs_batch_size** | Memory efficiency | `total_tokens_in_batch` vs `batch_size` | Memory Analysis |

### Example Output

```
📄 Dynamic Batching Paper Metrics:
  Decode step time:                42.15 ms
  Request rate (QPS):              10.23
  Capacity QPS under SLA:          9.54
  Avg output tokens/request:       278.3
```

---

## Core Performance Metrics

These are standard metrics reported by all schedulers:

| Metric | Description | Unit |
|--------|-------------|------|
| **Throughput (req/s)** | Request processing rate | requests/second |
| **Throughput (tok/s)** | Token processing rate | tokens/second |
| **Avg Latency** | Mean request latency | seconds |
| **P50/P95/P99 Latency** | Latency percentiles | seconds |
| **SLA Violation Rate** | Percentage exceeding SLA | % |
| **GPU Utilization** | Average GPU busy time | % |
| **Avg Queueing Delay** | Time waiting in queue | seconds |
| **Avg Service Time** | Time being processed | seconds |

---

## Batch Statistics

Track batch composition and efficiency:

| Metric | Description | Relevance |
|--------|-------------|-----------|
| **Num Batches** | Total batches formed | Efficiency |
| **Avg Batch Size** | Mean requests per batch | Memory utilization |
| **Max Batch Size** | Largest batch observed | Capacity planning |
| **Std Batch Size** | Batch size variability | Consistency |

---

## Memory Usage Estimates

Estimates memory consumption per batch:

| Metric | Description | Formula |
|--------|-------------|---------|
| **Total Tokens in Batch** | Sum of all sequence lengths | `batch_size × avg_seq_len` |
| **KV Cache Memory** | Key-Value cache size | `total_tokens × kv_mem_per_token` |
| **Total Memory** | Model + KV cache | `model_size + kv_cache_memory` |

**Example:**
```
💾 Memory Usage Estimate (Avg Batch):
  Batch size:          8
  Total tokens:        2048
  KV cache memory:     0.0102 GB
  Model memory:        2.00 GB
  Total memory:        2.0102 GB
```

---

## Improvement Metrics (vs Baseline)

When comparing against `static_fifo` baseline:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Throughput Δ (%)** | Throughput improvement | Positive = better |
| **Capacity Δ (%)** | SLA capacity improvement | Positive = better |
| **Latency Δ (%)** | Latency improvement | Positive = lower latency |
| **SLA Reduction (%)** | SLA violation reduction | Positive = fewer violations |

**Example:**
```
📊 Improvement vs Baseline (static_fifo):
  Throughput improvement:          +8.45%
  Capacity improvement:            +42.87%
  Latency improvement:             +15.34%
  SLA violation reduction:         +42.61%
```

---

## Usage Examples

### 1. Compare All Schedulers with Paper Metrics

```bash
python scripts/run_mb_dynamic.py --compare --num-requests 5000
```

**Output includes:**
- Core performance comparison table
- Multi-Bin Batching paper metrics
- Dynamic Batching paper metrics
- Improvement percentages vs baseline

### 2. Single Scheduler with Detailed Metrics

```bash
python scripts/run_mb_dynamic.py --scheduler multi_bin_dynamic --num-requests 1000
```

**Output includes:**
- All core metrics
- Paper-specific metrics
- Batch statistics
- Memory usage estimates

### 3. K_BINS Sensitivity Analysis

```bash
python scripts/run_mb_dynamic.py --k-bins-sensitivity --num-requests 5000
```

**Output includes:**
- Throughput vs K_BINS
- Latency vs K_BINS
- SLA violations vs K_BINS

---

## Validation Against Papers

### Multi-Bin Batching Paper

✅ **Metrics Covered:**
- Token throughput (tokens/sec)
- Request throughput (requests/sec)
- Average latency (seconds)
- Capacity threshold λ (max sustainable arrival rate)
- Throughput improvement % vs baseline

✅ **Analysis Supported:**
- K_BINS sensitivity (throughput scales with K)
- Batch composition efficiency
- SLA violation reduction

### Dynamic Batching Paper

✅ **Metrics Covered:**
- Decode step time (ms)
- Request rate (QPS)
- Capacity under SLA (QPS)
- Throughput improvement % vs static
- Capacity improvement % vs static
- Memory tokens usage

✅ **Analysis Supported:**
- Memory constraint validation
- SLA controller effectiveness
- Adaptive batching benefits

---

## Interpreting Results

### Good Performance Indicators

1. **High throughput improvement** (>5% vs baseline)
2. **Low SLA violation rate** (<10%)
3. **High capacity under SLA** (>90% of request rate)
4. **Consistent batch sizes** (low std deviation)
5. **High GPU utilization** (>80% without SLA violations)

### Warning Signs

1. **Identical metrics across schedulers** → Load too low or requests too similar
2. **Very low GPU utilization** (<30%) → Not saturated, invalid comparison
3. **100% SLA violations** → System overloaded or misconfigured
4. **Extremely variable batch sizes** → Unstable behavior

---

## References

### Multi-Bin Batching Paper
- **Key Result**: Multi-bin reduces E[max(t_j) | bin] by narrowing distributions
- **Metrics**: Throughput (tok/s), capacity threshold λ, latency (s)
- **Validation**: K_BINS sensitivity, composition variance

### Dynamic Batching Paper
- **Key Result**: Adaptive batching balances memory and SLA constraints
- **Metrics**: Decode time (ms), capacity QPS, improvement %
- **Validation**: b_mem constraint, b_SLA controller, memory usage

---

*Last Updated: November 24, 2025*
