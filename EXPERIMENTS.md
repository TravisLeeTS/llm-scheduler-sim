# Experimental Design and Protocol

## Document Overview

This document provides comprehensive details on the experimental setup, including the two-step evaluation framework, grid search parameters, method comparison protocol, and expected outputs.

**Date**: November 30, 2025  
**Version**: 1.0

---

## Table of Contents

1. [Experimental Framework](#1-experimental-framework)
2. [Step 1: Grid Search](#2-step-1-grid-search)
3. [Step 2: Method Comparison](#3-step-2-method-comparison)
4. [Load Intensity Levels](#4-load-intensity-levels)
5. [Scheduler Configurations](#5-scheduler-configurations)
6. [Output Specifications](#6-output-specifications)
7. [Execution Instructions](#7-execution-instructions)

---

## 1. Experimental Framework

### 1.1 Two-Step Evaluation Process

The experimental evaluation follows a systematic two-step approach:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: Grid Search                              │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │  Parameter Space: 192 configurations                          │ │
│  │  - Workload:  1K, 10K, 100K, 1M requests                     │ │
│  │  - GPUs:      1, 2, 4, 8, 16, 32, 64, 100                    │ │
│  │  - K_BINS:    1, 2, 4, 8, 16, 32                             │ │
│  │                                                               │ │
│  │  Output: Optimal configuration per workload size              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    STEP 2: Method Comparison                  │ │
│  │  4 Methods × 4 Workloads = 16 experiments                     │ │
│  │                                                               │ │
│  │  Methods:                                                     │ │
│  │  1. Static FIFO (1 GPU) - Baseline                           │ │
│  │  2. Dynamic No-Bins (1 GPU) - SLA-aware baseline             │ │
│  │  3. Multi-Bin Dynamic (1 GPU, K=8) - Minimal resources       │ │
│  │  4. Multi-Bin Dynamic (Optimal) - Best config from Step 1    │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Research Questions Addressed

| Step | Research Question | Metrics |
|------|-------------------|---------|
| **Step 1** | What are the optimal GPU/bin configurations for each workload size? | Token SLA %, Request SLA % |
| **Step 1** | How does performance scale with GPU count? | Throughput, Utilization |
| **Step 1** | What is the optimal number of bins (K)? | SLA violation rate |
| **Step 2** | Does multi-bin outperform baselines? | All metrics vs baselines |
| **Step 2** | What is the benefit of optimal configuration? | Improvement % |

---

## 2. Step 1: Grid Search

### 2.1 Parameter Space

**Independent Variables**:

| Parameter | Values | Count |
|-----------|--------|-------|
| NUM_REQUESTS | 1,000 / 10,000 / 100,000 / 1,000,000 | 4 |
| NUM_GPUS | 1 / 2 / 4 / 8 / 16 / 32 / 64 / 100 | 8 |
| K_BINS | 1 / 2 / 4 / 8 / 16 / 32 | 6 |

**Total Configurations**: 4 × 8 × 6 = **192 experiments**

### 2.2 Fixed Parameters

```python
SCHEDULER_TYPE = "multi_bin_dynamic"
B_MIN = 1
B_MAX = 128
D_SLA_TOKEN = 0.010      # 10ms per-token decode TBT
D_SLA_REQUEST = 20.0     # 20 seconds total request latency
LATENCY_EPSILON = 0.001  # 1ms tolerance
USE_REAL_CALIBRATION = True
WORKLOAD_SOURCE = "burstgpt_dataset"
USE_REAL_TIMESTAMPS = False  # Use RPS scaling
SEED = 42
```

### 2.3 Algorithm: Optimal Configuration Selection

```python
def find_optimal_config(step1_df, num_requests, sla_threshold=5.0):
    """
    Find optimal GPU/bin configuration for a given workload size.
    
    Selection Criteria:
    1. Filter configs where request_sla_pct <= sla_threshold
    2. Among passing configs, select minimum GPU count
    3. Among same GPU count, select lowest SLA violation
    """
    subset = step1_df[step1_df['num_requests'] == num_requests]
    
    # Find configs meeting SLA threshold
    good_configs = subset[subset['request_sla_pct'] <= sla_threshold]
    
    if good_configs.empty:
        # Fallback: return best available
        best = subset.loc[subset['request_sla_pct'].idxmin()]
        return int(best['num_gpus']), int(best['k_bins'])
    
    # Select minimum GPU count that meets SLA
    min_gpus = good_configs['num_gpus'].min()
    min_gpu_configs = good_configs[good_configs['num_gpus'] == min_gpus]
    
    # Among same GPU count, select lowest SLA violation
    best = min_gpu_configs.loc[min_gpu_configs['request_sla_pct'].idxmin()]
    
    return int(best['num_gpus']), int(best['k_bins'])
```

### 2.4 Output Schema: step1_grid_search.csv

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Experiment timestamp |
| `num_requests` | int | Workload size |
| `num_gpus` | int | GPU count |
| `k_bins` | int | Number of bins |
| `rps_scaling` | float | RPS multiplier |
| `d_sla_token_ms` | float | Token SLA threshold (ms) |
| `d_sla_request_s` | float | Request SLA threshold (s) |
| `token_sla_pct` | float | Token SLA violation % |
| `request_sla_pct` | float | Request SLA violation % |
| `avg_ttft_ms` | float | Average TTFT (ms) |
| `avg_decode_tbt_ms` | float | Average decode TBT (ms) |
| `avg_latency_s` | float | Average latency (s) |
| `p50_latency_s` | float | P50 latency (s) |
| `p95_latency_s` | float | P95 latency (s) |
| `p99_latency_s` | float | P99 latency (s) |
| `avg_queue_delay_s` | float | Average queueing delay (s) |
| `avg_service_time_s` | float | Average service time (s) |
| `throughput_tok_s` | float | Throughput (tokens/s) |
| `throughput_req_s` | float | Throughput (requests/s) |
| `avg_batch_size` | float | Average batch size |
| `gpu_utilization_pct` | float | GPU utilization % |
| `execution_time_s` | float | Experiment runtime (s) |
| `completed_requests` | int | Completed request count |

---

## 3. Step 2: Method Comparison

### 3.1 Methods Under Test

| Method | Scheduler Type | GPUs | K_BINS | Description |
|--------|----------------|------|--------|-------------|
| **1. Static FIFO** | `static_fifo` | 1 | 1 | Baseline: Fixed batch B=8, no adaptation |
| **2. Dynamic No-Bins** | `dynamic_no_bins` | 1 | 1 | Single queue + SLA controller |
| **3. Multi-Bin Dynamic (1 GPU)** | `multi_bin_dynamic` | 1 | 8 | Our method, minimal resources |
| **4. Multi-Bin Dynamic (Optimal)** | `multi_bin_dynamic` | From Step 1 | From Step 1 | Best config per workload |

### 3.2 Experiment Matrix

| Workload | Static FIFO | Dynamic No-Bins | MB-Dynamic (1 GPU) | MB-Dynamic (Optimal) |
|----------|-------------|-----------------|--------------------|-----------------------|
| 1K | ✓ | ✓ | ✓ | ✓ |
| 10K | ✓ | ✓ | ✓ | ✓ |
| 100K | ✓ | ✓ | ✓ | ✓ |
| 1M | ✓ | ✓ | ✓ | ✓ |

**Total Experiments**: 4 methods × 4 workloads = **16 experiments**

### 3.3 Comparison Metrics

**Primary Comparison**:
- Token SLA Violation %
- Request SLA Violation %
- Throughput (req/s)
- Average Latency

**Secondary Comparison**:
- GPU Utilization
- Queueing Delay
- Service Time
- Batch Size

### 3.4 Output Schema: step2_comparison.csv

Includes all columns from Step 1 plus:

| Column | Type | Description |
|--------|------|-------------|
| `method` | string | Method name (e.g., "Static FIFO (1 GPU)") |
| `scheduler_type` | string | Scheduler type (`static_fifo`, `dynamic_no_bins`, `multi_bin_dynamic`) |

---

## 4. Load Intensity Levels

### 4.1 Three RPS Scaling Levels

| Level | RPS_SCALING | Arrival Rate | Purpose | Folder |
|-------|-------------|--------------|---------|--------|
| **High Load** | 100x | ~27 req/s | Stress testing | `stress_test_final/` |
| **Medium Load** | 10x | ~2.7 req/s | Balanced evaluation | `stress_test_low_load/` |
| **Low Load** | 1x | ~0.27 req/s | Near-real workload | `stress_test_ultra_low_load/` |

### 4.2 Load Level Rationale

**High Load (100x)**:
- Creates significant queueing delays
- Forces scheduler differentiation
- Tests capacity limits
- **Use case**: Stress testing, capacity planning

**Medium Load (10x)**:
- Moderate queueing pressure
- Balanced throughput/latency trade-offs
- Reasonable resource requirements
- **Use case**: General performance comparison

**Low Load (1x)**:
- Near-real BurstGPT arrival pattern
- Minimal queueing (most requests served immediately)
- Tests steady-state behavior
- **Use case**: Baseline validation, real-world approximation

### 4.3 Arrival Rate Computation

The BurstGPT dataset has a base arrival rate of approximately 0.27 req/s:

$$\lambda_{scaled} = \lambda_{base} \times RPS\_SCALING = 0.27 \times X$$

| RPS_SCALING | Arrival Rate | Inter-arrival Time |
|-------------|--------------|-------------------|
| 1x | 0.27 req/s | 3.7s average |
| 10x | 2.7 req/s | 0.37s average |
| 100x | 27 req/s | 0.037s average |

---

## 5. Scheduler Configurations

### 5.1 Static FIFO (Baseline)

```python
# Configuration
scheduler_type = "static_fifo"
fixed_batch_size = 8

# Behavior
- Single FIFO queue
- Fixed batch size (no adaptation)
- No SLA awareness
- No length-based partitioning
```

**Pseudocode**:
```
function GET_BATCH():
    batch = []
    for i in range(min(8, queue.length)):
        batch.append(queue.dequeue())
    return batch
```

### 5.2 Dynamic No-Bins

```python
# Configuration
scheduler_type = "dynamic_no_bins"
B_MIN = 1
B_MAX = 128

# Behavior
- Single FIFO queue
- Adaptive batch sizing: b_target = min(b_mem, b_SLA)
- Global SLA controller
- No length-based partitioning
```

**Pseudocode**:
```
function GET_BATCH():
    candidates = queue.get_candidates(MAX_CANDIDATES)
    b_mem = compute_b_mem(global_stats)
    b_SLA = global_controller.compute_b_SLA()
    b_target = min(b_mem, b_SLA)
    
    batch = candidates[:b_target]
    return_unused_to_front(candidates[b_target:])
    return batch
```

### 5.3 Multi-Bin Dynamic (Our Contribution)

```python
# Configuration
scheduler_type = "multi_bin_dynamic"
K_BINS = variable (1, 2, 4, 8, 16, 32)
B_MIN = 1
B_MAX = 128 (reduced per-bin for longer sequences)

# Behavior
- K separate FIFO queues (one per bin)
- Bin-specific adaptive batch sizing
- Bin-specific SLA controllers
- Length-based request partitioning
```

**Pseudocode**:
```
function ENQUEUE(request):
    bin_idx = select_bin(request.predicted_output_len)
    bins[bin_idx].enqueue(request)

function GET_BATCH():
    bin_idx = select_bin_round_robin()  # or longest_queue
    candidates = bins[bin_idx].get_candidates(MAX_CANDIDATES)
    
    # Bin-specific adaptation
    stats = bin_stats[bin_idx]
    controller = bin_controllers[bin_idx]
    
    b_mem = compute_b_mem(stats, bin_idx)  # Per-bin B_MAX
    b_SLA = controller.compute_b_SLA()
    b_target = min(b_mem, b_SLA)
    
    batch = candidates[:b_target]
    return_unused_to_front_of_bin(candidates[b_target:], bin_idx)
    return batch
```

---

## 6. Output Specifications

### 6.1 Directory Structure

```
llm_scheduler_sim/
├── stress_test_final/           # High Load (100x)
│   ├── step1_grid_search.csv
│   └── step2_comparison.csv
├── stress_test_low_load/        # Medium Load (10x)
│   ├── step1_grid_search.csv
│   └── step2_comparison.csv
├── stress_test_ultra_low_load/  # Low Load (1x)
│   ├── step1_grid_search.csv
│   └── step2_comparison.csv
└── scripts/
    ├── step1_low_load.py
    ├── step2_low_load.py
    ├── step1_ultra_low_load.py
    └── step2_ultra_low_load.py
```

### 6.2 File Specifications

**step1_grid_search.csv**:
- Rows: 192 (4 workloads × 8 GPU configs × 6 bin configs)
- Columns: 22 metrics
- Size: ~50-100 KB

**step2_comparison.csv**:
- Rows: 16 (4 methods × 4 workloads)
- Columns: 24 metrics (includes method info)
- Size: ~5-10 KB

### 6.3 Expected Results Summary

#### High Load (100x RPS) - 1M Requests

| Method | Token SLA | Request SLA | Throughput | Avg Latency |
|--------|-----------|-------------|------------|-------------|
| Static FIFO (1 GPU) | 0.0% | ~99.9% | ~27 req/s | Hours |
| Dynamic No-Bins (1 GPU) | 0.0% | ~99% | ~27 req/s | Hours |
| MB-Dynamic (1 GPU) | 0.0% | ~99% | ~27 req/s | Hours |
| **MB-Dynamic (Optimal)** | **0.0%** | **~15%** | **~27 req/s** | **~seconds** |

#### Low Load (1x RPS) - 1M Requests

| Method | Token SLA | Request SLA | Throughput | Avg Latency |
|--------|-----------|-------------|------------|-------------|
| Static FIFO (1 GPU) | 0.0% | ~80% | ~0.27 req/s | Hours |
| Dynamic No-Bins (1 GPU) | 0.0% | ~56% | ~0.27 req/s | Minutes |
| MB-Dynamic (1 GPU) | 0.0% | ~55% | ~0.27 req/s | Minutes |
| **MB-Dynamic (Optimal)** | **0.0%** | **~3.8%** | **~0.27 req/s** | **~5s** |

---

## 7. Execution Instructions

### 7.1 Full Pipeline Execution

```bash
# High Load (100x) - if not already done
cd c:\Users\tings\llm_scheduler_sim
python scripts/step1_high_load.py
python scripts/step2_high_load.py

# Medium Load (10x)
python scripts/step1_low_load.py
python scripts/step2_low_load.py

# Low Load (1x)
python scripts/step1_ultra_low_load.py
python scripts/step2_ultra_low_load.py
```

### 7.2 Combined Execution (PowerShell)

```powershell
# Run Step 1 and Step 2 sequentially
cmd /c "cd /d c:\Users\tings\llm_scheduler_sim && python scripts/step1_low_load.py && python scripts/step2_low_load.py"
```

### 7.3 Resume Support

Both Step 1 and Step 2 scripts support **incremental saving**:
- Existing results are loaded on startup
- Only missing configurations are computed
- Results appended incrementally to CSV

**To restart from scratch**:
```bash
rm stress_test_low_load/*.csv
python scripts/step1_low_load.py
```

### 7.4 Expected Execution Times

| Experiment | Step 1 (192 configs) | Step 2 (16 experiments) |
|------------|---------------------|------------------------|
| 1M requests (High Load) | ~2-4 hours | ~30-60 min |
| 1M requests (Medium Load) | ~2-4 hours | ~30-60 min |
| 1M requests (Low Load) | ~2-4 hours | ~30-60 min |

*Times vary based on hardware and whether using incremental resume.*

---

## Key Findings Summary

See [METHODOLOGY.md](METHODOLOGY.md) for complete methodology details and [LATENCY_MODEL_AND_SLA.md](LATENCY_MODEL_AND_SLA.md) for latency model derivation.

---

**Document Version**: 1.0  
**Last Updated**: November 30, 2025
