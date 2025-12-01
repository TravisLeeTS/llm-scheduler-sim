# Methodology / Materials and Methods

## Document Overview

This document provides a comprehensive description of the research methodology, experimental setup, data sources, algorithms, and models used in the Multi-Bin Dynamic Batching scheduler research for LLM inference optimization.

**Date**: November 30, 2025  
**Version**: 1.0

---

## Table of Contents

1. [Research Overview](#1-research-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Sources](#3-data-sources)
4. [Latency Model](#4-latency-model)
5. [Scheduling Algorithms](#5-scheduling-algorithms)
6. [SLA Framework](#6-sla-framework)
7. [Experimental Design](#7-experimental-design)
8. [Implementation Details](#8-implementation-details)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Reproducibility](#10-reproducibility)

---

## 1. Research Overview

### 1.1 Problem Statement

Large Language Model (LLM) inference systems face critical challenges in balancing throughput and latency under production workloads. The heterogeneous nature of request lengths—where output lengths can vary by orders of magnitude—leads to inefficient batch composition when using traditional FIFO scheduling.

### 1.2 Research Objectives

1. **Primary**: Demonstrate that multi-bin batching improves LLM inference throughput while maintaining SLA compliance
2. **Secondary**: Quantify the optimal bin partitioning strategy (K parameter)
3. **Tertiary**: Compare multi-bin dynamic batching against baseline schedulers across varying workload intensities

### 1.3 Key Hypotheses

- **H1**: Partitioning requests by predicted output length reduces E[max(t_j) | bin], enabling larger batch sizes
- **H2**: Bin-specific SLA controllers outperform global controllers under heterogeneous workloads
- **H3**: Multi-bin scheduling provides better GPU utilization than single-queue approaches

### 1.4 Scientific Approach

This research follows the **wind tunnel testing** methodology for scheduler evaluation:

| Aspect | Real Deployment | Our Simulator |
|--------|----------------|---------------|
| Cost | $$$ (GPU cluster) | Free (CPU only) |
| Speed | Days/weeks | Seconds/minutes |
| Risk | User-facing | Zero (offline) |
| **Validity** | Absolute numbers | **Relative rankings** ✓ |

The simulator preserves algorithmic fidelity, enabling valid comparisons between scheduling strategies.

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Bin Dynamic Batching System                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐   │
│  │   Workload  │───▶│   Multi-Bin     │───▶│  Dynamic Batcher │   │
│  │  Generator  │    │   Scheduler     │    │  (SLA Controller)│   │
│  └─────────────┘    └─────────────────┘    └──────────────────┘   │
│         │                   │                       │              │
│         │                   │                       ▼              │
│         │            ┌──────┴──────┐         ┌──────────────┐     │
│         │            │  K Bins     │         │   GPU Pool   │     │
│         │            │ (length-    │         │  (N GPUs)    │     │
│         │            │  sorted)    │         └──────────────┘     │
│         │            └─────────────┘                │              │
│         │                                           ▼              │
│         │                                    ┌──────────────┐     │
│         └───────────────────────────────────▶│   Metrics    │     │
│                                              │  Collector   │     │
│                                              └──────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **Configuration** | `mb_dyn_sim/config.py` | System parameters and defaults |
| **Workload** | `mb_dyn_sim/workload.py` | Request generation from BurstGPT |
| **Schedulers** | `mb_dyn_sim/schedulers.py` | Multi-bin, dynamic, static FIFO |
| **Simulation** | `mb_dyn_sim/simulation.py` | Discrete-event simulator |
| **Latency Model** | `mb_dyn_sim/model_calibration.py` | GPU-calibrated latency prediction |
| **Metrics** | `mb_dyn_sim/metrics.py` | Performance measurement |

### 2.3 Simulator Design

The discrete-event simulator processes two event types:

1. **ARRIVAL**: Request enters the system and is enqueued to appropriate bin
2. **GPU_FREE**: GPU completes batch processing and becomes available for new work

**Key Optimizations**:
- Idle GPU set for O(1) lookup
- Workload caching for repeated experiments
- Incremental result saving for long-running tests

---

## 3. Data Sources

### 3.1 BurstGPT Dataset

**Source**: [HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT)  
**Citation**: Wang et al., "BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems", KDD 2025

#### Dataset Characteristics

| Property | Value |
|----------|-------|
| **Source** | Azure OpenAI production logs |
| **Duration** | 121 consecutive days |
| **Total Requests** | ~5.29 million |
| **Sample Used** | 1.43 million (BurstGPT_1.csv) |
| **Models** | ChatGPT (GPT-3.5) and GPT-4 |

#### Schema

| Column | Description | Statistics |
|--------|-------------|------------|
| `Timestamp` | Request submission time (seconds) | Start: 0, Duration: 5,270,400s |
| `Model` | Called model | ChatGPT: ~80%, GPT-4: ~20% |
| `Request tokens` | Input/prompt length | Mean: ~611, Median: ~251 |
| `Response tokens` | Output/completion length | Mean: ~123, Median: ~34 |
| `Total tokens` | Request + Response | Mean: ~734, Median: ~327 |

#### Workload Properties

```
Token Length Distribution:
- Prompt:  mean=611, median=251, p95=2461 tokens
- Output:  mean=123, median=34,  p95=460 tokens  
- Total:   mean=734, median=327, p95=2801 tokens

Arrival Pattern:
- Base rate: ~0.27 req/s (real timestamps)
- Burstiness: CV ≈ 13.28 (highly bursty)
- Daily patterns: Clear diurnal variation
```

### 3.2 GPU Calibration Data

**File**: `data/qwen3_1_7b_latency_grid.csv`  
**Model**: Qwen3-1.7B ([Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B))  
**Hardware**: NVIDIA RTX 4080 12GB  
**Precision**: FP16

#### Calibration Matrix

| Batch Sizes | Sequence Lengths | Trials | Total Configs |
|-------------|------------------|--------|---------------|
| 1, 2, 4, 8, 16, 32 | 128, 256, 512, 1024, 2048 | 5 per config | 30 |

#### Sample Measurements

```csv
batch_size,max_seq_len,mean_latency_sec,std_latency_sec,num_trials
1,128,0.080,0.006,5
1,512,0.169,0.012,5
8,512,0.222,0.015,5
32,1024,0.485,0.023,5
```

---

## 4. Latency Model

### 4.1 Model Formulation

The latency model captures the relationship between batch configuration and service time:

$$T(b, L) = \alpha + \beta \cdot L \cdot h(b)$$

Where:
- $T(b, L)$: Total batch service time (seconds)
- $b$: Batch size (number of requests)
- $L$: Maximum sequence length in batch (max of prompt_len + output_len)
- $\alpha$: Base latency (kernel launch, KV cache setup)
- $\beta$: Per-token coefficient
- $h(b)$: Batch overhead function

#### Batch Overhead Function

$$h(b) = 1 + \gamma \cdot \frac{b-1}{b}$$

Where $\gamma$ is the batch penalty factor, capturing the sublinear scaling of batched inference.

### 4.2 Calibrated Parameters

From RTX 4080 12GB measurements with Qwen3-1.7B FP16:

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| $\alpha$ | 59.653 | ms | Base latency (TTFT) |
| $\beta$ | 5.742 | ms/token | Per-token decode time |
| $\gamma$ | 0.316 | - | Batch penalty factor |
| R² | 0.9995 | - | Model fit quality |

### 4.3 Model Interpretation

**Time To First Token (TTFT)**:
$$TTFT = \alpha \approx 60ms$$

**Decode Time Between Tokens (TBT)**:
$$TBT = \beta \cdot h(b) \approx 5.74 \cdot (1 + 0.316 \cdot \frac{b-1}{b})$$

For batch size $b=8$:
$$TBT = 5.74 \cdot 1.28 = 7.3ms/token$$

### 4.4 Max-Dominates Property

**Critical Insight**: Batch completion time is determined by the longest request:

$$T_{batch} = T(b, \max_{r \in batch}(l_{prompt,r} + l_{output,r}))$$

This property is fundamental to why multi-bin batching improves throughput—by grouping similar-length requests, we reduce $\max(l_{output})$ within each bin.

---

## 5. Scheduling Algorithms

### 5.1 Algorithm Overview

Three scheduler types are implemented and compared:

| Scheduler | Queue Structure | Batch Sizing | Key Property |
|-----------|----------------|--------------|--------------|
| **static_fifo** | Single FIFO | Fixed (B=8) | Baseline |
| **dynamic_no_bins** | Single FIFO | Adaptive | SLA-aware |
| **multi_bin_dynamic** | K separate FIFOs | Bin-specific adaptive | **Our contribution** |

### 5.2 Static FIFO Scheduler (Baseline)

**Algorithm: Static Batch Formation**

```
Input: Queue Q, Fixed batch size B_FIXED = 8
Output: Batch of requests

1: function GET_BATCH(Q, B_FIXED)
2:    batch ← []
3:    for i = 1 to min(B_FIXED, |Q|) do
4:        batch.append(Q.dequeue())
5:    end for
6:    return batch
7: end function
```

**Properties**:
- No adaptation to workload characteristics
- No SLA awareness
- Mixes short and long requests (poor batch composition)

### 5.3 Dynamic No-Bins Scheduler

**Algorithm 1: Memory-Constrained Batch Size (b_mem)**

```
Input: Statistics S, Config C
Output: Maximum memory-safe batch size

1: function COMPUTE_B_MEM(S, C)
2:    η ← (M_max - M_model) / kv_mem_per_token  // Token capacity
3:    E[l_total] ← S.avg_prompt_len + S.avg_output_len
4:    L₀ ← 0.1 × η  // Safety buffer (ρ = 10%)
5:    
6:    b_mem ← floor((η - L₀) / E[l_total])
7:    
8:    return clamp(b_mem, B_MIN, B_MAX)
9: end function
```

**Statistics Update (EMA)**:
The running statistics $E[l_{prompt}]$ and $E[l_{output}]$ are maintained using exponential moving average with $\alpha_{EMA} = 0.2$:

$$E[l]_{new} = \alpha_{EMA} \cdot l_{batch} + (1 - \alpha_{EMA}) \cdot E[l]_{old}$$

**Algorithm 2: SLA-Constrained Batch Size (b_SLA)**

The SLA controller maintains an adaptive interval $[b_{low}, b_{high}]$ and exponential moving averages of decode TBT ($\tau_{avg}$) and batch size ($b_{avg}$).

**Parameters**:
- $\alpha_{step} = 4$: Interval expansion/contraction step
- $\delta_{step} = 2$: Small corrective adjustment
- $\alpha_{EMA} = 0.2$: EMA smoothing factor

**EMA Updates** (after each batch):
$$\tau_{avg}(t) = \alpha_{EMA} \cdot \tau_{recent} + (1 - \alpha_{EMA}) \cdot \tau_{avg}(t-1)$$
$$b_{avg}(t) = \alpha_{EMA} \cdot b_{recent} + (1 - \alpha_{EMA}) \cdot b_{avg}(t-1)$$

```
Input: Controller state (b_low, b_high, τ_avg, b_avg), SLA params (D_SLA, ε_D)
Output: SLA-safe batch size

1: function COMPUTE_B_SLA()
2:    α ← 4  // Interval adjustment step
3:    δ ← 2  // Corrective step
4:    
5:    if τ_avg > D_SLA + ε_D then
6:        // Case 1: Latency too high → shrink interval to left
7:        b_high ← max(b_avg, b_low + α)
8:        b_low ← max(b_low - δ, B_MIN)
9:    else if τ_avg < D_SLA - ε_D then
10:       // Case 2: Latency too low → expand interval to right
11:       b_low ← min(b_avg, b_high - α)
12:       b_high ← min(b_high + δ, B_MAX)
13:   else
14:       // Case 3: Within tolerance → center around average
15:       b_high ← min(b_avg + α/2, B_MAX)
16:       b_low ← max(b_avg - α/2, B_MIN)
17:   end if
18:   
19:   return floor((b_low + b_high) / 2)
20: end function
```

**Final Batch Size**:
$$b_{target} = \min(b_{mem}, b_{SLA})$$

### 5.4 Multi-Bin Dynamic Scheduler (Our Contribution)

**Key Innovation**: Partition requests by predicted output length into K bins, then apply dynamic batching within each bin.

**Algorithm 3: Equal-Mass Bin Boundaries**

```
Input: Predicted output lengths L = [l₁, l₂, ..., lₙ], Number of bins K
Output: Bin boundaries [(min₀, max₀), (min₁, max₁), ..., (min_{K-1}, max_{K-1})]

1: function COMPUTE_BIN_BOUNDARIES(L, K)
2:    quantiles ← linspace(0, 1, K+1)
3:    boundary_points ← quantile(L, quantiles)
4:    boundaries ← []
5:    
6:    for i = 0 to K-1 do
7:        min_len ← floor(boundary_points[i])
8:        max_len ← floor(boundary_points[i+1]) if i < K-1 else ∞
9:        boundaries.append((min_len, max_len))
10:   end for
11:   
12:   return boundaries
13: end function
```

**Algorithm 4: Multi-Bin Request Enqueue**

```
Input: Request r with predicted_output_len, Bin boundaries B
Output: Request assigned to appropriate bin

1: function ENQUEUE_REQUEST(r, B)
2:    for i = 0 to K-1 do
3:        if B[i].min ≤ r.predicted_output_len < B[i].max then
4:            bins[i].enqueue(r)
5:            return
6:        end if
7:    end for
8:    bins[K-1].enqueue(r)  // Fallback to last bin
9: end function
```

**Algorithm 5: Multi-Bin Batch Formation**

```
Input: K bins, GPU g, Max candidates M
Output: Batch from one bin, bin index

1: function GET_BATCH_MULTI_BIN(bins, g, M)
2:    // Select bin (round-robin or longest-queue)
3:    bin_idx ← SELECT_BIN(bins)
4:    if bin_idx = NULL then return [], -1
5:    
6:    // Get candidates from selected bin only
7:    candidates ← []
8:    for i = 1 to min(M, |bins[bin_idx]|) do
9:        candidates.append(bins[bin_idx].dequeue())
10:   end for
11:   
12:   // Apply bin-specific dynamic batching
13:   stats ← bin_stats[bin_idx]
14:   controller ← bin_controllers[bin_idx]
15:   
16:   b_mem ← COMPUTE_B_MEM(stats, config, bin_idx)
17:   b_SLA ← controller.COMPUTE_B_SLA()
18:   b_target ← min(b_mem, b_SLA)
19:   
20:   batch ← candidates[0:b_target]
21:   service_time ← ESTIMATE_SERVICE_TIME(batch)
22:   
23:   // Return unused candidates to front of bin queue
24:   for r in reversed(candidates[b_target:]) do
25:       bins[bin_idx].prepend(r)
26:   end for
27:   
28:   return batch, bin_idx
29: end function
```

### 5.5 Why Multi-Bin Improves Throughput

**Mathematical Foundation**:

Without bins (single queue):
```
Queue: [1 token, 100 tokens, 2 tokens, 3 tokens]
Batch: [1, 100, 2, 3]
→ Batch time dominated by max(100) 
→ Throughput = 4 / T(4, 100) ≈ 0.04 req/time
```

With bins (partitioned):
```
Bin 0 (0-64):   [1, 2, 3, 5]
Bin 1 (64-256): [100, 150, 200]

Batch from Bin 0: [1, 2, 3, 5]
→ Batch time = T(4, 5) << T(4, 100)
→ Throughput = 4 / T(4, 5) ≈ 0.80 req/time
```

**Result**: ~20x throughput improvement in this example!

**Formal Statement**: For K bins with boundaries $[L_{min,k}, L_{max,k}]$:

$$E[\max(l_j) | bin_k] < E[\max(l_j) | all]$$

This enables larger batch sizes while maintaining SLA, since:

$$TBT = \beta \cdot h(b) \propto \frac{T(b, L_{max,k})}{L_{max,k}}$$

is lower when $L_{max,k}$ is bounded by bin partitioning.

---

## 6. SLA Framework

### 6.1 Dual SLA Model (v2)

The system implements a dual-SLA framework for comprehensive latency management:

| SLA Type | Metric | Target | Purpose |
|----------|--------|--------|---------|
| **Per-Token SLA** | Decode TBT only | D_SLA_TOKEN = 10ms | Streaming UX |
| **Per-Request SLA** | Total latency | D_SLA_REQUEST = 20s | Interactive response |

### 6.2 Per-Token SLA (D_SLA_TOKEN)

**Definition**: Maximum allowed per-token decode latency (Time Between Tokens)

**Computation**:
$$TBT_{decode} = \beta \cdot h(b)$$

Where:
- $\beta = 5.742ms$ (per-token decode coefficient)
- $h(b) = 1 + \gamma \cdot \frac{b-1}{b}$ (batch overhead)

**Critical Design Decision**: Token SLA applies ONLY to decode TBT, NOT TTFT. This prevents structural violations where $\alpha/L$ dominates for short outputs.

**Violation Condition**:
$$\text{violates\_token\_sla}(r) = (TBT_{decode}(r) > D\_SLA\_TOKEN)$$

where $TBT_{decode}(r)$ is the decode time-between-tokens for request $r$, computed as $\beta \cdot h(b)$ for the batch containing $r$. Since all requests in a batch share the same decode TBT, this per-request metric accurately captures token-level SLA compliance.

### 6.3 Per-Request SLA (D_SLA_REQUEST)

**Definition**: Maximum allowed total request latency from arrival to completion

**Computation**:
$$Latency_{total} = T_{completion} - T_{arrival}$$

**Components**:
$$Latency_{total} = Queueing\_Delay + Service\_Time$$

Where:
- $Queueing\_Delay = T_{service\_start} - T_{arrival}$
- $Service\_Time = T_{completion} - T_{service\_start}$

**Violation Condition**:
$$\text{violates\_request\_sla} = (Latency_{total} > D\_SLA\_REQUEST)$$

### 6.4 SLA Thresholds Used in Experiments

| Experiment Set | D_SLA_TOKEN | D_SLA_REQUEST | RPS_SCALING |
|----------------|-------------|---------------|-------------|
| High Load (100x) | 10ms | 20s | 100 |
| Low Load (10x) | 10ms | 20s | 10 |
| Ultra-Low Load (1x) | 10ms | 20s | 1 |

### 6.5 Industry Reference

| Metric | Gemini 2.5 Flash-Lite | Our Target | Notes |
|--------|----------------------|------------|-------|
| TTFT | 240ms | 60ms (α) | Prefill latency |
| Output Speed | 410 tokens/sec | 174 tokens/sec | Decode rate |
| Per-token TBT | 2.44ms | 5.74ms (β) | Conservative for RTX 4080 |

---

## 7. Experimental Design

### 7.1 Experimental Framework

The experimental evaluation follows a two-step process:

#### Step 1: Grid Search (Parameter Space Exploration)

**Objective**: Find optimal GPU and bin configurations across workload sizes

**Independent Variables**:
| Parameter | Values | Purpose |
|-----------|--------|---------|
| NUM_REQUESTS | 1K, 10K, 100K, 1M | Scale stress testing |
| NUM_GPUS | 1, 2, 4, 8, 16, 32, 64, 100 | Resource scaling |
| K_BINS | 1, 2, 4, 8, 16, 32 | Partitioning granularity |

**Fixed Parameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| B_MIN | 1 | Minimum batch size |
| B_MAX | 128 | Maximum batch size |
| D_SLA_TOKEN | 10ms | Per-token latency threshold |
| D_SLA_REQUEST | 20s | Per-request latency threshold |
| RPS_SCALING | Variable | Load intensity control |
| SEED | 42 | Reproducibility |

**Total Configurations**: 
$$4 \times 8 \times 6 = 192 \text{ experiments per RPS level}$$

#### Step 2: Method Comparison

**Objective**: Compare scheduling methods with optimal configurations

**Methods Tested**:
1. **Static FIFO** (1 GPU): Baseline with fixed batch size B=8
2. **Dynamic No-Bins** (1 GPU): Single queue with SLA controller
3. **Multi-Bin Dynamic** (1 GPU, K=8): Our method with minimal resources
4. **Multi-Bin Dynamic** (Optimal GPUs, Optimal K): Best configuration from Step 1

**Workload Sizes**: 1K, 10K, 100K, 1M requests

**Total Experiments**: 4 methods × 4 workload sizes = 16 experiments per RPS level

### 7.2 Load Intensity Levels

Three RPS scaling factors create different stress levels:

| Load Level | RPS_SCALING | Arrival Rate | Purpose |
|------------|-------------|--------------|---------|
| **High** | 100x | ~27 req/s | Stress testing, capacity limits |
| **Medium** | 10x | ~2.7 req/s | Moderate load, balanced evaluation |
| **Low** | 1x | ~0.27 req/s | Near-real arrival rate |

**Why RPS Scaling?**
- Real BurstGPT arrival rate (0.27 req/s) is easily handled by all schedulers
- Scaling preserves temporal patterns (burstiness, CV≈13.3)
- Creates meaningful differentiation between scheduling strategies

### 7.3 Experiment Scripts

| Script | Purpose | Output Location |
|--------|---------|-----------------|
| `step1_low_load.py` | Grid search (10x RPS) | `stress_test_low_load/` |
| `step2_low_load.py` | Method comparison (10x RPS) | `stress_test_low_load/` |
| `step1_ultra_low_load.py` | Grid search (1x RPS) | `stress_test_ultra_low_load/` |
| `step2_ultra_low_load.py` | Method comparison (1x RPS) | `stress_test_ultra_low_load/` |

### 7.4 Experimental Controls

**Randomization Control**:
- Fixed SEED=42 for workload generation
- Deterministic request ordering by arrival time

**Hardware Normalization**:
- GPU-calibrated latency model ensures consistent timing
- Memory constraints based on RTX 4080 12GB specifications

**Measurement Protocol**:
- Each configuration run to completion (all requests processed)
- Metrics computed from completed requests only
- Incremental saves for long-running experiments

---

## 8. Implementation Details

### 8.1 Configuration Parameters

```python
@dataclass
class SchedulerConfig:
    # GPU Infrastructure
    NUM_GPUS: int = 4              # Number of parallel GPUs
    M_MAX_GB: float = 12.0         # GPU memory (RTX 4080)
    M_MODEL_GB: float = 4.0        # Model VRAM (Qwen3 1.7B FP16)
    KV_MEM_PER_TOKEN_GB: float = 1.875e-4  # KV cache per token
    
    # Multi-Bin Configuration
    K_BINS: int = 4                # Number of output length bins
    USE_EQUAL_MASS_BINS: bool = True  # Equal probability per bin
    BIN_BOUNDARIES: List[Tuple[int, int]] = None  # Auto-computed
    BIN_B_MAX: List[int] = None    # Per-bin batch limits
    
    # Dynamic Batching
    B_MIN: int = 1                 # Minimum batch size
    B_MAX: int = 128               # Maximum batch size
    
    # SLA Constraints (v2 Model)
    D_SLA_TOKEN: float = 0.010     # 10ms per-token decode
    D_SLA_REQUEST: float = 20.0    # 20s per-request latency
    LATENCY_EPSILON: float = 0.005 # 5ms tolerance band
    
    # Workload
    USE_REAL_TIMESTAMPS: bool = False  # RPS scaling mode
    RPS_SCALING: float = 10.0      # Arrival rate multiplier
```

### 8.2 Per-Bin Batch Size Limits

Longer bins receive smaller maximum batch sizes due to memory constraints:

$$b_{max,k} = \min\left(B_{MAX}, \left\lfloor\frac{M_{avail}}{avg\_seq\_len_k \times kv\_per\_token}\right\rfloor\right)$$

**Example Bin Configuration** (K=8, 1M requests):
```
Bin 0: [1, 30) tokens,   B_MAX=128
Bin 1: [30, 42) tokens,  B_MAX=128
Bin 2: [42, 56) tokens,  B_MAX=128
Bin 3: [56, 80) tokens,  B_MAX=128
Bin 4: [80, 117) tokens, B_MAX=128
Bin 5: [117, 167) tokens,B_MAX=128
Bin 6: [167, 281) tokens,B_MAX=83
Bin 7: [281, ∞) tokens,  B_MAX=8
```

### 8.3 Feedback Loop Implementation

The SLA controller uses exponential moving average (EMA) for stable adaptation:

```python
def update(self, recent_tbt, recent_batch_size, N_decode):
    # EMA smoothing (α = 0.2)
    self.tau_avg = 0.2 * recent_tbt + 0.8 * self.tau_avg
    self.b_avg = 0.2 * recent_batch_size + 0.8 * self.b_avg
    self.N_decode = N_decode
    self.update_count += 1
```

### 8.4 Request Data Structure

```python
@dataclass
class Request:
    id: int
    arrival_time: float
    prompt_len: int
    output_len: int
    predicted_output_len: int
    deadline: float  # D_SLA_TOKEN
    deadline_request: float = 20.0  # D_SLA_REQUEST
    
    # Filled during simulation
    start_service_time: float = -1.0
    completion_time: float = -1.0
    assigned_gpu: int = -1
    
    # v2 SLA metrics
    ttft: float = -1.0        # Time To First Token
    decode_tbt: float = -1.0  # Decode-only TBT
```

---

## 9. Evaluation Metrics

### 9.1 Primary Metrics

| Metric | Formula | Unit | Paper Reference |
|--------|---------|------|-----------------|
| **Throughput (req/s)** | $\frac{N_{completed}}{T_{total}}$ | requests/sec | Multi-Bin, Dynamic |
| **Throughput (tok/s)** | $\frac{\sum tokens}{T_{total}}$ | tokens/sec | Multi-Bin |
| **Avg Latency** | $\frac{1}{N}\sum(T_{completion} - T_{arrival})$ | seconds | Both |
| **Token SLA Violation %** | $\frac{N(TBT > D\_SLA\_TOKEN)}{N} \times 100$ | % | Dynamic |
| **Request SLA Violation %** | $\frac{N(Latency > D\_SLA\_REQUEST)}{N} \times 100$ | % | Production |

### 9.2 Latency Percentiles

| Metric | Description |
|--------|-------------|
| P50 Latency | Median request latency |
| P95 Latency | 95th percentile latency |
| P99 Latency | 99th percentile latency |
| Max Latency | Maximum observed latency |

### 9.3 Resource Utilization

| Metric | Formula | Description |
|--------|---------|-------------|
| GPU Utilization | $\frac{T_{busy}}{T_{total}}$ | Average GPU busy time |
| Avg Batch Size | $\frac{\sum |batch|}{N_{batches}}$ | Mean requests per batch |
| Queueing Delay | $T_{service\_start} - T_{arrival}$ | Time waiting in queue |

### 9.4 Paper-Specific Metrics

**Multi-Bin Batching Paper**:
- `capacity_threshold_lambda`: Maximum sustainable arrival rate under SLA
- `seconds_per_generated_token`: Time cost per output token

**Dynamic Batching Paper**:
- `decode_step_time_ms`: Average time per decode step
- `capacity_qps_under_sla`: Sustainable QPS meeting SLA

---

## 10. Reproducibility

### 10.1 Software Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
```

### 10.2 Hardware Requirements

**Simulation (CPU only)**:
- Any modern CPU
- 8GB+ RAM for 1M request workloads
- No GPU required

**Calibration (GPU required)**:
- NVIDIA RTX 4080 12GB (or equivalent)
- CUDA 12.x
- HuggingFace Transformers

### 10.3 Running Experiments

```bash
# Step 1: Grid Search (Low Load)
python scripts/step1_low_load.py

# Step 2: Method Comparison (Low Load)
python scripts/step2_low_load.py

# Full Pipeline
cmd /c "cd /d c:\Users\tings\llm_scheduler_sim && python scripts/step1_low_load.py && python scripts/step2_low_load.py"
```

### 10.4 Output Files

| File | Contents |
|------|----------|
| `step1_grid_search.csv` | 192 configurations × metrics |
| `step2_comparison.csv` | 16 experiments × metrics |

### 10.5 Random Seed Control

All experiments use SEED=42 for reproducibility:
```python
cfg = SchedulerConfig(SEED=42, ...)
```

---

## Document References

For additional details, see:

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture diagrams |
| [LATENCY_MODEL_AND_SLA.md](LATENCY_MODEL_AND_SLA.md) | Latency model derivation |
| [METRICS_GUIDE.md](METRICS_GUIDE.md) | Metrics computation details |
| [data/README.md](data/README.md) | Dataset format specification |

---

**Document Version**: 1.0  
**Last Updated**: November 30, 2025  
**Authors**: Research Team
