# System Architecture - Multi-Bin + Dynamic Batching Scheduler

## Document Overview

This document provides detailed architecture diagrams and process flows for the Multi-Bin Dynamic Batching scheduler for LLM inference optimization.

**Date**: November 30, 2025  
**Version**: 2.0

---

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Scheduler Architectures](#scheduler-architectures)
4. [Algorithm Details](#algorithm-details)
5. [Data Flow](#data-flow)
6. [Configuration Reference](#configuration)

---

## Overview

### Three Scheduler Types

| Scheduler | Queue Structure | Batching | Innovation |
|-----------|-----------------|----------|------------|
| **static_fifo** | Single FIFO | Fixed (B=8) | Baseline |
| **dynamic_no_bins** | Single FIFO | Adaptive | SLA-aware |
| **multi_bin_dynamic** | K separate FIFOs | Bin-specific adaptive | **Our contribution** |

### Production Configuration

| Component | Implementation | Details |
|-----------|----------------|---------|
| **Workload** | BurstGPT dataset | Real Azure ChatGPT traces (1.43M requests) |
| **Latency Model** | GPU-calibrated | Qwen3 1.7B FP16 on RTX 4080 12GB |
| **GPU Config** | Configurable | 1-100 GPUs, 12GB memory model |
| **SLA (Token)** | D_SLA_TOKEN = 10ms | Decode TBT threshold (streaming UX) |
| **SLA (Request)** | D_SLA_REQUEST = 20s | Total latency threshold (interactive) |

---

## Service Level Agreement (SLA) Definition

### Dual SLA Framework (v2 Model)

The system implements a dual-SLA framework separating token-level and request-level latency targets:

#### 1. Per-Token SLA (D_SLA_TOKEN = 10ms)

**Definition**: Maximum allowed per-token decode latency (Time Between Tokens)

$$TBT_{decode} = \beta \cdot h(b) \leq D\_SLA\_TOKEN$$

Where:
- $\beta = 5.742ms$ (per-token decode coefficient from calibration)
- $h(b) = 1 + \gamma \cdot (b-1)/b$ (batch overhead factor)
- $\gamma = 0.316$ (batch penalty from calibration)

**Critical Design**: Token SLA applies ONLY to decode TBT, NOT to TTFT (prefill latency).

#### 2. Per-Request SLA (D_SLA_REQUEST = 20s)

**Definition**: Maximum allowed total request latency

$$Latency_{total} = T_{completion} - T_{arrival} \leq D\_SLA\_REQUEST$$

**Components**:
- Queueing delay: Time waiting in scheduler queue
- Service time: Actual GPU processing time

### SLA Evaluation

| Metric | Computation | Threshold | Purpose |
|--------|-------------|-----------|---------|
| **Token SLA Violation** | `decode_tbt > D_SLA_TOKEN` | 10ms | Streaming UX |
| **Request SLA Violation** | `total_latency > D_SLA_REQUEST` | 20s | Interactive response |

---

## Core Components

### 1. Workload Generator (`mb_dyn_sim/workload.py`)

- **Input**: BurstGPT CSV dataset (real Azure ChatGPT traces)
- **Process**: Loads actual arrival times, prompt lengths, output lengths
- **Two Modes**:
  - **RPS Scaling** (default): Compress arrival times for stress testing
  - **Real Timestamps**: Preserves actual inter-arrival patterns

### 2. Latency Model (`mb_dyn_sim/model_calibration.py`)

**GPU-Calibrated Formula**:
```
T(batch_size, max_seq_len) = α + β · L · (1 + γ · (b-1)/b)
```

Where:
- α = base latency (kernel launch overhead)
- β = per-token coefficient  
- γ = batch penalty factor

**Calibration source**: `data/qwen3_1_7b_latency_grid.csv` (Qwen3 1.7B FP16 measurements)

### 3. Discrete-Event Simulator (`mb_dyn_sim/simulation.py`)

- **Event Queue**: Priority queue ordered by timestamp
- **Event Types**: 
  - `ARRIVAL`: Request enters system
  - `GPU_FREE`: GPU completes batch and becomes available
- **Optimization**: Idle GPU set for O(1) lookup

### 4. Schedulers (`mb_dyn_sim/schedulers.py`)

Three distinct scheduler implementations with different batching strategies.

---

## Scheduler Architectures

### static_fifo

```
Requests → Single FIFO Queue → Fixed Batch (B=8) → GPU → Completed
```

- Fixed batch size of 8 requests
- No adaptation to load or SLA
- No binning - mixes short and long requests

### dynamic_no_bins

```
Requests → Single FIFO Queue → Dynamic Batcher → GPU → Completed
                                     ↓
                              Feedback Loop
                            (SLA Controller)
```

- **Algorithm 1**: Memory constraint `b_mem = ⌊(η-L₀)/μ⌋`
- **Algorithm 2**: SLA controller with adaptive `[b_low, b_high]` search
- **Final**: `b_target = min(b_mem, b_SLA)`

### multi_bin_dynamic (Our Contribution)

```
Requests → Multi-Bin Scheduler → Bin Selection → Dynamic Batcher → GPU
               ↓                                        ↓
          Bin 0: [0, L1]                         Feedback Loop
          Bin 1: [L1, L2]                    (Per-Bin SLA Controller)
          Bin 2: [L2, L3]
          ...
          Bin K: [LK, ∞]
```

**Key Innovations**:

1. **Composition Control**: Bins partition requests by predicted output length
   - Reduces E[max(t_j) | bin] via narrower distributions
   - Better batch composition efficiency

2. **Bin-Specific Adaptation**: Each bin has separate controllers
   - **Per-bin statistics**: avg_prompt_len, avg_output_len
   - **Per-bin SLA controllers**: Independent batch size adaptation
   - Short requests (Bin 0): Can use larger batches
   - Long requests (Bin K): Needs smaller batches

3. **Mathematical Foundation**:
   - Throughput_k = B / E[T_batch,k] increases with k bins
   - max(B jobs from [10, 20]) << max(B jobs from [10, 200])

---

## Multi-Bin Key Insight

### What Multi-Bin Changes

**NOT the ordering** - still FIFO within bins

**WHAT CHANGES**: 
1. **Batch composition** (who gets batched together)
2. **Bin-specific adaptation** (each bin learns its characteristics)

### Example: Composition Control

**Without bins** (single queue):
```
Queue: [1 token, 100 tokens, 2 tokens, 3 tokens]
Batch: [1, 100, 2, 3]
→ Batch time dominated by 100 (longest)
→ Throughput = 4 / 100 = 0.04 req/time
```

**With bins** (partitioned):
```
Bin 0 (0-64):   [1, 2, 3, 5]
Bin 1 (64-256): [100, 150, 200]

Batch from Bin 0: [1, 2, 3, 5]
→ Batch time = 5 (max in bin)
→ Throughput = 4 / 5 = 0.80 req/time
```

**Result**: 20x better throughput!

---

## Configuration

### Main Config (`mb_dyn_sim/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_GPUS` | 4 | Number of GPUs |
| `K_BINS` | 4 | Number of bins |
| `D_SLA` | 0.05 | Per-token TBT SLA (50ms) |
| `B_MIN` | 1 | Minimum batch size |
| `B_MAX` | 128 | Maximum batch size |
| `M_MAX_GB` | 12.0 | GPU memory capacity |
| `USE_REAL_TIMESTAMPS` | False | Use real or scaled timestamps |
| `RPS_SCALING` | 200.0 | RPS scaling factor |

### Bin Boundaries

Default bin boundaries (auto-computed via equal-mass):
```python
BIN_BOUNDARIES = [
    (0, 64),      # Bin 0: short outputs
    (64, 256),    # Bin 1: medium outputs  
    (256, 1024),  # Bin 2: long outputs
    (1024, 10000) # Bin 3: very long outputs
]
```

---

## Scheduler Comparison

| Aspect | static_fifo | dynamic_no_bins | multi_bin_dynamic |
|--------|-------------|-----------------|-------------------|
| **Queue Structure** | 1 FIFO | 1 FIFO | K FIFO bins |
| **Batch Sizing** | Fixed (B=8) | Adaptive | **Bin-specific adaptive** |
| **Composition** | Uncontrolled | Uncontrolled | **Controlled** |
| **Statistics** | None | Global | **Per-bin** |
| **SLA Control** | No | Global | **Per-bin** |
| **Memory Awareness** | No | Global avg | **Bin-specific avg** |

---

## Running Experiments

### Quick Comparison
```bash
python scripts/run_mb_dynamic.py --compare --num-requests 1000
```

### Comprehensive Stress Test
```bash
# All 3 steps: request scaling, GPU scaling, K-bins sensitivity
python scripts/comprehensive_stress_test_optimized.py

# With real timestamps (realistic benchmarking)
python scripts/comprehensive_stress_test_optimized.py --use-real-timestamps
```

### Individual Schedulers
```bash
python scripts/run_mb_dynamic.py --scheduler static_fifo --num-requests 1000
python scripts/run_mb_dynamic.py --scheduler dynamic_no_bins --num-requests 1000
python scripts/run_mb_dynamic.py --scheduler multi_bin_dynamic --num-requests 1000
```

---

## Code Entry Points

| File | Purpose |
|------|---------|
| `mb_dyn_sim/config.py` | Configuration parameters |
| `mb_dyn_sim/workload.py` | Workload generation |
| `mb_dyn_sim/schedulers.py` | Scheduler implementations |
| `mb_dyn_sim/simulation.py` | Discrete-event simulator |
| `mb_dyn_sim/metrics.py` | Performance metrics |
| `mb_dyn_sim/model_calibration.py` | Latency model |
| `scripts/run_mb_dynamic.py` | Main experiment runner |
| `scripts/comprehensive_stress_test_optimized.py` | Stress test suite |

---

## References

### Papers
1. Multi-Bin Batching for LLM Inference Throughput Optimization
2. Memory-Aware and SLA-Constrained Dynamic Batching for LLM Inference

### Dataset
- **BurstGPT**: Real ChatGPT/GPT-4 workload traces from Azure
- Download: `python scripts/download_burstgpt.py --version 1`

### Model
- **Qwen3-1.7B**: Latency calibrated for RTX 4080 12GB
- Calibration file: `data/qwen3_1_7b_latency_grid.csv`

---

*Last Updated: November 2025*
