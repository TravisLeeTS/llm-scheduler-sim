# Algorithm Specifications

## Document Overview

This document provides formal algorithm specifications and pseudocode for all scheduling algorithms implemented in the Multi-Bin Dynamic Batching system.

**Date**: November 30, 2025  
**Version**: 1.0

---

## Table of Contents

1. [Algorithm Summary](#1-algorithm-summary)
2. [Algorithm 1: Memory-Constrained Batch Size](#2-algorithm-1-memory-constrained-batch-size)
3. [Algorithm 2: SLA-Constrained Batch Size](#3-algorithm-2-sla-constrained-batch-size)
4. [Algorithm 3: Dynamic Batch Formation](#4-algorithm-3-dynamic-batch-formation)
5. [Algorithm 4: Multi-Bin Request Partitioning](#5-algorithm-4-multi-bin-request-partitioning)
6. [Algorithm 5: Multi-Bin Batch Formation](#6-algorithm-5-multi-bin-batch-formation)
7. [Algorithm 6: Equal-Mass Bin Boundaries](#7-algorithm-6-equal-mass-bin-boundaries)
8. [Algorithm 7: SLA Controller Feedback](#8-algorithm-7-sla-controller-feedback)

---

## 1. Algorithm Summary

| Algorithm | Purpose | Input | Output |
|-----------|---------|-------|--------|
| **Algorithm 1** | Memory constraint | Stats, Config | b_mem |
| **Algorithm 2** | SLA constraint | Controller state | b_SLA |
| **Algorithm 3** | Dynamic batching | Candidates, Constraints | Batch |
| **Algorithm 4** | Request partitioning | Request, Boundaries | Bin index |
| **Algorithm 5** | Multi-bin batching | K bins, GPU | Batch from one bin |
| **Algorithm 6** | Bin computation | Output lengths, K | K boundaries |
| **Algorithm 7** | Feedback loop | Batch metrics | Updated state |

---

## 2. Algorithm 1: Memory-Constrained Batch Size

### Purpose
Compute the maximum batch size that fits in GPU memory without out-of-memory (OOM) errors.

### Mathematical Foundation

**Memory Model (KV Cache)**:
$$S = \sum_{i \in batch}(l_{in,i} + l_{out,i})$$

**Expected Total Tokens**:
$$E[S] = b_t \cdot E[l_{in} + l_{out}]$$

**Probabilistic Constraint**:
$$Pr(S > \eta) \leq \epsilon_M$$

Where $\eta$ is GPU token capacity.

### Pseudocode

```
ALGORITHM 1: ComputeMemoryConstrainedBatchSize

Input:
    stats: BatchStatistics (avg_prompt_len, avg_output_len)
    cfg: SchedulerConfig (M_MAX_GB, M_MODEL_GB, KV_MEM_PER_TOKEN_GB, B_MAX, B_MIN)
    bin_idx: int (-1 for global, ≥0 for specific bin)

Output:
    b_mem: int (memory-constrained batch size)

BEGIN
    // Step 1: Compute token capacity from GPU memory
    η ← (M_MAX_GB - M_MODEL_GB) / KV_MEM_PER_TOKEN_GB
    
    // Step 2: Compute expected tokens per request
    E_l_total ← stats.avg_prompt_len + stats.avg_output_len
    
    // Safety: use conservative default if no stats
    IF E_l_total ≤ 0 THEN
        E_l_total ← 500.0
    END IF
    
    // Step 3: Compute safety buffer L₀ (ρ = 10% of capacity)
    // Corresponds to ε_M ≈ 0.05 (5% OOM probability)
    L₀ ← 0.1 × η
    
    // Step 4: Compute memory-constrained batch size
    b_mem ← FLOOR((η - L₀) / E_l_total)
    
    // Step 5: Apply per-bin batch limit if available
    IF bin_idx ≥ 0 AND cfg.BIN_B_MAX exists THEN
        bin_b_max ← cfg.BIN_B_MAX[bin_idx]
        b_mem ← MIN(b_mem, bin_b_max)
    END IF
    
    // Step 6: Clamp to valid range
    b_mem ← CLAMP(b_mem, cfg.B_MIN, cfg.B_MAX)
    
    RETURN b_mem
END
```

### Complexity
- Time: O(1)
- Space: O(1)

---

## 3. Algorithm 2: SLA-Constrained Batch Size

### Purpose
Compute the batch size that maintains SLA compliance using adaptive search.

### Mathematical Foundation

**SLA Constraint**:
$$\tau_{avg} \leq D_{SLA} + \epsilon_D$$

Where $\tau_{avg}$ is the exponential moving average of per-token decode TBT.

### Pseudocode

```
ALGORITHM 2: ComputeSLAConstrainedBatchSize

Input:
    controller: SLAController with state:
        - b_low, b_high: adaptive interval bounds
        - τ_avg: EMA of per-token decode TBT
        - b_avg: EMA of batch size
        - D_SLA, ε_D: SLA threshold and tolerance
        - B_min, B_max: global bounds
        - N_decode: decode requests in system

Output:
    b_SLA: int (SLA-constrained batch size)

Constants:
    α ← 4   // Expansion/contraction step
    δ ← 2   // Small correction step

BEGIN
    // Warmup: use midpoint during initial learning
    IF τ_avg = 0 OR update_count < 3 THEN
        RETURN FLOOR((b_low + b_high) / 2)
    END IF
    
    // Case 1: Latency too high → shrink interval to left
    IF τ_avg > D_SLA + ε_D THEN
        new_b_high ← MAX(FLOOR(b_avg), b_low + α)
        new_b_low ← MAX(b_low - δ, B_min)
        b_high ← MIN(b_high, new_b_high)  // Only shrink, never expand
        b_low ← new_b_low
        
    // Case 2: Latency comfortably below SLA → expand interval to right
    ELSE IF τ_avg < D_SLA - ε_D THEN
        new_b_low ← MIN(FLOOR(b_avg), b_high - α)
        new_b_high ← MIN(b_high + δ, B_max)
        b_low ← MAX(b_low, new_b_low)  // Only shift right
        b_high ← new_b_high
        
    // Case 3: SLA approximately satisfied → center around average
    ELSE
        half_α ← FLOOR(α / 2)
        b_high ← MIN(FLOOR(b_avg) + half_α, B_max)
        b_low ← MAX(FLOOR(b_avg) - half_α, B_min)
    END IF
    
    // Ensure valid interval
    b_low ← MAX(B_min, b_low)
    b_high ← MIN(B_max, b_high)
    IF b_low > b_high THEN
        b_low ← b_high
    END IF
    
    // Compute midpoint (paper algorithm)
    b_SLA ← FLOOR((b_low + b_high) / 2)
    
    // Paper requirement: b_SLA ≥ N_decode
    IF N_decode > 0 THEN
        b_SLA ← MAX(b_SLA, N_decode)
    END IF
    
    // Final clamp
    b_SLA ← CLAMP(b_SLA, B_min, B_max)
    
    RETURN b_SLA
END
```

### Complexity
- Time: O(1)
- Space: O(1) per controller

---

## 4. Algorithm 3: Dynamic Batch Formation

### Purpose
Form a batch from candidates using combined memory and SLA constraints.

### Pseudocode

```
ALGORITHM 3: FormDynamicBatch

Input:
    candidates: List[Request] (sorted by arrival time)
    stats: BatchStatistics
    sla_controller: SLAController
    cfg: SchedulerConfig
    bin_idx: int (for bin-specific constraints)
    service_time_fn: (batch_size, max_seq_len) → float

Output:
    batch: List[Request]
    service_time: float

BEGIN
    IF candidates is EMPTY THEN
        RETURN [], 0.0
    END IF
    
    // Step 1: Compute target batch size
    b_mem ← ALGORITHM_1(stats, cfg, bin_idx)
    b_SLA ← ALGORITHM_2(sla_controller)
    b_target ← MIN(b_mem, b_SLA)
    
    // Step 2: Take first b_target candidates (FIFO order)
    batch ← candidates[0:b_target]
    
    // Step 3: Verify memory constraint (safety check)
    WHILE batch is NOT EMPTY AND NOT check_memory_constraint(batch, cfg) DO
        batch.pop()
    END WHILE
    
    IF batch is EMPTY THEN
        RETURN [], 0.0
    END IF
    
    // Step 4: Compute service time
    max_seq_len ← MAX(r.prompt_len + r.output_len for r in batch)
    service_time ← service_time_fn(|batch|, max_seq_len)
    
    RETURN batch, service_time
END
```

### Complexity
- Time: O(b_target)
- Space: O(b_target)

---

## 5. Algorithm 4: Multi-Bin Request Partitioning

### Purpose
Assign incoming requests to appropriate bins based on predicted output length.

### Pseudocode

```
ALGORITHM 4: PartitionRequestToBin

Input:
    request: Request with predicted_output_len
    boundaries: List[(min_len, max_len)] of K bins

Output:
    bin_idx: int (0 to K-1)

BEGIN
    predicted_len ← request.predicted_output_len
    
    FOR i ← 0 TO K-1 DO
        (min_len, max_len) ← boundaries[i]
        
        IF min_len ≤ predicted_len < max_len THEN
            RETURN i
        END IF
    END FOR
    
    // Fallback: put in last bin
    RETURN K - 1
END
```

### Complexity
- Time: O(K) worst case, O(log K) with binary search
- Space: O(1)

---

## 6. Algorithm 5: Multi-Bin Batch Formation

### Purpose
Select a bin and form a batch from it, applying bin-specific constraints.

### Pseudocode

```
ALGORITHM 5: FormMultiBinBatch

Input:
    bins: List[Deque[Request]] of K bins
    bin_stats: List[BatchStatistics] of K statistics
    bin_controllers: List[SLAController] of K controllers
    cfg: SchedulerConfig
    max_candidates: int
    service_time_fn: (batch_size, max_seq_len) → float

Output:
    batch: List[Request]
    service_time: float
    bin_idx: int

BEGIN
    // Step 1: Select bin (round-robin or longest-queue)
    bin_idx ← SELECT_BIN_ROUND_ROBIN(bins)
    // Alternative: bin_idx ← SELECT_BIN_LONGEST_QUEUE(bins)
    
    IF bin_idx is NULL THEN
        RETURN [], 0.0, -1
    END IF
    
    // Step 2: Get candidates from selected bin ONLY
    candidates ← []
    num_to_fetch ← MIN(max_candidates, |bins[bin_idx]|)
    FOR i ← 1 TO num_to_fetch DO
        candidates.append(bins[bin_idx].dequeue())
    END FOR
    
    // Step 3: Apply bin-specific dynamic batching
    stats ← bin_stats[bin_idx]
    controller ← bin_controllers[bin_idx]
    
    batch, service_time ← ALGORITHM_3(
        candidates, stats, controller, cfg, bin_idx, service_time_fn
    )
    
    // Step 4: Return unused candidates to FRONT of bin queue (preserve FIFO)
    unused ← [c for c in candidates if c not in batch]
    FOR req IN REVERSED(unused) DO
        bins[bin_idx].prepend(req)
    END FOR
    
    RETURN batch, service_time, bin_idx
END


FUNCTION SELECT_BIN_ROUND_ROBIN(bins):
    // Try each bin starting from current_bin_index
    FOR offset ← 0 TO K-1 DO
        idx ← (current_bin_index + offset) MOD K
        IF |bins[idx]| > 0 THEN
            current_bin_index ← (idx + 1) MOD K
            RETURN idx
        END IF
    END FOR
    RETURN NULL
END FUNCTION


FUNCTION SELECT_BIN_LONGEST_QUEUE(bins):
    max_len ← MAX(|b| for b in bins)
    IF max_len = 0 THEN
        RETURN NULL
    END IF
    RETURN ARGMAX(|b| for b in bins)
END FUNCTION
```

### Complexity
- Time: O(K + max_candidates)
- Space: O(max_candidates)

---

## 7. Algorithm 6: Equal-Mass Bin Boundaries

### Purpose
Compute bin boundaries such that each bin has approximately equal probability mass (equal number of requests).

### Mathematical Foundation

Uses quantile-based partitioning:
$$boundary_i = Q_{i/K}(L)$$

Where $Q_p$ is the p-th quantile of output length distribution $L$.

### Pseudocode

```
ALGORITHM 6: ComputeEqualMassBinBoundaries

Input:
    predicted_lengths: List[int] of N predicted output lengths
    K: int (number of bins)

Output:
    boundaries: List[(min_len, max_len)] of K bins

BEGIN
    IF K = 1 THEN
        RETURN [(0, 10000)]
    END IF
    
    // Step 1: Compute quantiles
    quantiles ← LINSPACE(0, 1, K+1)  // [0, 1/K, 2/K, ..., 1]
    
    // Step 2: Find boundary points
    boundary_points ← QUANTILE(predicted_lengths, quantiles)
    
    // Step 3: Build boundary tuples
    boundaries ← []
    FOR i ← 0 TO K-1 DO
        min_len ← FLOOR(boundary_points[i])
        IF i < K-1 THEN
            max_len ← FLOOR(boundary_points[i+1])
        ELSE
            max_len ← ∞  // Last bin captures all remaining
        END IF
        boundaries.append((min_len, max_len))
    END FOR
    
    // Step 4: Ensure last bin captures everything
    boundaries[K-1] ← (boundaries[K-1][0], 10000)
    
    RETURN boundaries
END
```

### Example Output (K=4, 1M requests)

```
Bin 0: [1, 42) tokens
Bin 1: [42, 80) tokens
Bin 2: [80, 167) tokens
Bin 3: [167, 10000) tokens
```

Each bin contains ~25% of requests.

### Complexity
- Time: O(N log N) for quantile computation
- Space: O(N) for sorted copy

---

## 8. Algorithm 7: SLA Controller Feedback

### Purpose
Update controller state after batch completion to enable adaptive learning.

### Pseudocode

```
ALGORITHM 7: UpdateSLAControllerAfterBatch

Input:
    controller: SLAController
    stats: BatchStatistics
    batch: List[Request] (completed)
    service_time: float
    decode_tbt: float (optional, v2 model)
    N_decode: int

BEGIN
    IF batch is EMPTY THEN
        RETURN
    END IF
    
    // Step 1: Update running statistics (for Algorithm 1)
    batch_avg_prompt ← MEAN(r.prompt_len for r in batch)
    batch_avg_output ← MEAN(r.output_len for r in batch)
    
    α_ema ← 0.2  // EMA smoothing factor (fixed parameter)
    stats.avg_prompt_len ← α_ema × batch_avg_prompt + (1 - α_ema) × stats.avg_prompt_len
    stats.avg_output_len ← α_ema × batch_avg_output + (1 - α_ema) × stats.avg_output_len
    
    // Step 2: Compute recent TBT for SLA controller (per-request decode TBT)
    IF decode_tbt is PROVIDED THEN
        // v2 model: use decode-only TBT (excludes TTFT)
        recent_tbt ← decode_tbt
    ELSE
        // Legacy: total TBT (includes TTFT)
        max_output_len ← MAX(r.output_len for r in batch)
        recent_tbt ← service_time / max_output_len
    END IF
    
    // Step 3: Update SLA controller state
    controller.τ_avg ← α_ema × recent_tbt + (1 - α_ema) × controller.τ_avg
    controller.b_avg ← α_ema × |batch| + (1 - α_ema) × controller.b_avg
    controller.N_decode ← N_decode
    controller.update_count ← controller.update_count + 1
END
```

### v2 Model: Decode TBT Computation

```
FUNCTION ComputeDecodeTBT(latency_model, batch_size):
    IF latency_model is CALIBRATED THEN
        β ← latency_model.beta      // 5.74ms
        γ ← latency_model.gamma     // 0.316
        h_b ← 1.0 + γ × (batch_size - 1) / MAX(1, batch_size)
        RETURN β × h_b
    ELSE
        RETURN 0.00574  // Default 5.74ms
    END IF
END FUNCTION
```

### Complexity
- Time: O(|batch|) for mean computation
- Space: O(1)

---

## Algorithm Interaction Diagram

```
                    Request Arrival
                          │
                          ▼
            ┌─────────────────────────────┐
            │    Algorithm 4: Partition    │
            │    (Select bin by length)    │
            └─────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │         K Bins              │
            │  [Bin 0] [Bin 1] ... [Bin K]│
            └─────────────────────────────┘
                          │
                          │ GPU becomes free
                          ▼
            ┌─────────────────────────────┐
            │   Algorithm 5: Multi-Bin    │
            │   (Select bin, get batch)   │
            └─────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │   Algorithm 3: Dynamic      │
            │   Batch Formation           │
            │   ┌───────────────────────┐ │
            │   │ Algo 1: b_mem        │ │
            │   │ Algo 2: b_SLA        │ │
            │   │ b_target = min(both) │ │
            │   └───────────────────────┘ │
            └─────────────────────────────┘
                          │
                          ▼
            ┌─────────────────────────────┐
            │      GPU Processing         │
            │   (Batch execution)         │
            └─────────────────────────────┘
                          │
                          │ Batch completes
                          ▼
            ┌─────────────────────────────┐
            │   Algorithm 7: Feedback     │
            │   (Update stats, controller)│
            └─────────────────────────────┘
```

---

## References

- **Implementation**: `mb_dyn_sim/schedulers.py`
- **Configuration**: `mb_dyn_sim/config.py`
- **Simulation**: `mb_dyn_sim/simulation.py`

---

**Document Version**: 1.0  
**Last Updated**: November 30, 2025
