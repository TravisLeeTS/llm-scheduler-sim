# Claims and Paper Fidelity Assessment

**Date:** November 20, 2025  
**Status:** Based on professor feedback and validation testing

---

## ✅ What We Can Confidently Claim

### 1. Component-Level Paper Faithfulness

**Multi-Bin Batching (Guldogan et al.)**
- ✅ "We implement the Multi-Bin Batching algorithm with K separate FIFO queues indexed by predicted output length"
- ✅ "Bin boundaries computed using equal-mass partitioning via empirical quantiles (Lemma 4.1)"
- ✅ "Requests assigned to bins based on predicted output length; bins served via round-robin or longest-queue policy"
- ✅ "K-sensitivity analysis showing throughput behavior for K ∈ {1, 2, 4, 8}"

**SLA-Constrained Dynamic Batching**
- ✅ "We implement a two-layer dynamic batcher: memory-limited b_mem and SLA-feedback b_SLA, with final batch size min(b_mem, b_SLA)"
- ✅ "Memory constraint based on KV-cache capacity and token estimates, following Algorithm 1's design"
- ✅ "SLA feedback controller maintains adaptive batch size window [b_low, b_high], adjusting based on observed latency"
- ⚠️ "Controller follows Algorithm 2's conceptual design but uses a window-based heuristic rather than the exact update rule"

**Discrete-Event Simulation**
- ✅ "Proper queueing simulator with arrival events, GPU service completion events, per-request latency tracking"
- ✅ "Configurable arrival processes: Poisson, BurstGPT-like ON/OFF, trace-driven"
- ✅ "Parametric latency model T(b,L) = α + β·L·(1 + γ·(b-1)/b) fitted from GPU measurements or synthetic"

### 2. System Validation

**Distinct Policy Behavior**
- ✅ "Three scheduler modes produce statistically distinct performance under identical workloads"
- ✅ "Dynamic batching reduces SLA violations by ~20% vs static fixed-batch baseline"
- ✅ "Multi-bin + dynamic hybrid reduces violations by 63% vs dynamic-only under realistic workloads"

**Realistic Evaluation**
- ✅ "Evaluated on BurstGPT-like bursty arrival patterns with ON/OFF traffic"
- ✅ "GPU-calibrated latency model fitted from real RTX 4080 measurements (R²=1.0)"
- ✅ "Multi-GPU extension (N=1,2,4) showing scalability beyond single-server theory"

### 3. Research Contribution

**Novel Hybrid Policy**
- ✅ "The `multi_bin_dynamic` scheduler combines Multi-Bin binning with SLA-constrained dynamic batching"
- ✅ "Neither original paper analyzes this combination—it is our research contribution"
- ✅ "Demonstrates that binning by predicted length significantly improves SLA compliance when combined with dynamic batching"

---

## ⚠️ What We Should NOT Claim (Without Qualification)

### 1. Exact Paper Reproduction

**Multi-Bin Theory**
- ❌ "We reproduce the Multi-Bin paper's theoretical results"
  - **Why:** Paper assumes M/G/1 queue (single server), Poisson arrivals, U(l_min, l_max) service times, fixed B
  - **We use:** Multi-GPU, BurstGPT-like arrivals, fitted T(b,L) model, variable batch sizes

**Better phrasing:**
> "We implement the Multi-Bin algorithm faithfully but evaluate it under more realistic multi-GPU, bursty workload conditions that extend beyond the paper's theoretical M/G/1 analysis"

### 2. Algorithm-Level Exactness

**SLA Controller**
- ❌ "We implement Algorithm 2 exactly as specified"
  - **Why:** We use a heuristic window adjustment, not the exact gradient-based step rule

**Better phrasing:**
> "We implement a feedback SLA controller following Algorithm 2's design (same control variables, objective, and update structure) but with a simplified window-based heuristic"

### 3. SLA Definition

**End-to-End vs Decode-Only**
- ❌ "Our SLA violations directly compare to the paper's reported rates"
  - **Why:** Paper focuses on decode-phase latency; we measure queueing + service (stricter)

**Better phrasing:**
> "We measure SLA violations on end-to-end latency (queueing + service), which is stricter and more practical than the decode-only SLA in the original paper"

### 4. Capacity Analysis

**Single Load Point vs Capacity Curves**
- ❌ "We achieve a 22% capacity improvement like the SLA paper"
  - **Why:** We test at single load levels, not sweep arrival rate to find capacity frontier

**Better phrasing:**
> "At a fixed high load, multi-bin+dynamic achieves 63% lower SLA violations than dynamic-only. To quantify capacity gains (as in the original paper), we would need to sweep arrival rates and find the maximum throughput at 1% violation threshold"

---

## 🔧 What Would Be Needed for Full Paper Reproduction

### Multi-Bin Paper (Guldogan et al.)

**"Theory Validation Mode"**
```python
cfg = SchedulerConfig(
    NUM_GPUS=1,                    # Single server (M/G/1)
    K_BINS=k,                      # Variable K ∈ {1,2,4,8}
    ARRIVAL_PROFILE="poisson",     # Poisson arrivals
    SERVICE_DIST="uniform",        # U(l_min, l_max) service times
    B_FIXED=8,                     # Fixed batch size B
    EXPERIMENT_MODE="multi_bin_only"
)
```

**Required Experiment:**
1. Run for K ∈ {1, 2, 4, 8}
2. Measure throughput λ_max for each K
3. Plot throughput vs K and compare shape to Theorem 4.2
4. Show diminishing returns as K increases (as predicted by theory)

**Expected Result:**
- Monotonic increase in throughput with K
- Curve shape matching theoretical predictions (not exact numbers, but qualitative trend)

### SLA-Constrained Paper

**"Capacity Frontier Experiment"**
```python
# For each policy
for policy in ["static_fifo", "dynamic_no_bins"]:
    for arrival_rate in [10, 20, 30, ..., 100]:
        run_simulation(lambda=arrival_rate)
        measure_sla_violation_rate()
    
    # Find max λ where violations < 1%
    capacity = find_threshold(target_violation=0.01)
```

**Required Experiment:**
1. Sweep arrival rate λ from low to saturation
2. Find capacity (max λ) for 1% SLA violation threshold
3. Compare capacity_dynamic / capacity_static
4. Report percentage capacity gain

**Expected Result:**
- Dynamic batching: 15-25% capacity improvement (paper reports 22%)
- Multi-bin+dynamic: potentially higher (novel contribution)

### Exact Algorithm 2 Implementation

**Optional for completeness:**
```python
class ExactAlgorithm2Controller:
    """Exact line-by-line implementation of Algorithm 2 from paper"""
    
    def update(self, tau_obs, b_prev):
        # Use paper's exact gradient step rule
        # Include all hyperparameters (α, δ, learning rate)
        # Match termination conditions exactly
        ...
```

**Validation:**
- Show heuristic and exact controllers converge to similar batch sizes
- Demonstrates our simplification doesn't change qualitative behavior

---

## 📊 Current Validation Status Summary

| Component | Implementation | Evaluation Setup | Status |
|-----------|---------------|-----------------|--------|
| Multi-Bin binning | ✅ Paper-faithful | Multi-GPU, BurstGPT | ✅ Component validated |
| Equal-mass bins | ✅ Quantile-based | Empirical distribution | ✅ Lemma 4.1 compliant |
| SLA controller | ⚠️ Heuristic variant | End-to-end latency | ⚠️ Conceptually aligned |
| Dynamic batching | ✅ b_mem + b_SLA | Multi-GPU | ✅ Algorithm 1/2 design |
| Latency model | ✅ Fitted T(b,L) | RTX 4080 data | ✅ R²=1.0 fit |
| Workload | ✅ BurstGPT-like | Synthetic + trace | ✅ Realistic bursty |
| Hybrid policy | ✅ Novel combination | Multi-GPU | ✅ Research contribution |
| Theory reproduction | ❌ Not attempted | N/A | 🔲 Future work |

**Legend:**
- ✅ Fully validated
- ⚠️ Partially validated / simplified
- ❌ Not implemented
- 🔲 Planned / optional

---

## 🎓 Recommended Thesis/Report Phrasing

### Introduction / Claims

> "We implement paper-faithful versions of Multi-Bin Batching (Guldogan et al.) and SLA-constrained Dynamic Batching, combining them into a novel hybrid scheduler. While our individual components follow the papers' algorithmic designs, we evaluate the hybrid policy under more realistic conditions: multi-GPU systems, bursty BurstGPT-like workloads, and GPU-calibrated latency models. Our results demonstrate that the combination reduces SLA violations by 63% compared to dynamic batching alone under high load."

### Methods Section

> "Our MultiBinScheduler implements K separate FIFO queues with equal-mass bin boundaries computed via empirical quantiles (following Lemma 4.1 from Guldogan et al.). The DynamicBatcher implements a two-layer control mechanism: a memory-aware constraint b_mem and an SLA feedback controller that maintains an adaptive batch size window, following the conceptual design of Algorithm 1 and Algorithm 2 from the SLA-constrained batching paper. Our controller uses a simplified window-based heuristic rather than the exact gradient step rule for practical efficiency."

### Results / Discussion

> "We evaluate three policies (static FIFO, dynamic-only, multi-bin+dynamic) under identical workloads to ensure fair comparison. While our setup extends beyond the original papers' single-server Poisson assumptions, the relative performance rankings remain valid: dynamic batching provides a 20% improvement in SLA compliance over static, and our hybrid multi-bin approach provides an additional 63% reduction in violations. These results demonstrate the practical benefits of combining length-aware binning with adaptive batch sizing in multi-GPU LLM serving systems."

---

## ✍️ Bottom Line Summary for Professor

**What is solid:**
1. ✅ Component implementations are algorithmically faithful
2. ✅ System actually works and shows distinct policy behavior
3. ✅ Validation demonstrates clear performance differences
4. ✅ GPU calibration pipeline functional with R²=1.0 fit
5. ✅ Hybrid policy is a genuine research contribution

**What needs careful wording:**
1. ⚠️ "Paper-faithful components, extended evaluation" not "exact reproduction"
2. ⚠️ "Algorithm 2-inspired" not "Algorithm 2 implementation"
3. ⚠️ "End-to-end SLA" not "decode-phase SLA"
4. ⚠️ Multi-GPU and BurstGPT workloads are realistic extensions, not theory validation

**What would strengthen to "full reproduction":**
1. 🔲 Theory mode: single GPU, Poisson, uniform service, fixed B
2. 🔲 Capacity curves: sweep λ and find threshold
3. 🔲 Optional: exact Algorithm 2 side-by-side comparison

**Current positioning:**
> "Paper-faithful building blocks, novel composition, realistic evaluation beyond original theory scope"

This is honest, defensible, and scientifically valid for systems research.

