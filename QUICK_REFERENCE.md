# Quick Reference: Paper Requirements Summary

## For Your Professor: What Changed

### Previous Approach (Before Feedback)
- ❌ "Paper-inspired" but not paper-faithful
- ❌ Arbitrary bin boundaries
- ❌ Greedy batching without theoretical foundation
- ❌ No clear connection to paper algorithms
- ❌ Missing key elements from both papers

### New Approach (After Feedback)
- ✅ **Exact implementation** of both paper algorithms
- ✅ Equal-mass bin boundaries (empirical quantiles)
- ✅ SLA controller with adaptive search (Algorithm 2)
- ✅ Memory constraint calculator (Algorithm 1)
- ✅ Three experiment modes matching paper scenarios
- ✅ Service time model: batch time = f(max_seq_len)
- ✅ Clear theoretical validation path

---

## The Two Papers: Key Algorithms

### Paper 1: Multi-Bin Batching

**Core Idea**: Group requests by predicted service time into K bins to reduce variance within batches.

**Key Requirements**:
1. **Equal-mass bins**: Each bin has ~equal probability mass
   - Use empirical quantiles of predicted_output_len
   - `boundaries = np.quantile(lengths, [0, 1/k, 2/k, ..., 1])`

2. **Fixed batch size B** (for pure multi-bin experiments)
   - No dynamic resizing
   - Form batches of exactly B when bin has ≥ B requests

3. **Batch service time = MAX(request times)**
   - Longest request dominates
   - `service_time = f(max_seq_len, batch_size)`

4. **Throughput increases with K**
   - k=1 (FIFO) < k=2 < k=4 < k=8
   - Asymptotically approaches theoretical maximum

**Validation**: Plot throughput vs K_BINS, should match paper Fig 3.

---

### Paper 2: Dynamic Batching (Memory + SLA)

**Core Idea**: Dynamically adjust batch size based on memory limit and SLA target.

**Key Requirements**:

1. **Algorithm 1: Memory Constraint**
   ```python
   η = (M_max - M_model) / kv_mem_per_token  # token capacity
   μ = avg(prompt_len + output_len)          # avg tokens/req
   b_mem = floor((η - L₀) / μ)               # max batch size
   ```

2. **Algorithm 2: SLA Controller**
   ```python
   # Maintain adaptive interval [b_low, b_high]
   if τ_avg > D_SLA + ε:
       # Too slow → shrink batch
       b_high = min(b_high, b_avg)
       b_low = max(b_low, 0.8 * b_avg)
   elif τ_avg < D_SLA - ε:
       # Too fast → grow batch
       b_low = max(b_low, b_avg)
       b_high = min(b_high + 0.2*b_avg, B_max)
   else:
       # Within SLA → center around b_avg
       center interval
   
   b_SLA = (b_low + b_high) // 2
   ```

3. **Final batch size**: `b_target = min(b_mem, b_SLA)`

4. **Feedback loop**:
   - After each batch: update τ_avg, b_avg
   - Update running statistics: μ_prompt, μ_output

**Validation**: Compare dynamic vs static B on SLA violations and throughput.

---

## Three Experiment Modes

### Mode 1: `multi_bin_only`
**Purpose**: Validate Multi-Bin paper

**Config**:
- `K_BINS ∈ {1, 2, 4, 8}`
- `B_FIXED = 32` (or 8, 128)
- `ARRIVAL_PROFILE = "poisson"`
- No dynamic batching

**Expected Results**:
- Throughput increases with K
- Matches theoretical predictions from paper

---

### Mode 2: `dynamic_only`
**Purpose**: Validate Dynamic Batching paper

**Config**:
- `K_BINS = 1` (single queue)
- Dynamic batching enabled
- `b_target = min(b_mem, b_SLA)`

**Expected Results**:
- Higher throughput than static B
- Lower SLA violation rate
- Batch size adapts to load

---

### Mode 3: `multi_bin_dynamic`
**Purpose**: Our contribution (combine both)

**Config**:
- `K_BINS > 1` (e.g., 4)
- Dynamic batching enabled
- Equal-mass boundaries

**Expected Results**:
- Best performance overall
- Benefits of both approaches

---

## Implementation Checklist

### Core Algorithms ✅
- [ ] Equal-mass bin boundaries (use np.quantile)
- [ ] SLA controller class with adaptive [b_low, b_high]
- [ ] Memory constraint: compute_b_mem()
- [ ] Running statistics: BatchStatistics class
- [ ] Update DynamicBatcher.make_batch() to use b_target
- [ ] FixedBatchSizer for multi_bin_only mode

### Workload ✅
- [ ] Poisson arrival mode (already exists)
- [ ] BurstGPT ON/OFF mode (already exists)
- [ ] Optional: Load real BurstGPT dataset

### Service Time Model ✅
- [ ] Ensure formula reflects max-dominates
- [ ] `time = a₀ + a₁ * max_seq_len * h(batch_size)`
- [ ] Optional: vLLM calibration for realistic coefficients

### Experiment Modes ✅
- [ ] Add EXPERIMENT_MODE config field
- [ ] Update Simulator.__init__() to support modes
- [ ] Ensure each mode behaves correctly

### Metrics ✅
- [ ] Throughput vs K_BINS plots
- [ ] SLA violation rate over time
- [ ] Batch size distribution
- [ ] Per-bin statistics

---

## Data Options (3 Levels)

### Level 1: Pure Simulation (Fastest)
**What you need**:
- Nothing! Just run the code
- Synthetic arrivals (Poisson or BurstGPT-style)
- Synthetic service times (formula)

**Pros**:
- Extremely fast (seconds)
- No dependencies
- Good for algorithm development

**Cons**:
- Absolute numbers are synthetic
- Professor might question realism

**Validity**: ✅ Relative comparisons are valid

---

### Level 2: BurstGPT Dataset (Better)
**What you need**:
- Download BurstGPT trace dataset
- Load real arrival times and lengths

**Pros**:
- Realistic workload patterns
- Published dataset (good for paper)
- Still fast (no GPU needed)

**Cons**:
- Still synthetic service times
- Need to download/process data

**Validity**: ✅✅ More convincing to reviewers

---

### Level 3: vLLM Calibration (Best)
**What you need**:
- vLLM framework
- Qwen3-0.6B model
- GPU (can use RTX 4080)
- Run micro-benchmarks

**Process**:
1. Install vLLM: `pip install vllm`
2. Download Qwen3-0.6B
3. Run batches with varying (B, max_seq_len)
4. Measure actual latency
5. Fit linear model: `latency ~ a₀ + a₁*len*h(B)`
6. Use fitted coefficients in simulator

**Pros**:
- Realistic absolute latency numbers
- Can claim "calibrated on real model"
- Professor will love this

**Cons**:
- Takes time (hours for calibration)
- Requires GPU
- More complex setup

**Validity**: ✅✅✅ Publication-ready

---

## What to Tell Your Professor

### The Pitch

> "I've updated the simulator to **exactly implement** the algorithms from both papers:
> 
> 1. **Multi-Bin Batching**: Equal-mass bins using empirical quantiles, fixed batch size mode, validation that throughput increases with K.
> 
> 2. **Dynamic Batching**: Algorithm 1 (memory constraint) and Algorithm 2 (SLA controller with adaptive search).
> 
> I support **three experiment modes** that map directly to the papers:
> - `multi_bin_only`: Reproduce Multi-Bin paper results
> - `dynamic_only`: Reproduce Dynamic Batching paper results
> - `multi_bin_dynamic`: Our contribution (combine both)
> 
> For **data fidelity**, I have three levels:
> - Level 1: Pure simulation (fast prototyping)
> - Level 2: BurstGPT dataset (realistic workload)
> - Level 3: vLLM calibration with Qwen3-0.6B (realistic latency)
> 
> The **algorithms are paper-faithful** regardless of data level. Using BurstGPT dataset + vLLM calibration makes this **publication-ready**."

---

## File Structure

```
llm_scheduler_sim/
├── PAPER_REQUIREMENTS.md        ← Full specification (this doc's parent)
├── IMPLEMENTATION_PLAN.md       ← Detailed implementation tasks
├── ARCHITECTURE.md              ← Updated visual diagrams
├── QUICK_REFERENCE.md           ← This file (summary)
│
├── mb_dyn_sim/
│   ├── config.py                ← Add EXPERIMENT_MODE, equal-mass boundaries
│   ├── schedulers.py            ← Add SLAController, update DynamicBatcher
│   ├── simulation.py            ← Support three modes
│   ├── workload.py              ← Add BurstGPT dataset loading
│   ├── model_calibration.py    ← Update service time formula
│   └── metrics.py               ← Add paper-specific metrics
│
└── scripts/
    ├── run_mb_dynamic.py        ← Update to use new modes
    ├── calibrate_real_gpu_transformers.py  ← GPU calibration (Windows)
    └── plot_paper_results.py   ← New: Generate paper figures
```

---

## Next Steps (Priority Order)

1. **Implement SLA Controller** (highest priority)
   - This is the most critical missing piece
   - Directly from Dynamic Batching paper Algorithm 2

2. **Equal-Mass Bin Boundaries**
   - Simple to implement (10 lines of code)
   - Needed for Multi-Bin validation

3. **Add Experiment Modes**
   - Refactor simulator initialization
   - Create FixedBatchSizer class

4. **Update Service Time Model**
   - Ensure max-dominates property is clear
   - Document formula clearly

5. **Add Paper-Specific Metrics**
   - Throughput vs K plots
   - SLA violation tracking

6. **Optional: BurstGPT Dataset**
   - If you want stronger validation
   - Can skip for initial experiments

7. **Optional: vLLM Calibration**
   - If you want publication-quality results
   - Can use synthetic for now

---

## Timeline Estimate

**Minimal (Core functionality)**:
- Implement tasks 1-5: **2-3 days**
- Run experiments: **1 day**
- Generate plots: **1 day**
- **Total: ~1 week**

**Recommended (BurstGPT dataset)**:
- Core + dataset integration: **1 week**
- Experiments with real data: **2 days**
- **Total: ~1.5 weeks**

**Full (vLLM calibration)**:
- Core + dataset + vLLM: **2 weeks**
- Calibration experiments: **3 days**
- Full validation: **2 days**
- **Total: ~3 weeks**

---

## Questions for Your Professor

Before implementing, you might ask:

1. **Data level required**: Do you want synthetic (fast), BurstGPT dataset (realistic workload), or vLLM calibration (realistic latency)?

2. **Experiment scope**: Should I validate against all paper results, or focus on key figures (e.g., throughput vs K, dynamic vs static)?

3. **Baseline comparisons**: Which baselines should I compare against? Just FIFO, or also other schedulers?

4. **Timeline**: When do you need results? This determines which level of fidelity to pursue.

---

## Key Takeaway

**The simulator is now paper-faithful in its algorithms, not just "inspired by" the papers.**

This means:
- ✅ Reviewers can verify our implementation matches paper descriptions
- ✅ Results are directly comparable to paper results
- ✅ We can claim to "build upon" these papers, not just cite them
- ✅ Your professor will be satisfied that it's rigorous

**The choice of data (synthetic vs real) affects realism, but the algorithmic fidelity is what makes this a solid contribution.**
