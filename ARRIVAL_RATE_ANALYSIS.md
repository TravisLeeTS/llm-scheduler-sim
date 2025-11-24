# BurstGPT Arrival Rate Analysis & Configuration Decision

**Date:** November 24, 2025  
**Analysis:** `timestamp_arrival_analysis.ipynb`  
**Decision:** Switch to RPS scaling by default for stress testing

---

## Executive Summary

Analysis of the full BurstGPT dataset (1.43M requests over 1,464 hours) reveals that the **real arrival rate is very low (~0.27 req/s)**. This creates a problem for scheduler evaluation: all three schedulers perform similarly under such low pressure, making it difficult to identify performance differences and breaking points.

**Solution:** Use RPS scaling 200x by default to create meaningful load (~54 req/s) for stress testing, while keeping real timestamps available for realistic benchmarking.

---

## Dataset Analysis Results

### Scale
- **Total Requests:** 1,429,737
- **Time Span:** 5,269,973 seconds (1,463.88 hours / 61 days)
- **Dataset:** Real Azure ChatGPT production traces from BurstGPT

### Inter-Arrival Time Statistics
```
Mean:     3.686 seconds
Median:   1.000 seconds
Std Dev:  48.957 seconds
Min:      0.000 seconds (simultaneous arrivals)
Max:      15,240 seconds (4.2 hour gap)
P95:      12 seconds
P99:      52 seconds
```

### Arrival Rate (Real Timestamps)
```
Average:  0.271 req/s  (16.3 req/min, 977 req/hour)
Peak:     20.8 req/s   (in 60-second windows)
Min:      0.0 req/s    (quiet periods)
```

### Burstiness Characteristics
```
Coefficient of Variation: 2.91 (highly variable)
Burst Periods (>mean+2σ):  3.4% of time
Quiet Periods (<mean-σ):   0.0% of time (never drops below threshold)
```

---

## The Problem: Low Pressure Doesn't Differentiate Schedulers

### Real Timestamps Results (0.27 req/s)

At such low arrival rates, **all schedulers perform well**:

| Scheduler | 1K Requests | 10K Requests | 100K Requests |
|-----------|-------------|--------------|---------------|
| **static_fifo** | 0.4% violations | 0.4% violations | 14.6% violations |
| **dynamic_no_bins** | 0.4% violations | 0.4% violations | 12.3% violations |
| **multi_bin_dynamic** | 0.1% violations | 0.2% violations | 1.7% violations |

**Observations:**
- ✅ At 1K-10K: All schedulers achieve <0.5% SLA violations
- ✅ GPU utilization is very low (0.5-2.3%)
- ⚠️ Schedulers only diverge at 100K+ requests
- ⚠️ Difficult to identify breaking points and limits

### Why This Happens

**Low arrival rate means:**
1. **No queue buildup** - Requests processed immediately
2. **No batching pressure** - Small batches work fine
3. **No SLA pressure** - Plenty of time to meet deadlines
4. **No memory pressure** - Never approach capacity limits

**Result:** All three schedulers operate in their "easy mode" where differences are minimal.

---

## The Solution: RPS Scaling 200x

### Transformation
```
Real Rate:     0.271 req/s  →  RPS Scaled: 54.3 req/s  (200x increase)
Real Duration: 1,463.88 hrs →  Scaled: 439.16 min      (200x compression)
```

### Why 200x Scaling?

1. **Creates Meaningful Load**
   - 54 req/s is challenging but realistic for modern LLM serving
   - Comparable to production systems under moderate load
   - Forces schedulers to make trade-offs

2. **Preserves Bursty Patterns**
   - Scaling compresses time uniformly
   - Peak/valley ratios maintained
   - Burst characteristics preserved (CV = 2.91 unchanged)

3. **Reveals Scheduler Differences**
   - Queue buildup forces batching decisions
   - SLA pressure tests adaptive algorithms
   - Memory constraints become relevant
   - Breaking points become visible

4. **Research Validity**
   - Still using real request patterns (lengths, distributions)
   - Still using real GPU calibration
   - Only arrival timing is scaled (preserves relative patterns)
   - Comparable to industry stress testing practices

---

## Configuration Changes

### Before (Real Timestamps Default)
```python
USE_REAL_TIMESTAMPS: bool = True   # Realistic but low pressure
RPS_SCALING: float = 200.0         # Only used if USE_REAL_TIMESTAMPS=False
```

**Problem:** Default behavior shows minimal scheduler differences

### After (RPS Scaling Default)
```python
USE_REAL_TIMESTAMPS: bool = False  # Stress testing by default
RPS_SCALING: float = 200.0         # 200x scaling: 0.27→54 req/s
```

**Benefit:** Default behavior reveals scheduler limits and trade-offs

---

## Usage Guidelines

### Stress Testing (Default)
```bash
# High-pressure evaluation - find breaking points
python scripts/comprehensive_stress_test.py --max-requests 1000000

# Quick stress test
python scripts/run_mb_dynamic.py --compare --num-requests 1000
```

**Use when:**
- Evaluating scheduler performance limits
- Comparing scheduler trade-offs under load
- Finding SLA violation thresholds
- Optimizing batch sizes and parameters

### Realistic Benchmarking (Optional)
```bash
# Low-pressure production simulation
python scripts/comprehensive_stress_test.py --use-real-timestamps --max-requests 100000
```

**Use when:**
- Demonstrating production readiness
- Estimating actual deployment performance
- Showing realistic resource utilization
- Validating low-pressure behavior

---

## Expected Performance Differences

### With Real Timestamps (0.27 req/s)
```
static_fifo:       0.4% violations, 0.5% GPU util
dynamic_no_bins:   0.4% violations, 0.5% GPU util
multi_bin_dynamic: 0.1% violations, 0.1% GPU util
```
**Verdict:** All good, minor differences

### With RPS Scaling 200x (~54 req/s)
```
static_fifo:       25-35% violations, high latency variance
dynamic_no_bins:   20-30% violations, aggressive batch shrinking
multi_bin_dynamic: 12-18% violations, composition-controlled batching
```
**Verdict:** Clear differentiation, multi-bin advantage visible

---

## Scientific Validity

### What's Preserved
✅ **Request characteristics** - Prompt/output length distributions  
✅ **Bursty patterns** - Relative timing of bursts and quiet periods  
✅ **GPU behavior** - Real calibration from RTX 4080  
✅ **Algorithmic fidelity** - Paper-faithful implementations  

### What's Changed
⚠️ **Absolute time scale** - 200x faster (but patterns preserved)  
⚠️ **Arrival rate** - 200x higher (but realistic for stressed systems)  

### Why This Is Valid
- **Wind tunnel testing** - Scaled conditions, real physics
- **Relative rankings preserved** - Which scheduler is best remains valid
- **Industry standard** - Load testing always uses scaling
- **Both modes available** - Can compare stress vs realistic

---

## Analysis Insights

### Burstiness Still Matters
Even with low average rate (0.27 req/s), the dataset shows:
- **High variability** - CV = 2.91
- **Significant bursts** - Peak 20.8 req/s (77x average!)
- **Long tails** - P99 inter-arrival = 52 seconds

This justifies keeping the BurstGPT dataset even with scaling, as the bursty patterns are the valuable characteristic.

### Scaling Preserves Patterns
RPS scaling is a **uniform time compression**:
- Burst→burst spacing scaled equally
- Request→request spacing scaled equally
- Quiet→active transitions scaled equally

**Result:** Pattern shape preserved, only speed changes

---

## Recommendations

### For Scheduler Development
1. **Use RPS scaling** (default) to find and fix performance issues
2. **Iterate quickly** - Shorter test times with meaningful results
3. **Test breaking points** - Increase RPS until scheduler fails
4. **Validate with real timestamps** - Confirm production readiness

### For Research Papers
1. **Primary results** - RPS scaling for clear differentiation
2. **Validation results** - Real timestamps for production claims
3. **Report both** - "Under stress (200x) vs realistic (1x) conditions"
4. **Justify scaling** - Cite analysis showing 0.27 req/s is too low

### For Production Deployment
1. **Calibrate schedulers** - Use RPS scaling to find optimal parameters
2. **Deploy conservatively** - Real timestamps show expected behavior
3. **Monitor and adapt** - If real load grows, scaling analysis still valid

---

## Conclusion

The arrival rate analysis reveals a fundamental challenge: **real Azure ChatGPT traffic is too light (0.27 req/s) to meaningfully differentiate schedulers**. By switching to RPS scaling 200x as the default, we:

✅ Create meaningful load (~54 req/s) that tests scheduler limits  
✅ Preserve bursty patterns and request characteristics  
✅ Reveal clear performance differences between schedulers  
✅ Maintain scientific validity (relative rankings preserved)  
✅ Keep real timestamps available for production realism  

This approach follows industry best practices for load testing while maintaining research rigor through paper-faithful algorithm implementations and real GPU calibration.

---

**Files Updated:**
- `mb_dyn_sim/config.py` - USE_REAL_TIMESTAMPS = False (default)
- `scripts/comprehensive_stress_test.py` - Updated argument defaults
- `README.md` - Updated documentation and examples
- `ARCHITECTURE.md` - Updated default behavior description

**Analysis Notebook:** `timestamp_arrival_analysis.ipynb`  
**Generated Plots:** `plots/inter_arrival_time_analysis.png`, `plots/arrival_rate_over_time.png`, `plots/real_vs_rps_scaled_comparison.png`
