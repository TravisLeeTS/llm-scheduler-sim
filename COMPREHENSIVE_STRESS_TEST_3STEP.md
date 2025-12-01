# Comprehensive Stress Test - 3-Step Research Plan

## Overview
This document describes the updated comprehensive stress testing suite designed for research paper evaluation. The test suite uses **RPS scaling (200x)** by default to create meaningful load for scheduler differentiation.

## Test Architecture

### Step 1: Request Scaling Stress Test (1K → 1M requests)
**Objective:** Evaluate algorithm capacity to handle increasing request volumes

**Configuration:**
- Request volumes: 1K → 10K → 100K → 1M (10x increments)
- Fixed: K=4 bins, RPS scaling 200x
- Schedulers tested:
  - `static_fifo`: 1 GPU (baseline, no dynamic batching)
  - `dynamic_no_bins`: 1 GPU (dynamic batching, single queue)
  - `multi_bin_dynamic`: **1 GPU, 2 GPUs, 4 GPUs** (test GPU scalability)

**Purpose:**
- Identify breaking points for each scheduler
- Compare single-GPU performance (all 3 schedulers)
- Demonstrate multi-bin's ability to scale with GPUs
- Show capacity limits under increasing load

**Total Tests:** 4 request volumes × 5 configs = **20 tests**

### Step 2: GPU Scaling Stress Test (1M requests, 1-100 GPUs)
**Objective:** Evaluate multi-bin's GPU scaling efficiency

**Configuration:**
- Fixed: 1,000,000 requests, K=4 bins, RPS scaling 200x
- GPU counts: **1, 2, 4, 8, 16, 32, 64, 100**
- Scheduler: `multi_bin_dynamic` only

**Purpose:**
- Test algorithm's ability to assign tasks across large GPU pools
- Identify GPU scaling saturation point
- Measure scaling efficiency (linear vs sublinear)
- Find optimal GPU allocation for production

**Total Tests:** 8 GPU configurations = **8 tests**

### Step 3: K-Bins Sensitivity Analysis (Best GPU config)
**Objective:** Find optimal bin partitioning strategy

**Configuration:**
- Fixed: 1,000,000 requests, best GPU count from Step 2
- K-bins values: **1, 2, 4, 8, 16, 32**
- Scheduler: `multi_bin_dynamic` only

**Purpose:**
- Determine optimal bin count (K parameter)
- Measure performance sensitivity to binning strategy
- Validate K=4 default or suggest better value
- Understand trade-offs between partitioning granularity

**Total Tests:** 6 K values = **6 tests**

## Total Test Count
- **Step 1:** 20 tests
- **Step 2:** 8 tests  
- **Step 3:** 6 tests
- **TOTAL:** 34 tests (comprehensive research evaluation)

## Usage

### Run All 3 Steps (Default)
```bash
python scripts/comprehensive_stress_test.py
```

### Run Individual Steps
```bash
# Step 1 only (request scaling)
python scripts/comprehensive_stress_test.py --step1-only

# Step 2 only (GPU scaling)
python scripts/comprehensive_stress_test.py --step2-only

# Step 3 only (K-bins sensitivity)
python scripts/comprehensive_stress_test.py --step3-only --best-gpu-count 32
```

### Custom Configuration
```bash
# Adjust parameters
python scripts/comprehensive_stress_test.py \
  --max-requests 10000000 \
  --max-gpus 200 \
  --rps-scaling 500.0 \
  --d-sla 0.5
```

## Output Files

The test generates 5 output files:
1. `comprehensive_stress_results_step1_request_scaling.csv` - Step 1 results
2. `comprehensive_stress_results_step2_gpu_scaling.csv` - Step 2 results
3. `comprehensive_stress_results_step3_kbins_sensitivity.csv` - Step 3 results
4. `comprehensive_stress_results_all_results.csv` - Combined results
5. `comprehensive_stress_results_analysis.json` - Analysis summary

## Key Metrics Tracked

### Performance Metrics
- **SLA Violation Rate** - % of requests exceeding 1.0s deadline
- **Capacity QPS** - Sustainable throughput under SLA
- **Latency** (avg, p50, p95, p99, max)
- **Throughput** (requests/sec, tokens/sec)

### Resource Metrics
- **GPU Utilization** (avg, min, max)
- **Batch Statistics** (size, count)
- **Queueing Delay** vs Service Time

## Analysis Features

### Step 1 Analysis
- Request volume vs SLA violations
- Scheduler comparison at each scale
- GPU scaling benefit (multi-bin 1/2/4 GPUs)
- Capacity limits and breaking points

### Step 2 Analysis
- GPU scaling efficiency (linear/sublinear)
- Saturation point detection
- Optimal GPU allocation
- Cost-benefit trade-offs

### Step 3 Analysis
- Optimal K-bins value
- Performance variance across K values
- Sensitivity analysis (how critical is K?)
- Recommendations for production

**Key Findings** (see [KBINS_PERFORMANCE_ANALYSIS.md](KBINS_PERFORMANCE_ANALYSIS.md) for full analysis):
- **K=1 → K=2**: Dramatic 70% latency reduction and 24% QPS improvement
- **K=8-16**: Optimal range (best performance/complexity trade-off)
- **K>16**: Diminishing returns (<1% QPS gain per doubling)
- **Long-tail latencies**: P99 remains high (10-13s) but many are legitimate long requests (not violations)
- **No paradox**: 8% SLA violation (normalized space) vs p95=1.56s (raw space) are mathematically consistent
- **Per-token SLA**: Long requests (2000 tokens) can take 12s and still meet normalized SLA
- **Batch dynamics**: Avg batch size stays small (1.18-1.23), showing SLA control is effective

## Expected Execution Time

With **RPS scaling 200x** (stress testing mode):
- **Step 1:** ~10-15 minutes (20 tests)
- **Step 2:** ~5-8 minutes (8 tests)
- **Step 3:** ~3-5 minutes (6 tests)
- **Total:** ~20-30 minutes for full suite

*(Times vary based on hardware and dataset size)*

## Research Paper Insights

This 3-step plan provides comprehensive evidence for:

1. **Algorithm Correctness** (Step 1)
   - Multi-bin outperforms baselines at all scales
   - GPU scaling provides meaningful improvements
   - Handles realistic bursty workloads (200x compressed)

2. **Scalability** (Step 2)
   - Efficient GPU allocation across 1-100 GPUs
   - Identifies saturation point (e.g., ~32 GPUs)
   - Demonstrates production viability

3. **Parameter Optimization** (Step 3)
   - K=4 bins is optimal (or find better value)
   - Performance is/isn't sensitive to K
   - Provides tuning guidance for practitioners

## Configuration Notes

### Stress Testing Mode (Default)
```python
USE_REAL_TIMESTAMPS = False
RPS_SCALING = 200.0  # 0.27 req/s → 54 req/s
```

**Why RPS Scaling?**
- Real BurstGPT arrival rate is very low (0.27 req/s)
- All schedulers handle this load easily (<0.5% SLA violations)
- 200x scaling creates meaningful stress (~54 req/s)
- Preserves temporal patterns (burstiness, CV=13.28)
- Enables scheduler differentiation

### Realistic Benchmarking Mode
```bash
python scripts/comprehensive_stress_test.py --use-real-timestamps
```

Use this for:
- Production capacity planning
- Real-world SLA compliance testing
- Conservative estimates

## Next Steps

After running the comprehensive test:

1. **Review Analysis:** Check `_analysis.json` for key findings
2. **Visualize Results:** Use analysis scripts to generate plots
3. **Write Paper:** Use metrics for performance comparison section
4. **Tune Parameters:** Apply Step 3 findings to production config

## Example Results Structure

```json
{
  "experiment_metadata": {
    "date": "2025-01-26 14:30:00",
    "total_tests": 34,
    "step1_tests": 20,
    "step2_tests": 8,
    "step3_tests": 6,
    "successful_tests": 34
  },
  "step1_request_scaling": {
    "multi_bin_dynamic": {
      "request_volumes": [1000, 10000, 100000, 1000000],
      "sla_violations_pct": [0.1, 0.5, 2.3, 8.7],
      "capacity_qps": [45.2, 52.1, 48.3, 42.5]
    }
  },
  "step2_gpu_scaling": {
    "gpu_counts": [1, 2, 4, 8, 16, 32, 64, 100],
    "capacity_qps": [42.5, 78.3, 142.1, 251.3, 412.5, 485.2, 492.1, 493.5],
    "bottleneck_analysis": {
      "optimal_gpu_count": 32
    }
  },
  "step3_kbins_sensitivity": {
    "k_values": [1, 2, 4, 8, 16, 32],
    "capacity_qps": [380.2, 445.1, 485.2, 482.3, 471.5, 460.1],
    "optimal_k": {
      "k_bins": 4,
      "capacity_qps": 485.2
    }
  }
}
```

## Troubleshooting

### Low SLA Violations Across All Schedulers
→ Increase `--rps-scaling` (try 500x or 1000x)

### Out of Memory Errors
→ Reduce `--max-requests` or test smaller subsets

### Step 3 Not Running
→ Ensure Step 2 completed successfully, or specify `--best-gpu-count`

### Very Long Execution Time
→ Use `--step1-only` first, then run other steps separately
