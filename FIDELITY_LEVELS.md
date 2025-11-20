# Four Fidelity Levels - Quick Reference

## Level Comparison Table

| Level | Requirements | Validity | Speed | Use Case |
|-------|-------------|----------|-------|----------|
| **1: Synthetic** | None | ✓ | ~40ms | Algorithm development, quick iteration |
| **2: BurstGPT** | CSV file | ✓✓ | ~70ms | Realistic workload validation |
| **3: GPU Calibrated** | GPU measurements | ✓✓✓ | ~280ms | GPU-specific performance analysis |
| **4: Full Production** | Both CSVs | ✓✓✓✓ | ~80ms | **Publication results, maximum realism** ⭐ |

---

## Quick Test Commands

### Test Individual Levels
```bash
# Level 1: Synthetic (fastest, no dependencies)
python scripts/test_all_levels.py --level 1

# Level 2: BurstGPT (requires data/BurstGPT_sample.csv)
python scripts/test_all_levels.py --level 2

# Level 3: GPU Calibrated (requires data/qwen3_1_7b_latency_grid.csv)
python scripts/test_all_levels.py --level 3

# Level 4: Full Production (requires both CSVs) ⭐
python scripts/test_all_levels.py --level 4
```

### Test All Levels
```bash
# Comprehensive test (~0.5 second total)
python scripts/test_all_levels.py --level all
```

---

## Expected Results

### Level 1: Synthetic
- **Purpose:** Algorithm validation with formula-based latency
- **SLA Violations:** static=43%, dynamic=1.5%, multi_bin=0.7%
- **Key Finding:** Multi-bin improves by 53% over dynamic-only

### Level 2: BurstGPT
- **Purpose:** Realistic bursty workload patterns
- **Dataset:** 1000 real Azure trace requests
- **Key Finding:** Tests scheduler behavior under production-like arrivals

### Level 3: GPU Calibrated
- **Purpose:** Production-ready accuracy with real GPU latency
- **SLA Violations:** static=96.5%, dynamic=61.7%, multi_bin=22.4%
- **Key Finding:** Multi-bin improves by 64% over dynamic-only

### Level 4: Full Production ⭐
- **Purpose:** Maximum realism (real workload + real GPU)
- **SLA Violations:** static=11.5%, dynamic=11.8%, multi_bin=6.8%
- **Key Finding:** Multi-bin improves by 42% over dynamic-only
- **Why it matters:** Most realistic simulation for publication

---

## What Each Level Tests

### All Levels Test:
✓ Three schedulers produce distinct results  
✓ Multi-bin + dynamic outperforms baselines  
✓ Equal-mass bin boundary computation  
✓ Dynamic batch sizing with SLA control  
✓ Poisson arrival generation  

### Level 2 Additionally Tests:
✓ BurstGPT dataset loading  
✓ Real Azure arrival patterns  
✓ Bursty ON/OFF traffic handling  

### Level 3 Additionally Tests:
✓ GPU calibration data loading  
✓ Real latency model fitting (R²=1.0)  
✓ Production-realistic service times  

### Level 4 Additionally Tests: ⭐
✓ **Combination of real workload + real GPU**  
✓ **Maximum fidelity for production simulation**  
✓ **Best for publication results**  

---

## Dependencies Status

| Level | File Required | Status | Notes |
|-------|--------------|--------|-------|
| 1 | None | ✅ Built-in | Always available |
| 2 | `data/BurstGPT_sample.csv` | ✅ Available | 1000 request sample |
| 3 | `data/qwen3_1_7b_latency_grid.csv` | ✅ Available | RTX 4080 measurements |
| 4 | Both CSVs above | ✅ Both Available | Maximum realism ⭐ |

**All dependencies are included in the repository!**

---

## Run Modes

### Quick Validation (Default)
```bash
python scripts/test_all_levels.py --level all
```
- Tests all levels
- 1000 requests per scheduler
- Takes ~1 second total
- Displays summary table

### Individual Level
```bash
python scripts/test_all_levels.py --level 1  # or 2, or 3
```
- Tests single level
- Faster execution
- Detailed output for that level

---

## Output Interpretation

### Success Indicators
✓ "PASSED" status for all tested levels  
✓ Schedulers show distinct SLA violation rates  
✓ Multi-bin < dynamic < static (violation ranking)  
✓ Throughput metrics > 0  
✓ No errors during execution  

### Key Metrics to Check
- **SLA Violations:** Lower is better (multi_bin should be lowest)
- **Throughput:** Higher is better (dynamic modes should match/exceed static)
- **Latency:** Lower is better (multi_bin should have lowest average)

---

## Troubleshooting

### "BurstGPT dataset not found"
**Solution:** File already exists at `data/BurstGPT_sample.csv`. Check path.

### "GPU calibration data not found"
**Solution:** File already exists at `data/qwen3_1_7b_latency_grid.csv`. Check path.

### "All schedulers identical results"
**Cause:** Load too low, GPU not saturated  
**Solution:** Increase `--num-requests` or arrival rate

### Unicode encoding errors (Windows)
**Solution:** Script auto-fixes this with UTF-8 encoding wrapper

---

## Integration with Main Scripts

### After Testing, Run Experiments:
```bash
# Use Level 1 (Synthetic - fastest)
python scripts/run_mb_dynamic.py --compare

# Use Level 2 (BurstGPT - realistic)
python scripts/run_mb_dynamic.py \
    --arrival-profile burstgpt_dataset \
    --dataset-path data/BurstGPT_sample.csv \
    --compare

# Use Level 3 (GPU Calibrated - accurate)
python scripts/run_mb_dynamic.py \
    --use-real-calibration \
    --calibration-csv data/qwen3_1_7b_latency_grid.csv \
    --compare
```

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/test_all_levels.py` | Automated test suite |
| `TESTING_SUMMARY.md` | Detailed test results and analysis |
| `README.md` | Updated with testing section |

---

## Next Steps After Testing

1. ✅ Validated all three levels work
2. Run larger experiments with `scripts/run_mb_dynamic.py`
3. Generate plots for paper/presentation
4. Sweep parameters (K_BINS, D_SLA, arrival rates)
5. Write up results using validated metrics

---

*Last Updated: November 20, 2025*
