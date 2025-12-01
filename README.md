# Multi-Bin Dynamic Batching Scheduler for LLM Inference

A **paper-faithful** discrete-event simulator implementing algorithms from:
1. **Multi-Bin Batching for LLM Inference Throughput Optimization**
2. **Memory-Aware and SLA-Constrained Dynamic Batching for LLM Inference**

## Features

✅ **Real BurstGPT Dataset** - Azure ChatGPT traces (1.43M+ requests)  
✅ **GPU-Calibrated Latency** - Qwen3 1.7B on RTX 4080 12GB (R²=0.9995)  
✅ **Three Scheduler Modes** - static_fifo, dynamic_no_bins, multi_bin_dynamic  
✅ **Dual SLA Model** - Per-token (10ms) and Per-request (20s) SLA tracking  
✅ **Performance Optimized** - Workload caching, incremental saves  

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Quick Comparison
```bash
python scripts/run_mb_dynamic.py --compare --num-requests 1000
```

### 3. Run Two-Step Evaluation
```bash
# Step 1: Grid Search (192 configurations)
python scripts/step1_low_load.py

# Step 2: Method Comparison (4 methods × 4 workloads)
python scripts/step2_low_load.py
```

---

## Three Scheduler Modes

### 1. `static_fifo` - Baseline
- Single FIFO queue with fixed batch size (B=8)
- No dynamic adaptation

### 2. `dynamic_no_bins` - Dynamic Batching
- Single FIFO queue with SLA controller
- Adaptive batch sizing: `b_target = min(b_mem, b_SLA)`

### 3. `multi_bin_dynamic` - Our Contribution
- K bins partition requests by predicted output length
- **Bin-specific** statistics and SLA controllers
- Better batch composition → higher throughput

**Key Innovation**: Multi-bin reduces E[max(t_j) | bin] by grouping similar-length requests, allowing larger batches while maintaining SLA.

---

## Project Structure

```
llm_scheduler_sim/
├── mb_dyn_sim/                    # Core simulator
│   ├── config.py                  # Configuration parameters
│   ├── schedulers.py              # Scheduler implementations
│   ├── simulation.py              # Discrete-event simulator
│   ├── workload.py                # Workload generation
│   ├── model_calibration.py       # GPU latency model
│   └── metrics.py                 # Performance metrics
├── scripts/
│   ├── run_mb_dynamic.py          # Main experiment runner
│   ├── comprehensive_stress_test_optimized.py  # Stress test suite
│   └── download_burstgpt.py       # Download BurstGPT dataset
├── data/
│   ├── BurstGPT_sample.csv        # Real Azure traces
│   └── qwen3_1_7b_latency_grid.csv # GPU calibration data
└── docs/                          # Documentation (MD files)
```

---

## Configuration

### Key Parameters (`mb_dyn_sim/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_GPUS` | 4 | Number of GPUs |
| `K_BINS` | 4 | Number of bins for multi_bin_dynamic |
| `D_SLA` | 0.05 | Per-token decode latency SLA (50ms) |
| `B_MAX` | 128 | Maximum batch size |
| `M_MAX_GB` | 12.0 | GPU memory (RTX 4080) |
| `USE_REAL_TIMESTAMPS` | False | True = real arrival times, False = RPS scaling |
| `RPS_SCALING` | 200.0 | RPS multiplier (when USE_REAL_TIMESTAMPS=False) |

### Arrival Modes

- **RPS Scaling** (default): Compress arrival times 200x for stress testing
- **Real Timestamps**: Use actual Azure arrival patterns for realistic benchmarking

---

## Usage Examples

### Download BurstGPT Dataset
```bash
python scripts/download_burstgpt.py --version 1
```

### Run Individual Schedulers
```bash
python scripts/run_mb_dynamic.py --scheduler static_fifo --num-requests 5000
python scripts/run_mb_dynamic.py --scheduler dynamic_no_bins --num-requests 5000
python scripts/run_mb_dynamic.py --scheduler multi_bin_dynamic --num-requests 5000
```

### K-Bins Sensitivity Analysis
```bash
python scripts/run_mb_dynamic.py --k-bins 2 --num-requests 5000
python scripts/run_mb_dynamic.py --k-bins 4 --num-requests 5000
python scripts/run_mb_dynamic.py --k-bins 8 --num-requests 5000
```

### Custom SLA
```bash
python scripts/run_mb_dynamic.py --compare --d-sla 0.1 --num-requests 1000
```

---

## Documentation

### Primary Documents (Start Here)

| File | Description |
|------|-------------|
| **[METHODOLOGY.md](METHODOLOGY.md)** | **Complete research methodology** - data sources, algorithms, experimental design |
| **[EXPERIMENTS.md](EXPERIMENTS.md)** | **Detailed experimental protocol** - Step 1 grid search, Step 2 comparison |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture and process flows |

### Technical Reference

| File | Description |
|------|-------------|
| [LATENCY_MODEL_AND_SLA.md](LATENCY_MODEL_AND_SLA.md) | Latency model derivation and SLA framework |
| [METRICS_GUIDE.md](METRICS_GUIDE.md) | Performance metrics computation |
| [GPU_SPECIFICATIONS.md](GPU_SPECIFICATIONS.md) | Hardware specifications |
| [data/README.md](data/README.md) | Dataset format specification |

### Analysis Documents

| File | Description |
|------|-------------|
| [COMPREHENSIVE_STRESS_TEST_3STEP.md](COMPREHENSIVE_STRESS_TEST_3STEP.md) | Legacy stress test guide |
| [KBINS_PERFORMANCE_ANALYSIS.md](KBINS_PERFORMANCE_ANALYSIS.md) | K-bins sensitivity analysis |
| [INDUSTRY_METRICS_VALIDATION.md](INDUSTRY_METRICS_VALIDATION.md) | Industry benchmark comparison |

### Algorithm Specifications

| File | Description |
|------|-------------|
| [ALGORITHMS.md](ALGORITHMS.md) | **Formal algorithm pseudocode** - all 7 core algorithms |

---

## Research Paper Writing Guide

### For "Methodology / Materials and Methods" Section

**Primary document to reference**: [METHODOLOGY.md](METHODOLOGY.md)

This covers:
1. ✅ Research overview and hypotheses
2. ✅ System architecture
3. ✅ Data sources (BurstGPT dataset)
4. ✅ Latency model derivation
5. ✅ All 7 scheduling algorithms (pseudocode in [ALGORITHMS.md](ALGORITHMS.md))
6. ✅ Dual SLA framework (token + request)
7. ✅ Experimental design (Step 1 grid search + Step 2 comparison)
8. ✅ Implementation details
9. ✅ Evaluation metrics
10. ✅ Reproducibility information

### Document Reading Order

For comprehensive understanding:

```
1. README.md          ← Start here (overview)
2. METHODOLOGY.md     ← Complete methodology (main reference)
3. ALGORITHMS.md      ← Formal algorithm specifications
4. EXPERIMENTS.md     ← Detailed experimental protocol
5. LATENCY_MODEL_AND_SLA.md ← Latency model derivation
6. ARCHITECTURE.md    ← System architecture diagrams
7. METRICS_GUIDE.md   ← Metrics computation details
```

---

## SLA Definition

### Dual SLA Model (v2)

| SLA Type | Metric | Threshold | Purpose |
|----------|--------|-----------|---------|
| **Per-Token SLA** | Decode TBT (β·h(b)) | 10ms | Streaming UX |
| **Per-Request SLA** | Total latency | 20s | Interactive response |

**Key Innovation**: Token SLA applies ONLY to decode TBT, NOT TTFT (prefill latency).

### Calibrated Parameters (RTX 4080 12GB + Qwen3 1.7B)

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (TTFT) | 59.65ms | Time To First Token (prefill) |
| β (TBT) | 5.74ms/token | Per-token decode time |
| γ (penalty) | 0.316 | Batch overhead factor |
| R² | 0.9995 | Model fit quality |

---

## Scientific Validity

This simulator follows the **wind tunnel testing** approach:

| Aspect | Real Deployment | Our Simulator |
|--------|----------------|---------------|
| Cost | $$$ (GPU cluster) | Free (CPU only) |
| Speed | Days/weeks | Seconds |
| Risk | User-facing | Zero (offline) |
| **Validity** | **Absolute numbers** | **Relative rankings** ✓ |

The simulator preserves algorithmic fidelity for valid scheduler comparisons.

---

## FAQ

**Q: Do I need a GPU?**  
A: No. The simulator runs on CPU with pre-calibrated latency data.

**Q: Do I need the Qwen model?**  
A: No. Latency data is already provided in `data/qwen3_1_7b_latency_grid.csv`.

**Q: Are the results valid?**  
A: Yes for relative comparisons. The simulator preserves algorithmic differences between schedulers.

---

## References

### Papers
1. Multi-Bin Batching for LLM Inference Throughput Optimization
2. Memory-Aware and SLA-Constrained Dynamic Batching for LLM Inference

### Dataset
- **BurstGPT**: [github.com/HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT)
- Real ChatGPT/GPT-4 workload traces from Azure (121 days, 5.29M requests)

### Model
- **Qwen3-1.7B**: [huggingface.co/Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

---

## Citation

```bibtex
@software{multibin_dynamic_scheduler,
  title={Multi-Bin Dynamic Batching Scheduler for LLM Inference},
  year={2025},
  note={Paper-faithful implementation of multi-bin batching and dynamic batching algorithms}
}
```

BurstGPT dataset:
```bibtex
@inproceedings{BurstGPT,
  author = {Yuxin Wang and Yuhan Chen and Zeyu Li and Xueze Kang and others},
  title = {{BurstGPT}: A Real-World Workload Dataset to Optimize LLM Serving Systems},
  booktitle = {KDD '25},
  year = {2025},
}
```

---

**Status:** ✅ Production-ready  
**Last Updated:** November 2025
