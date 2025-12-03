# Multi-Bin Dynamic Batching Scheduler for LLM Inference

A discrete-event simulator for LLM request scheduling, implementing algorithms from:
1. **Multi-Bin Batching for LLM Inference Throughput Optimization**
2. **Memory-Aware and SLA-Constrained Dynamic Batching for LLM Inference**

## Key Results

| Load Level | Best SLA Violation | Throughput Improvement | Configuration |
|------------|-------------------|------------------------|---------------|
| High (100×) | 3.5% | 38.5× vs FIFO | 16 GPUs, K=4 |
| Medium (10×) | 1.8% | 3.9× vs FIFO | 4 GPUs, K=4 |
| Low (1×) | 1.6% | 1.0× vs FIFO | 1 GPU, K=2 |

Full results: [`RESULTS_AND_DISCUSSION.md`](RESULTS_AND_DISCUSSION.md)

---

## Features

✅ **Real BurstGPT Dataset** - Azure ChatGPT traces (1.43M+ requests)  
✅ **GPU-Calibrated Latency** - Qwen3 1.7B on RTX 4080 12GB (R²=0.9995)  
✅ **Three Scheduler Modes** - static_fifo, dynamic_no_bins, multi_bin_dynamic  
✅ **Dual SLA Model** - Per-token (10ms) and Per-request (20s) SLA tracking  

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

### 3. Run Full Evaluation (Three Load Levels)
```bash
# High Load (100× RPS) - ~27 req/s
python scripts/step1_grid_search.py    # Grid search
python scripts/step2_comparison.py     # Method comparison

# Medium Load (10× RPS) - ~2.7 req/s
python scripts/step1_low_load.py
python scripts/step2_low_load.py

# Low Load (1× RPS) - ~0.27 req/s
python scripts/step1_ultra_low_load.py
python scripts/step2_ultra_low_load.py
```

---

## Repository Structure

```
llm-scheduler-sim/
├── README.md                      # This file
├── RESULTS_AND_DISCUSSION.md      # Full methodology, results, and discussion
├── CODE_TO_RESULTS_MAPPING.md     # Script-to-output mapping
├── requirements.txt
│
├── mb_dyn_sim/                    # Core simulation library
│   ├── config.py                  # SchedulerConfig class
│   ├── schedulers.py              # FIFO, Dynamic, Multi-Bin schedulers
│   ├── simulation.py              # Discrete-event simulator
│   ├── workload.py                # BurstGPT workload generation
│   ├── metrics.py                 # SLA, throughput, latency metrics
│   └── model_calibration.py       # GPU latency model
│
├── scripts/
│   ├── step1_grid_search.py       # High load grid search → stress_test_final/
│   ├── step1_low_load.py          # Medium load grid search → stress_test_low_load/
│   ├── step1_ultra_low_load.py    # Low load grid search → stress_test_ultra_low_load/
│   ├── step2_comparison.py        # High load method comparison
│   ├── step2_low_load.py          # Medium load method comparison
│   ├── step2_ultra_low_load.py    # Low load method comparison
│   ├── generate_analysis_plots.py # Generate figures → figures/
│   ├── download_burstgpt.py       # Download BurstGPT dataset
│   └── run_mb_dynamic.py          # Interactive experiment runner
│
├── data/
│   ├── BurstGPT_sample.csv        # Azure ChatGPT traces (1.43M requests)
│   ├── qwen3_1_7b_latency_grid.csv # GPU calibration data
│   └── README.md                  # Dataset documentation
│
├── figures/                       # Generated figures
│   ├── fig1_throughput_vs_gpu.png
│   ├── fig2_sla_pareto_frontier.png
│   ├── fig3_latency_throughput_tradeoff.png
│   ├── fig4_scheduler_comparison.png
│   ├── fig5_sensitivity_variance.png
│   ├── sensitivity_analysis.csv
│   └── multi_seed_variance.csv
│
├── stress_test_final/             # High load (100×) results
│   ├── step1_grid_search.csv
│   └── step2_comparison.csv
├── stress_test_low_load/          # Medium load (10×) results
│   ├── step1_grid_search.csv
│   └── step2_comparison.csv
└── stress_test_ultra_low_load/    # Low load (1×) results
    ├── step1_grid_search.csv
    └── step2_comparison.csv
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

## Configuration

### Key Parameters (`mb_dyn_sim/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_GPUS` | 4 | Number of GPUs |
| `K_BINS` | 4 | Number of bins for multi_bin_dynamic |
| `D_SLA_TOKEN` | 0.010 | Per-token decode latency SLA (10ms) |
| `D_SLA_REQUEST` | 20.0 | Per-request total latency SLA (20s) |
| `B_MAX` | 128 | Maximum batch size |
| `RPS_SCALING` | 100.0 | Arrival rate multiplier |

---

## Dual SLA Model

| SLA Type | Metric | Threshold | Purpose |
|----------|--------|-----------|---------|
| **Per-Token SLA** | Decode TBT | 10ms | Streaming UX |
| **Per-Request SLA** | Total latency | 20s | Interactive response |

### Calibrated Parameters (RTX 4080 12GB + Qwen3 1.7B)

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (TTFT) | 59.65ms | Time To First Token |
| β (TBT) | 5.74ms/token | Per-token decode time |
| γ (penalty) | 0.316 | Batch overhead factor |
| R² | 0.9995 | Model fit quality |

---

## Results Summary

### Method Comparison (10K requests, Medium Load)

| Method | Request SLA (%) | Throughput (tok/s) | Latency Reduction |
|--------|-----------------|-------------------|-------------------|
| Static FIFO | 84.78 | 270.4 | baseline |
| Dynamic No-Bins | 56.77 | 270.3 | 93.1% ↓ |
| Multi-Bin (1 GPU) | 71.35 | 270.1 | 85.4% ↓ |
| **Multi-Bin (Optimal)** | **1.80** | **270.4** | **97.9% ↓** |

See [`RESULTS_AND_DISCUSSION.md`](RESULTS_AND_DISCUSSION.md) for complete results.

---

## FAQ

**Q: Do I need a GPU?**  
A: No. The simulator runs on CPU with pre-calibrated latency data.

**Q: Are the results valid?**  
A: Yes for relative comparisons. The simulator preserves algorithmic differences between schedulers.

**Q: How long do experiments take?**  
A: Quick tests (~1K requests): seconds. Full grid search (1M requests): hours.

---

## References

### Papers
1. Multi-Bin Batching for LLM Inference Throughput Optimization
2. Memory-Aware and SLA-Constrained Dynamic Batching for LLM Inference

### Dataset
- **BurstGPT**: [github.com/HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT)

### Model
- **Qwen3-1.7B**: [huggingface.co/Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

---

## Citation

```bibtex
@software{multibin_dynamic_scheduler,
  title={Multi-Bin Dynamic Batching Scheduler for LLM Inference},
  author={Travis Lee},
  year={2025},
  url={https://github.com/TravisLeeTS/llm-scheduler-sim}
}
```

---

**Last Updated:** December 2025
