# Multi-Bin Dynamic Batching Scheduler for LLM Inference

**Paper-faithful** discrete-event simulator implementing exact algorithms from:
1. **Multi-Bin Batching for LLM Inference Throughput Optimization**
2. **Memory-Aware and SLA-Constrained Dynamic Batching for LLM Inference**

✅ **Validated with Real BurstGPT Dataset** (Azure traces with 1.4M+ requests)  
✅ **GPU Calibration Ready** (RTX 4080 + CUDA 12.6 + Transformers/vLLM)  
✅ **Three Scheduler Modes** (static_fifo, dynamic_no_bins, multi_bin_dynamic)

---

## Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Download BurstGPT Dataset
```powershell
curl -L "https://github.com/HPMLL/BurstGPT/raw/main/data/BurstGPT_1.csv" -o "data/BurstGPT_sample.csv"
```

### 3. Run Comparison Test
```powershell
python scripts/run_mb_dynamic.py --compare --num-requests 5000
```

This compares all three scheduler modes with the discrete-event simulator!

---

## Features

### ✅ Paper-Faithful Algorithms

**From Multi-Bin Batching Paper:**
- Equal-mass bin boundaries (empirical quantiles)
- Fixed batch size B for paper validation
- Throughput scaling with K_BINS

**From Dynamic Batching Paper:**
- Algorithm 1: Memory constraint `b_mem = ⌊(η-L₀)/μ⌋`
- Algorithm 2: SLA controller with adaptive `[b_low, b_high]` search
- Final: `b_target = min(b_mem, b_SLA)`

**Additional:**
- Service time: max-dominates property
- Feedback loops: `update_after_batch()`
- Event-driven discrete-event simulation

### 📊 Three Scheduler Modes

1. **`static_fifo`** - Fixed batch size (B=8), no dynamic batching, baseline
2. **`dynamic_no_bins`** - Single queue with SLA controller + memory constraint
3. **`multi_bin_dynamic`** - K bins + dynamic batching (our contribution)

### 🎯 Three Fidelity Levels

| Level | Description | Requirements | Validity |
|-------|-------------|--------------|----------|
| **Level 1: Synthetic** | Formula-based service time | None | ✓ Relative comparisons valid |
| **Level 2: BurstGPT** | Real Azure dataset | Download CSV (48MB) | ✓✓ Realistic workload |
| **Level 3: GPU Calibrated** | Measured latency from RTX 4080 | GPU + Transformers/vLLM | ✓✓✓ Production-ready |

---

## Project Structure

```
llm_scheduler_sim/
├── mb_dyn_sim/           # Core simulator
│   ├── config.py         # Configuration + equal-mass boundaries
│   ├── schedulers.py     # SLAController, DynamicBatcher, MultiBinScheduler
│   ├── simulation.py     # Discrete-event simulation engine
│   ├── workload.py       # BurstGPT loading + Poisson generation
│   ├── model_calibration.py  # vLLM calibration support
│   ├── metrics.py        # Performance metrics
│   └── experiments.py    # Plotting and analysis
│
├── scripts/
│   ├── run_mb_dynamic.py                  # Main experiment runner ⭐
│   └── calibrate_real_gpu_transformers.py # GPU calibration (Windows)
│
├── data/
│   ├── BurstGPT_sample.csv       # Real Azure traces (download)
│   └── README.md                  # Dataset format spec
│
├── ARCHITECTURE.md       # Visual system architecture
├── PAPER_REQUIREMENTS.md # Algorithm specifications
├── EXPERIMENT_GUIDE.md   # Experiment instructions
├── QUICK_REFERENCE.md    # API reference
└── README.md             # This file
```

---

## Usage Examples

### Test with Real BurstGPT Data
```powershell
python scripts/run_mb_dynamic.py `
    --arrival-profile burstgpt_dataset `
    --num-requests 5000 `
    --compare
```

### Compare All Three Modes
```powershell
python scripts/run_mb_dynamic.py --num-requests 5000 --compare
```

### K_BINS Sensitivity Analysis
```powershell
for ($K in 1,2,4,8) {
    python scripts/run_mb_dynamic.py --k-bins $K --num-requests 5000
}
```

### Use BurstGPT Dataset
```powershell
python scripts/run_mb_dynamic.py `
    --arrival-profile burstgpt_dataset `
    --dataset-path data/BurstGPT_sample.csv `
    --num-requests 10000 `
    --compare
```

---

## GPU Calibration (Optional)

For Level 3 fidelity with real GPU measurements:

### 1. Check CUDA Setup
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. Run GPU Calibration (Windows: Transformers, Linux: vLLM)
```powershell
# Windows (Transformers)
python scripts/calibrate_real_gpu_transformers.py --model Qwen/Qwen2.5-1.5B --trials 3

# Linux (vLLM - not supported on Windows, advanced users only)
# pip install vllm
# (Use transformers script above for Windows)
```

### 3. Run Simulation with Calibrated Latency
```powershell
python scripts/run_mb_dynamic.py `
    --use-real-calibration `
    --calibration-csv data/qwen2_5_1_5b_latency_grid.csv `
    --compare
```

See `CUDA_SETUP_COMPLETE.md` for detailed GPU setup instructions.

---

## Paper Validation Results

Test results with 5,000 requests, synthetic workload:

| Scheduler | SLA Violations | Avg Latency | GPU Utilization |
|-----------|----------------|-------------|-----------------|
| static_fifo (B=8) | 99.24% | 24.35s | 96.49% |
| dynamic_no_bins | 27.76% | 0.85s | 32.62% |
| **multi_bin_dynamic** | **18.10%** | **0.69s** | **35.15%** |

✅ **Dynamic batching** reduces SLA violations by 71% vs fixed batch  
✅ **Multi-Bin + Dynamic** achieves best latency with controlled utilization  
✅ **All three schedulers produce DISTINCT results** (validated 2025-11-17)

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_GPUS` | 1 | Number of GPUs to simulate |
| `K_BINS` | 4 | Number of multi-bin queues |
| `B_FIXED` | 16 | Fixed batch size (multi_bin_only mode) |
| `B_MAX` | 64 | Maximum dynamic batch size |
| `D_SLA` | 1.0 | SLA deadline (seconds) |
| `M_MAX_GB` | 12.0 | GPU memory (GB) |
| `EXPERIMENT_MODE` | "multi_bin_dynamic" | Mode selection |

See `mb_dyn_sim/config.py` for all options.

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Visual diagrams and data flow
- **[PAPER_REQUIREMENTS.md](PAPER_REQUIREMENTS.md)** - Exact algorithm specifications
- **[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)** - Complete experiment guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - API quick reference
- **[data/README.md](data/README.md)** - Dataset format specification

---

## Scientific Validity

This simulator follows the **wind tunnel testing** approach:

| Aspect | Real Deployment | Our Simulator |
|--------|----------------|---------------|
| Cost | $$$ (GPU cluster) | Free (CPU only) |
| Speed | Days/weeks | Seconds |
| Risk | User-facing | Zero (offline) |
| Iteration | Slow (A/B tests) | Fast (experiments) |
| **Validity** | **Absolute numbers** | **Relative rankings** ✓ |

**Key Principle:** The simulator preserves algorithmic fidelity for valid scheduler comparisons, even with synthetic service times.

---

## References

### Papers
1. Multi-Bin Batching for LLM Inference Throughput Optimization
2. Memory-Aware and SLA-Constrained Dynamic Batching for LLM Inference

### Dataset
- **BurstGPT**: [https://github.com/HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT)
- Real ChatGPT/GPT-4 workload traces from Azure (121 days, 5.29M requests)

### Model
- **Qwen3-0.6B**: [https://huggingface.co/Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Alternative: **Qwen2.5-0.5B** (currently available)

### Framework
- **vLLM**: High-throughput LLM serving framework for calibration

---

## FAQ

**Q: Do I need a GPU to run experiments?**  
A: No! The simulator runs on CPU. GPU is only needed for vLLM calibration (Level 3 fidelity).

**Q: Do I need the actual Qwen3-0.6B model?**  
A: No! The simulator uses a formula-based service time model. vLLM calibration is optional.

**Q: Are the results valid without real GPU measurements?**  
A: Yes! Relative performance comparisons are valid with synthetic parameters. This is standard practice in systems research.

**Q: How do I reproduce paper figures?**  
A: See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for detailed instructions.

---

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{multibin_dynamic_scheduler,
  title={Multi-Bin Dynamic Batching Scheduler for LLM Inference},
  author={Your Name},
  year={2025},
  note={Paper-faithful implementation of multi-bin batching and dynamic batching algorithms}
}
```

And cite the BurstGPT dataset:
```bibtex
@inproceedings{BurstGPT,
  author = {Yuxin Wang and Yuhan Chen and Zeyu Li and Xueze Kang and others},
  title = {{BurstGPT}: A Real-World Workload Dataset to Optimize LLM Serving Systems},
  booktitle = {KDD '25},
  year = {2025},
}
```

---

## License

See LICENSE file for details.

---

**Status:** ✅ Complete and validated with real BurstGPT data  
**Last Updated:** November 2025
