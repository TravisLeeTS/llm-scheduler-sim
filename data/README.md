# LLM Scheduler Simulation - Dataset Directory

This directory contains datasets and GPU calibration files for the LLM scheduler simulator.

---

## Files

### 1. BurstGPT Dataset
**File**: `BurstGPT_sample.csv`

**Source**: [HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT) - Official Repository

Real Azure ChatGPT/GPT-4 trace data from Microsoft Azure containing request arrival patterns and token lengths.

**Citation**:
```bibtex
@inproceedings{BurstGPT,
  author    = {Yuxin Wang and Yuhan Chen and Zeyu Li and Xueze Kang and Yuchu Fang and Yeju Zhou and Yang Zheng and Zhenheng Tang and Xin He and Rui Guo and Xin Wang and Qiang Wang and Amelie Chi Zhou and Xiaowen Chu},
  title     = {{BurstGPT}: A Real-World Workload Dataset to Optimize LLM Serving Systems},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25)},
  year      = {2025},
  address   = {Toronto, ON, Canada},
  publisher = {ACM},
  doi       = {https://doi.org/10.1145/3711896.3737413},
}
```

**Paper**: [arXiv:2401.17644](https://arxiv.org/pdf/2401.17644.pdf)

## BurstGPT Dataset Schema

The dataset follows the official BurstGPT format:

| Column | Description |
|--------|-------------|
| `Timestamp` | Request submission time (seconds from 0:00:00 on first day) |
| `Model` | Called model: `ChatGPT` (GPT-3.5) or `GPT-4` |
| `Request tokens` | Input/prompt token length |
| `Response tokens` | Output/completion token length |
| `Total tokens` | Request + Response tokens |
| `Log Type` | `Conversation log` or `API log` |

**Dataset Characteristics**:
- Duration: 121 consecutive days across 4 months
- Total size: ~5.29M lines (~188MB full dataset)
- Current file: `BurstGPT_1.csv` (first 2 months, ~1.43M lines)

---

### 2. GPU Calibration File
**File**: `qwen3_1_7b_latency_grid.csv`

**Model**: Qwen3-1.7B ([Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B))  
**Hardware Target**: RTX 4080 12GB  
**Calibration Source**: [Qwen Speed Benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)

**Format**:
```csv
batch_size,max_seq_len,mean_latency_sec,std_latency_sec,num_trials
1,128,0.80,0.056,5
1,256,1.55,0.109,5
...
```

**Calibration Matrix**:
- Batch sizes: 1, 2, 4, 8, 16, 32
- Sequence lengths: 128, 256, 512, 1024, 2048  
- Trials per config: 5
- Total measurements: 30 configurations

**Latency Model**: `T(b, L) = α + β·L·(1 + γ·(b-1)/b)`

Fitted parameters (R² = 0.9995):
- α = 59.65ms (base latency)
- β = 5.74ms/token (per-token coefficient)
- γ = 0.316 (batch penalty factor)

See [LATENCY_GRID_REVISION.md](../LATENCY_GRID_REVISION.md) for calibration details.

---

## How to Download BurstGPT Dataset

### Option 1: Download from GitHub Releases (Recommended)

1. Go to [BurstGPT Releases](https://github.com/HPMLL/BurstGPT/releases/tag/v1.1)
2. Download one of:
   - `BurstGPT_1.csv` - First 2 months with failures (~1.43M lines)
   - `BurstGPT_without_fails_1.csv` - First 2 months without failures (~1.40M lines)
   - `BurstGPT_2.csv` - Second 2 months with failures (~3.86M lines)
   - `BurstGPT_without_fails_2.csv` - Second 2 months without failures (~3.78M lines)
3. Save to this directory as `BurstGPT_sample.csv`

### Option 2: Clone Repository

```bash
git clone https://github.com/HPMLL/BurstGPT.git
cp BurstGPT/data/BurstGPT_1.csv data/BurstGPT_sample.csv
```

## Running with Dataset

Once you have the dataset file:

```bash
# Run quick validation test
python scripts/quick_validation_test.py

# Run comprehensive stress test
python scripts/comprehensive_stress_test_optimized.py

# Or run specific scheduler
python scripts/run_mb_dynamic.py \
    --arrival-profile burstgpt_dataset \
    --dataset-path data/BurstGPT_sample.csv \
    --num-requests 10000 \
    --compare
```

## License

BurstGPT dataset is released under [CC-BY-4.0 license](https://github.com/HPMLL/BurstGPT/blob/main/LICENSE).
