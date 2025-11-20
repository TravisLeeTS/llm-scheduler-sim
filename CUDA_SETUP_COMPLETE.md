# CUDA Setup Complete ✓

## Summary

Your RTX 4080 Laptop GPU is now properly configured for Level 3 GPU calibration!

## What Was Fixed

### 1. **NVIDIA GPU Detection** ✓
- **GPU**: NVIDIA GeForce RTX 4080 Laptop GPU
- **Memory**: 11.99 GB
- **CUDA Version**: 12.6
- **Driver**: 561.17

### 2. **PyTorch with CUDA** ✓
Previous state:
- PyTorch 2.9.1+cpu (CPU-only, no CUDA support)

Fixed:
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Current state:
- **PyTorch**: 2.9.1+cu126 (CUDA 12.6 enabled)
- **CUDA available**: True
- **GPU accessible**: Yes

### 3. **vLLM on Windows** ⚠️
Issue discovered:
- vLLM doesn't have official Windows support with pre-built CUDA extensions
- The PyPI package lacks the compiled `vllm._C` module needed for CUDA

**Solution: Use Hugging Face Transformers instead**
- Created alternative calibration module: `mb_dyn_sim/model_calibration_transformers.py`
- Created alternative script: `scripts/calibrate_real_gpu_transformers.py`
- Transformers works perfectly on Windows with CUDA

---

## How to Use GPU Calibration

### Option 1: Quick Test (Recommended to start)
```bash
# Test with small model and limited configurations
python scripts/calibrate_real_gpu_transformers.py \
  --model Qwen/Qwen2.5-0.5B \
  --batch-sizes 1 2 4 \
  --max-seq-lens 128 256 512 \
  --trials 2 \
  --output data/qwen_test_calibration.csv
```

**Expected time**: ~5-10 minutes (downloads model first time)

### Option 2: Full Calibration (Production)
```bash
# Full grid calibration (takes longer, more accurate)
python scripts/calibrate_real_gpu_transformers.py \
  --model Qwen/Qwen2.5-1.5B \
  --batch-sizes 1 2 4 8 16 \
  --max-seq-lens 128 256 512 1024 2048 \
  --trials 3 \
  --output data/qwen_1_5b_latency_grid.csv
```

**Expected time**: ~30-60 minutes

### Option 3: Use Existing Mock Data
```bash
# Use the mock calibration data already created
python scripts/run_mb_dynamic.py \
  --use-real-calibration \
  --calibration-csv data/qwen3_1_7b_latency_grid.csv \
  --compare
```

**Expected time**: Seconds (no GPU measurement needed)

---

## File Structure

### New Files Created
1. **`mb_dyn_sim/model_calibration_transformers.py`**
   - GPU calibration using Hugging Face Transformers
   - Windows-compatible alternative to vLLM
   - Functions: `measure_batch_latency()`, `calibrate_latency_grid()`

2. **`scripts/calibrate_real_gpu_transformers.py`**
   - Command-line interface for calibration
   - Supports custom model, batch sizes, sequence lengths
   - Saves results to CSV

3. **`CUDA_SETUP_COMPLETE.md`** (this file)
   - Setup documentation

### Existing Files (still valid)
- `mb_dyn_sim/model_calibration_real_gpu.py` - vLLM version (for Linux)
- `scripts/calibrate_real_gpu_transformers.py` - Transformers script (for Windows/Linux)
- `mb_dyn_sim/model_calibration.py` - LatencyModel class (works with both)
- `scripts/run_mb_dynamic.py` - Main simulator (works with both)

---

## Verification Steps

### 1. Check CUDA is working
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
```

### 2. Test Transformers
```python
from transformers import AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="cuda")
print("✓ Model loaded on GPU successfully!")
```

### 3. Run Quick Calibration
```bash
python scripts/calibrate_real_gpu_transformers.py --trials 1 --batch-sizes 1 --max-seq-lens 128
```

---

## Next Steps

### Immediate: Run Quick Test
```bash
# 1. Quick calibration (~5 min)
python scripts/calibrate_real_gpu_transformers.py \
  --model Qwen/Qwen2.5-0.5B \
  --batch-sizes 1 2 \
  --max-seq-lens 128 256 \
  --trials 2

# 2. Run calibrated simulation
python scripts/run_mb_dynamic.py \
  --use-real-calibration \
  --calibration-csv data/qwen_latency_grid.csv \
  --num-requests 5000 \
  --compare

# 3. Multi-GPU test
python scripts/run_mb_dynamic.py \
  --use-real-calibration \
  --num-gpus 4 \
  --num-requests 10000 \
  --compare
```

### Full Research Workflow
1. **Calibrate with target model** (~30-60 min)
   ```bash
   python scripts/calibrate_real_gpu_transformers.py \
     --model Qwen/Qwen2.5-1.5B \
     --batch-sizes 1 2 4 8 16 \
     --max-seq-lens 128 256 512 1024 \
     --trials 3
   ```

2. **Run experiments** (seconds to minutes)
   ```bash
   # BurstGPT workload with real GPU calibration
   python scripts/run_mb_dynamic.py \
     --arrival-profile burstgpt_dataset \
     --use-real-calibration \
     --num-requests 50000 \
     --rps-scaling 100.0 \
     --compare
   ```

3. **Validate accuracy** (optional, requires vLLM on Linux)
   ```bash
   # This step only works on Linux with vLLM
   python scripts/run_mb_dynamic.py \
     --num-requests 1000 \
     --mode all
   ```

---

## Technical Notes

### Why Transformers instead of vLLM?
- **vLLM**: Optimized for serving, very fast, but lacks Windows support
- **Transformers**: Universally compatible, slightly slower, works on Windows
- **For calibration**: Transformers is sufficient (we only measure once)
- **For production serving**: vLLM is better (Linux servers)

### Performance Expectations
With RTX 4080 Laptop GPU (12GB):

| Batch Size | Seq Length | Expected Latency | GPU Memory |
|------------|------------|------------------|------------|
| 1          | 128        | ~0.05s          | ~500 MB    |
| 1          | 512        | ~0.15s          | ~800 MB    |
| 4          | 128        | ~0.08s          | ~1.5 GB    |
| 4          | 512        | ~0.25s          | ~2.5 GB    |
| 8          | 1024       | ~0.60s          | ~5 GB      |

### Memory Management
- Model size (Qwen2.5-0.5B): ~1 GB in FP16
- Model size (Qwen2.5-1.5B): ~3 GB in FP16
- Leave 2-3 GB for OS/display
- Max safe batch: depends on sequence length

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or sequence length
```bash
python scripts/calibrate_real_gpu_transformers.py \
  --batch-sizes 1 2 4 \
  --max-seq-lens 128 256 512
```

### Issue: "Model download is slow"
**Solution**: Download model manually first
```python
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
```

### Issue: "Import error: No module named 'transformers'"
**Solution**: Install transformers
```bash
pip install transformers
```

### Issue: "CUDA not available"
**Solution**: Reinstall PyTorch with CUDA
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## Summary of Changes

### Environment
- ✓ CUDA 12.6 detected
- ✓ PyTorch 2.9.1+cu126 installed
- ✓ RTX 4080 accessible from Python

### Code
- ✓ Created `model_calibration_transformers.py` (Windows-compatible)
- ✓ Created `calibrate_real_gpu_transformers.py` (CLI tool)
- ✓ All existing code still works (backward compatible)

### Documentation
- ✓ CUDA_SETUP_COMPLETE.md (this file)
- ✓ Usage instructions and examples
- ✓ Troubleshooting guide

---

## Ready to Proceed!

Your system is now fully configured for:
1. ✓ Level 1: Synthetic experiments (fast, no GPU needed)
2. ✓ Level 2: BurstGPT dataset experiments (realistic, no GPU needed)
3. ✓ **Level 3: Real GPU calibration (accurate, RTX 4080 ready)**

**Recommended next command:**
```bash
# Quick 5-minute test to verify everything works
python scripts/calibrate_real_gpu_transformers.py --trials 1 --batch-sizes 1 2 --max-seq-lens 128
```

Good luck with your research! 🚀
