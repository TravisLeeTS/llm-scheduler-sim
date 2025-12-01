# GPU Specifications and Memory Model

This document details the GPU hardware specifications and memory constraints used in the LLM scheduler simulation.

---

## Hardware Configuration

### GPU: NVIDIA RTX 4080 12GB

- **Model**: NVIDIA GeForce RTX 4080
- **VRAM**: 12 GB GDDR6X
- **CUDA Cores**: 9,728
- **Architecture**: Ada Lovelace (Compute Capability 8.9)
- **Memory Bandwidth**: 716.8 GB/s
- **TDP**: 320W

**Why RTX 4080?**
- Consumer-grade GPU (accessible for research)
- 12GB VRAM sufficient for 1.5B parameter models
- Excellent price/performance for LLM inference
- Good availability compared to datacenter GPUs

---

## Language Model: Qwen3 1.7B

### Model Architecture

- **Model Family**: Qwen3 (Alibaba Cloud)
- **Parameters**: 1.7 Billion
- **Hidden Size**: 2,048
- **Intermediate Size**: 11,008
- **Layers**: 24
- **Attention Heads**: 16
- **Head Dimension**: 128
- **Vocabulary Size**: 151,936

### Precision: FP16 (Half Precision)

- **Format**: IEEE 754 half-precision floating-point
- **Bits per parameter**: 16 bits = 2 bytes
- **Memory per parameter**: 2 bytes

---

## Memory Footprint Calculation

### 1. Model Weights (Static)

**Formula**: `model_size = num_parameters × bytes_per_parameter`

```
Parameters:     1.7B
Precision:      FP16 (2 bytes)
Model Size:     1.7B × 2 bytes = 3.4 GB
Overhead:       ~18% (optimizer states, buffers)
Total:          4.0 GB
```

**Config Parameter**: `M_MODEL_GB = 4.0`

### 2. KV Cache Memory (Dynamic)

The KV cache stores Key and Value tensors for all tokens in the batch during autoregressive generation.

**Formula**: 
```
kv_cache_per_token = 2 (K+V) × num_layers × hidden_size × bytes_per_element
```

**For Qwen3 1.7B FP16:**
```
KV cache/token = 2 × 24 layers × 2048 hidden × 2 bytes
               = 196,608 bytes
               = 0.1875 MB
               = 0.0001875 GB
               = 1.875e-4 GB
```

**Config Parameter**: `KV_MEM_PER_TOKEN_GB = 1.875e-4`

**Previous (INCORRECT) value**: `5e-6 GB` (37.5x too small!)

### 3. Total Memory Budget

**Available for KV cache:**
```
M_available = M_MAX_GB - M_MODEL_GB
            = 12.0 GB - 4.0 GB
            = 8.0 GB
```

**Maximum tokens in memory (η):**
```
η = M_available / KV_MEM_PER_TOKEN_GB
  = 8.0 GB / 1.875e-4 GB
  = 42,667 tokens
```

**With 10% safety buffer (L₀):**
```
L₀ = 0.1 × η = 4,267 tokens
Usable capacity: η - L₀ = 38,400 tokens
```

---

## Dynamic Batching Memory Constraint

### Algorithm 1: Memory-Aware Batch Sizing

**From Paper**: *Memory-Aware and SLA-Constrained Dynamic Batching*

```python
def compute_b_mem(stats: BatchStatistics, cfg: SchedulerConfig) -> int:
    """
    Compute maximum batch size based on GPU memory constraint.
    
    η = (M_max - M_model) / kv_mem_per_token  # Total token capacity
    μ = avg(prompt_len + output_len)          # Avg tokens per request
    L₀ = 0.1 × η                              # 10% safety buffer
    b_mem = ⌊(η - L₀) / μ⌋                    # Max batch size
    """
    eta = (cfg.M_MAX_GB - cfg.M_MODEL_GB) / cfg.KV_MEM_PER_TOKEN_GB
    avg_tokens = stats.avg_prompt_len + stats.avg_output_len
    L0 = 0.1 * eta  # Safety margin
    b_mem = int((eta - L0) / avg_tokens)
    return max(cfg.B_MIN, min(b_mem, cfg.B_MAX))
```

### Example Calculation

**Assumptions:**
- Average prompt length: 200 tokens
- Average output length: 300 tokens
- Total per request: 500 tokens

**With CORRECT KV cache coefficient (1.875e-4):**
```
η = 42,667 tokens
L₀ = 4,267 tokens (safety buffer)
μ = 500 tokens/request
b_mem = ⌊(42,667 - 4,267) / 500⌋ = ⌊76.8⌋ = 76 requests
```

**With INCORRECT coefficient (5e-6):**
```
η = 1,600,000 tokens (UNREALISTIC!)
L₀ = 160,000 tokens
μ = 500 tokens/request
b_mem = ⌊1,440,000 / 500⌋ = 2,880 requests (IMPOSSIBLE!)
```

**Impact**: The incorrect coefficient made the memory constraint ineffective, allowing the simulator to schedule impossibly large batches.

---

## Calibration Data

### File: `data/qwen3_1_7b_latency_grid.csv`

**Model**: Qwen3 1.7B FP16

**Calibration Matrix:**
- Batch sizes: 1, 2, 4, 8, 16, 32
- Sequence lengths: 128, 256, 512, 1024, 2048
- Trials per config: 3
- Total measurements: 30 configurations

**Sample measurements (batch=8, seq_len=512):**
```
batch_size=8, max_seq_len=512, mean_latency=0.222s, std=0.011s
```

**Latency Model**: `T(b, L) = α + β·L·(1 + γ·(b-1)/b)`

Fitted parameters:
- `α = 0.010s` (base latency)
- `β = 0.0002s/token` (per-token coefficient)
- `γ = 0.3` (batch penalty factor)

---

## Validation: Memory vs Latency Constraints

### Realistic Workload (BurstGPT Dataset)

**Observed distribution:**
```
Prompt length:  mean=611, p50=251, p95=2461
Output length:  mean=123, p50=34,  p95=460
Total tokens:   mean=734, p50=327, p95=2801
```

### Effective Batch Size Limits

**Memory constraint (b_mem):**
```
For avg request (400 tokens): b_mem ≈ 96
For p95 request (1000 tokens): b_mem ≈ 38
```

**SLA constraint (b_SLA):**
```
For D_SLA = 1.0s, typical b_SLA ≈ 20-30 (depends on queue state)
```

**Final batch size**: `b_target = min(b_mem, b_SLA, B_MAX)`

**Observation**: With correct memory parameters, SLA constraint is typically the bottleneck (b_SLA < b_mem), which matches real production systems!

---

## Configuration Summary

### Updated `mb_dyn_sim/config.py`

```python
# GPU Infrastructure (RTX 4080 12GB)
NUM_GPUS: int = 4
M_MAX_GB: float = 12.0                  # Total VRAM
M_MODEL_GB: float = 4.0                 # Qwen3 1.7B FP16
KV_MEM_PER_TOKEN_GB: float = 1.875e-4   # 0.1875 MB/token (CORRECTED)

# Memory constraint safety margin
MEMORY_MARGIN_GB: float = 1.0

# Calibration file
USE_REAL_CALIBRATION: bool = True
CALIBRATION_CSV_PATH: str = "data/qwen3_1_7b_latency_grid.csv"
```

### Memory Budget Breakdown

```
Total VRAM:        12.0 GB (100%)
├─ Model weights:   4.0 GB ( 33%)
├─ KV cache:        7.0 GB ( 58%)  [up to ~37K tokens]
└─ Safety margin:   1.0 GB (  8%)
```

---

## Impact on Simulation

### Before Fix (KV_MEM_PER_TOKEN_GB = 5e-6)

❌ Memory constraint allowed 2880+ requests per batch (unrealistic)  
❌ SLA controller never hit memory limits  
❌ Batch sizes constrained only by B_MAX=128  
❌ Simulator didn't model real GPU memory pressure

### After Fix (KV_MEM_PER_TOKEN_GB = 1.875e-4)

✅ Memory constraint limits batches to 40-100 requests (realistic)  
✅ Both SLA and memory constraints are active  
✅ Matches real-world GPU behavior  
✅ Properly validates dynamic batching algorithms

---

## References

### Model Architecture
- [Qwen3 Technical Report](https://qwenlm.github.io/blog/qwen3/)
- [Hugging Face Model Card](https://huggingface.co/Qwen/Qwen3-1.7B)

### GPU Specifications
- [NVIDIA RTX 4080 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/)

### Papers
1. **Multi-Bin Batching**: Guldogan et al. - Bin-based request batching
2. **Dynamic Batching**: Memory-aware scheduling with SLA constraints

### Calibration Tools
- `scripts/calibrate_real_gpu_transformers.py` - HuggingFace Transformers-based calibration
- `mb_dyn_sim/model_calibration_transformers.py` - Measurement utilities

---

## FAQ

### Why FP16 instead of FP32?
- 2x memory savings (critical for 12GB GPU)
- Minimal accuracy loss for inference
- Native Tensor Core acceleration on Ada Lovelace

### Why not use INT8 quantization?
- FP16 is standard for production LLM inference
- INT8 requires additional calibration steps
- FP16 provides better accuracy baseline

### Can I use a different GPU?
Yes! Update these parameters in `config.py`:
```python
M_MAX_GB = <your_gpu_vram_gb>
```
Then re-run calibration:
```bash
python scripts/calibrate_real_gpu_transformers.py
```

### Can I use a different model?
Yes! Calculate KV cache coefficient:
```python
kv_mem_per_token = 2 × num_layers × hidden_size × 2 bytes / (1024^3)
# Example for Qwen3 1.7B:
# kv_mem_per_token = 2 × 24 × 2048 × 2 / (1024^3) = 1.875e-4 GB
```
Update `config.py` and re-calibrate latency grid.

---

**Last Updated**: November 27, 2025  
**Hardware**: RTX 4080 12GB  
**Model**: Qwen3 1.7B FP16  
**Calibration File**: `data/qwen3_1_7b_latency_grid.csv`
