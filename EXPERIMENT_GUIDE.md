# Experiment Guide

This guide shows how to run experiments that reproduce the paper results.

## Quick Start

### 1. Test Paper-Faithful Implementation

```bash
# Run comprehensive tests
python scripts/test_paper_faithful.py
```

This validates:
- ✓ Equal-mass bin boundaries
- ✓ Multi-bin throughput scaling  
- ✓ Dynamic batching adaptation
- ✓ Three experiment modes
- ✓ Poisson arrival generation

### 2. Run Basic Comparison

```bash
# Compare all three modes
python scripts/run_mb_dynamic.py --num-requests 1000 --compare
```

This produces:
- Throughput comparison plots
- Latency distribution plots
- SLA violation rates

## Experiment Fidelity Levels

### Level 1: Synthetic (Default)

**Purpose**: Quick testing and algorithm validation

```bash
python scripts/run_mb_dynamic.py \
    --num-requests 1000 \
    --experiment-mode multi_bin_dynamic \
    --compare
```

**Characteristics**:
- Poisson arrivals (λ = 50 req/s)
- Synthetic length distributions
- Idealized service time model
- Fast execution (~seconds)

### Level 2: BurstGPT Dataset

**Purpose**: Realistic workload patterns from Azure traces

```bash
# Generate synthetic BurstGPT-like dataset
python scripts/create_synthetic_dataset.py \
    --output data/azure_trace.csv \
    --num-requests 5000

# Run with dataset
python scripts/run_mb_dynamic.py \
    --arrival-profile burstgpt_dataset \
    --dataset-path data/azure_trace.csv \
    --compare
```

**Characteristics**:
- Real arrival patterns (bursty)
- Real length distributions
- Idealized service time model
- Medium execution (~minutes)

### Level 3: vLLM Calibration

**Purpose**: GPU-accurate latency measurements with real model

```bash
# 1. Install vLLM (requires CUDA GPU)
pip install vllm

# 2. Calibrate with Qwen3-0.6B
python scripts/calibrate_real_gpu_transformers.py \
    --model Qwen/Qwen2.5-0.5B \
    --output data/vllm_calibration.json

# 3. Run experiments with calibrated parameters
python scripts/run_mb_dynamic.py \
    --use-vllm \
    --num-requests 1000 \
    --compare
```

**Characteristics**:
- Poisson or BurstGPT arrivals
- Real GPU latency measurements
- Calibrated to specific hardware
- Slow execution (~hours for calibration)

## Paper Validation Experiments

### Experiment 1: Multi-Bin Throughput Scaling

**Paper Claim**: "Throughput increases with K_BINS due to reduced max-sequence-length variance within bins"

```bash
# Test K_BINS = 1, 2, 4, 8
for K in 1 2 4 8; do
    python scripts/run_mb_dynamic.py \
        --experiment-mode multi_bin_only \
        --k-bins $K \
        --num-requests 5000 \
        --output plots/k_bins_${K}.png
done
```

**Expected Result**: Throughput should increase with K_BINS

### Experiment 2: SLA Controller Effectiveness

**Paper Claim**: "SLA controller maintains latency < D_SLA while maximizing batch size"

```bash
# Test different SLA targets
for SLA in 0.5 1.0 2.0; do
    python scripts/run_mb_dynamic.py \
        --experiment-mode multi_bin_dynamic \
        --sla-target $SLA \
        --num-requests 5000 \
        --output plots/sla_${SLA}.png
done
```

**Expected Result**: Tighter SLA → smaller batches → lower latency but lower throughput

### Experiment 3: Three-Way Comparison

**Paper Claim**: "Multi-Bin + Dynamic (ours) outperforms both baselines"

```bash
python scripts/run_mb_dynamic.py \
    --num-requests 10000 \
    --compare \
    --output plots/three_way_comparison.png
```

**Expected Results**:
- `multi_bin_only`: High throughput, poor latency consistency
- `dynamic_only`: Good latency control, moderate throughput  
- `multi_bin_dynamic`: Best of both worlds

## Advanced Options

### Equal-Mass Bin Boundaries

```bash
# Analyze optimal boundaries for your workload
python scripts/prepare_bins.py

# Use equal-mass bins (default)
python scripts/run_mb_dynamic.py --use-equal-mass-bins
```

### Custom Service Time Parameters

```bash
# Override default latency model
python scripts/run_mb_dynamic.py \
    --a0 0.05 \
    --a1 0.0001 \
    --beta 0.3 \
    --num-requests 1000
```

### Poisson Arrivals with Custom Rate

```bash
python scripts/run_mb_dynamic.py \
    --arrival-profile poisson \
    --poisson-lambda 100.0 \
    --num-requests 5000
```

### Memory Constraint Testing

```bash
# Test different GPU memory limits
for MEM in 16 24 32 40; do
    python scripts/run_mb_dynamic.py \
        --max-memory ${MEM} \
        --num-requests 1000 \
        --output plots/mem_${MEM}GB.png
done
```

## Reproducing Paper Figures

### Figure 1: Throughput vs K_BINS

```bash
python scripts/run_mb_dynamic.py \
    --experiment-mode multi_bin_only \
    --k-bins 1 \
    --num-requests 10000 \
    --seed 42 \
    --output plots/fig1_k1.png

python scripts/run_mb_dynamic.py \
    --experiment-mode multi_bin_only \
    --k-bins 4 \
    --num-requests 10000 \
    --seed 42 \
    --output plots/fig1_k4.png
```

### Figure 2: Latency Distribution

```bash
python scripts/run_mb_dynamic.py \
    --num-requests 10000 \
    --compare \
    --output plots/fig2_latency.png
```

### Figure 3: SLA Violation Rate

```bash
python scripts/run_mb_dynamic.py \
    --experiment-mode multi_bin_dynamic \
    --sla-target 1.0 \
    --num-requests 10000 \
    --output plots/fig3_sla.png
```

## Common Issues

### Issue: "No module named 'vllm'"

**Solution**: vLLM is optional. Either:
1. Skip Level 3 experiments: Don't use `--use-vllm`
2. Install vLLM: `pip install vllm` (requires CUDA GPU)

### Issue: "Equal-mass boundaries result in empty bins"

**Solution**: Reduce K_BINS or increase NUM_REQUESTS:

```bash
python scripts/run_mb_dynamic.py \
    --k-bins 2 \
    --num-requests 5000
```

### Issue: "Throughput not scaling with K_BINS"

**Possible causes**:
1. Too few requests (need >1000 per bin)
2. Service time model not calibrated
3. Statistical noise (run multiple trials)

**Solution**: Run with more requests and average over seeds:

```bash
for SEED in 42 123 999 777 555; do
    python scripts/run_mb_dynamic.py \
        --experiment-mode multi_bin_only \
        --k-bins 4 \
        --num-requests 10000 \
        --seed $SEED \
        --output plots/k4_seed_${SEED}.png
done
```

### Issue: "SLA violation rate too high"

**Solution**: Increase SLA target or reduce arrival rate:

```bash
python scripts/run_mb_dynamic.py \
    --sla-target 2.0 \
    --poisson-lambda 30.0 \
    --num-requests 1000
```

## Output Files

All experiments produce:

1. **Console Output**: Summary statistics (throughput, latency, SLA violations)
2. **Plots**: Visualization saved to `--output` path or `plots/` directory
3. **CSV (optional)**: Raw data for further analysis

Example:

```bash
python scripts/run_mb_dynamic.py \
    --num-requests 1000 \
    --compare \
    --output plots/experiment_2024.png

# Produces:
# - plots/experiment_2024.png (visualization)
# - Console output with metrics
```

## Next Steps

1. **Start Simple**: Run `test_paper_faithful.py` to validate implementation
2. **Basic Experiments**: Use synthetic workload with `--compare`
3. **Realistic Workload**: Generate BurstGPT dataset and test
4. **GPU Calibration**: If you have CUDA GPU, calibrate with vLLM
5. **Paper Reproduction**: Run the paper validation experiments above

For more details, see:
- `PAPER_REQUIREMENTS.md` - Algorithm specifications
- `IMPLEMENTATION_PLAN.md` - Code structure
- `ARCHITECTURE.md` - System design
- `QUICK_REFERENCE.md` - API reference
