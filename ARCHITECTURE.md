# System Architecture (Implementation Reference)

## 🏗️ Overview

Three-layer LLM scheduler simulator implementing paper-faithful Multi-Bin and SLA-constrained Dynamic Batching components, combined into a novel hybrid scheduler:

1. **Algorithm Layer**: Paper-faithful building blocks (Multi-Bin binning + SLA dynamic batching controller)
2. **Latency Model Layer**: Synthetic → GPU-calibrated latency model
3. **Validation Layer**: Discrete-event simulation with Poisson and BurstGPT-like workloads

**Current Status:** ✅ Paper-faithful components implemented and validated; hybrid policy evaluated under realistic multi-GPU workloads

**Note:** The individual components (Multi-Bin binning, SLA controller) follow their respective papers. The `multi_bin_dynamic` mode combining both is our research contribution, evaluated beyond the original papers' single-server, Poisson assumptions.

---

## 🔧 Scheduler Modes (Current Implementation)

### 1. `static_fifo` - Fixed Batch Baseline
```
Scheduler:   StaticFIFOScheduler(fixed_batch_size=8)
Batcher:     None (no dynamic batching)
Binning:     No
SLA Control: No
```

**Behavior:**
- Single FIFO queue
- Always batches exactly 8 requests (or all available if <8)
- No memory or SLA constraints
- Simple baseline for comparison

### 2. `dynamic_no_bins` - Dynamic Single Queue
```
Scheduler:   DynamicNoBinsScheduler() 
Batcher:     DynamicBatcher(with SLA controller + memory constraint)
Binning:     No (single queue)
SLA Control: Yes (Algorithm 2)
```

**Behavior:**
- Single FIFO queue
- Dynamic batch sizing: `b_target = min(b_mem, b_SLA)`
- Memory constraint: `b_mem = ⌊(η-L₀)/μ_tokens⌋`
- SLA controller: Adaptive `[b_low, b_high]` search

### 3. `multi_bin_dynamic` - Hybrid System (Our Contribution)
```
Scheduler:   MultiBinScheduler(K_BINS=4, equal-mass boundaries)
Batcher:     DynamicBatcher(with SLA controller + memory constraint)  
Binning:     Yes (K_BINS configurable 1/2/4/8)
SLA Control: Yes (Algorithm 2-inspired heuristic)
```

**Behavior:**
- K separate FIFO queues (bins) by predicted output length
- Equal-mass bin boundaries from empirical quantiles (Multi-Bin Lemma 4.1)
- Bin selection: round-robin or longest-queue
- Dynamic batch sizing per bin using SLA feedback control

**Key Distinction:** This combines Multi-Bin binning (from Guldogan et al.) with SLA-constrained dynamic batching (from the SLA paper). Neither original paper analyzes this combination—it is our research contribution.

---

## 📊 System Architecture Diagram

```
┌────────────────────────────────────────────────────────┐
│           SIMULATION ENGINE (Discrete-Event)            │
│  Event Queue: [ARRIVAL @ 5.5s, GPU_FREE @ 5.8s]       │
│  Modes: static_fifo | dynamic_no_bins | multi_bin_dyn  │
└────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌──────────────────┐         ┌──────────────────────┐
│   WORKLOAD       │         │    SCHEDULER         │
│   GENERATOR      │─Request─│    (Queue Layer)     │
│                  │         │                      │
│  • Poisson(λ)    │         │  static_fifo:        │
│  • BurstGPT      │         │    • 1 FIFO queue    │
│    dataset       │         │                      │
│                  │         │  dynamic_no_bins:    │
│  Latency Model:  │         │    • 1 FIFO queue    │
│  • Synthetic     │         │                      │
│  • Calibrated    │         │  multi_bin_dynamic:  │
│    (GPU)         │         │    • K_BINS queues   │
└──────────────────┘         │    • Equal-mass bins │
                             └──────────────────────┘
                                        ↓
                   ┌────────────────────┴────────────────────┐
                   │   DYNAMIC BATCHER (if enabled)          │
                   │   ──────────────────────────────────    │
                   │   Algorithm 1: b_mem (memory limit)     │
                   │   Algorithm 2: b_SLA (SLA controller)   │
                   │   Final: b_target = min(b_mem, b_SLA)   │
                   └─────────────────────────────────────────┘
                                        ↓
                   ┌────────────────────┴────────────────────┐
                   │   SERVICE TIME ESTIMATOR                │
                   │   ──────────────────────────────────    │
                   │   T(b, L) = α + β·L·(1 + γ·(b-1)/b)    │
                   │                                          │
                   │   Level 1: Synthetic (α,β,γ hardcoded)  │
                   │   Level 3: GPU Calibrated (fitted)      │
                   └─────────────────────────────────────────┘
                                        ↓
                   ┌────────────────────┴────────────────────┐
                   │   MULTI-GPU STATE (k independent GPUs)  │
                   │   GPU 0: busy=True, free_at=5.23s       │
                   │   GPU 1: busy=False                     │
                   │   ...                                    │
                   └─────────────────────────────────────────┘
                                        ↓
                   ┌────────────────────┴────────────────────┐
                   │   METRICS & FEEDBACK                    │
                   │   • Latency, SLA violations             │
                   │   • GPU utilization                     │
                   │   • Feedback to SLA controller          │
                   └─────────────────────────────────────────┘
```

---

## 📁 Code Structure

```
llm_scheduler_sim/
├── mb_dyn_sim/
│   ├── config.py                  # SchedulerConfig dataclass
│   ├── workload.py                # Request generation + BurstGPT
│   ├── schedulers.py              # 3 scheduler modes + components
│   ├── simulation.py              # Discrete-event Simulator
│   ├── metrics.py                 # Metric computation
│   ├── model_calibration.py       # LatencyModel (scipy fitting)
│   ├── model_calibration_transformers.py  # GPU calibration
│   └── experiments.py             # Experiment runners
│
├── scripts/
│   ├── run_mb_dynamic.py                  # Main entry point
│   └── calibrate_real_gpu_transformers.py # GPU calibration (Windows)
│
├── data/
│   ├── BurstGPT_sample.csv        # Real Azure traces
│   └── *_latency_grid.csv         # Calibration data
│
├── README.md                      # Quick start
├── ARCHITECTURE.md                # This file
└── SCHEDULER_FIXES_SUMMARY.md     # Recent updates
```

---

## 🎯 Key Implementation Details

### Scheduler Initialization

```python
# simulation.py
if scheduler_type == "static_fifo":
    self.scheduler = StaticFIFOScheduler(cfg, fixed_batch_size=8)
    self.batcher = None
    
elif scheduler_type == "dynamic_no_bins":
    self.scheduler = DynamicNoBinsScheduler(cfg)
    self.batcher = DynamicBatcher(cfg, self.service_time_fn)
    
elif scheduler_type == "multi_bin_dynamic":
    self.scheduler = MultiBinScheduler(cfg)
    self.batcher = DynamicBatcher(cfg, self.service_time_fn)
```

### Batch Formation

```python
# _try_schedule_gpu in simulation.py
candidates = self.scheduler.get_candidates_for_gpu(gpu_id, MAX_CANDIDATES)

if self.batcher is None:
    # static_fifo: use all candidates
    batch = candidates
else:
    # dynamic batching
    batch, service_time = self.batcher.make_batch(current_time, candidates)
    
    # Return unused to queue
    unused = [c for c in candidates if c not in batch]
    for req in unused:
        self.scheduler.enqueue_request(req)
```

---

## ✅ Validation Results

**Test Configuration:** 1500 requests, 2 GPUs, BurstGPT-like arrivals, GPU-calibrated latency model

| Scheduler          | SLA Violations | Avg Latency | Throughput | GPU Util |
|--------------------|----------------|-------------|------------|----------|
| static_fifo        | 93.9%          | 5.81s       | 28.9 req/s | 85.9%    |
| dynamic_no_bins    | 75.4%          | 1.20s       | 35.2 req/s | 39.4%    |
| multi_bin_dynamic  | 27.9%          | 0.77s       | 35.2 req/s | 39.2%    |

**Key Findings:**
- ✓ All three schedulers produce DISTINCT results
- ✓ Dynamic batching: -20% SLA violations vs static (93.9% → 75.4%)
- ✓ Multi-bin + dynamic: -63% violations vs dynamic-only (75.4% → 27.9%)
- ✓ Multi-bin maintains throughput while significantly improving latency distribution

**Caveat:** These results evaluate our hybrid policy under multi-GPU, realistic workloads, not the single-server Poisson assumptions of the original Multi-Bin theory.

---

## 🔬 Three Evaluation Modes

### Mode 1: Synthetic Workload (Fast Iteration)
- **Speed:** Seconds
- **GPU:** Not required
- **Latency:** Synthetic model T(b,L) with hardcoded α,β,γ
- **Workload:** Poisson or BurstGPT-like synthetic arrivals
- **Use Case:** Algorithm development, policy comparison

```bash
python scripts/run_mb_dynamic.py --compare --num-requests 10000
```

### Mode 2: BurstGPT-like Workload
- **Speed:** Minutes  
- **GPU:** Not required
- **Latency:** Synthetic model
- **Workload:** BurstGPT-style ON/OFF arrivals or dataset traces
- **Use Case:** Realistic arrival patterns, bursty load testing

```bash
python scripts/run_mb_dynamic.py \
  --arrival-profile burstgpt_dataset \
  --num-requests 50000 \
  --rps-scaling 100.0 --compare
```

### Mode 3: GPU-Calibrated Latency Model
- **Speed:** Hours (one-time calibration) + Seconds (per simulation)
- **GPU:** RTX 4080 required for calibration
- **Latency:** Fitted T(b,L) from real GPU measurements (α,β,γ from regression)
- **Workload:** Any arrival pattern
- **Use Case:** Realistic end-to-end validation
- **Status:** Pipeline ready; currently using fitted model from `qwen3_1_7b_latency_grid.csv`

```bash
# Calibrate (one-time, ~30-45 min)
python scripts/calibrate_real_gpu_transformers.py \
  --model Qwen/Qwen2.5-1.5B --trials 3

# Simulate with calibrated model (seconds)
python scripts/run_mb_dynamic.py \
  --use-real-calibration \
  --calibration-csv data/qwen2_5_1_5b_latency_grid.csv \
  --compare
```

**Note:** The calibrated latency model uses the same parametric form T(b,L) as the synthetic model but with parameters fitted from actual GPU execution times, providing realistic relative performance between policies.

---

## 🧪 Implementation Fidelity and Assumptions

### What We Implement Faithfully

1. **Multi-Bin Binning (Guldogan et al.)**
   - ✅ K bins with separate FIFO queues
   - ✅ Equal-mass bin boundaries via empirical quantiles (Lemma 4.1)
   - ✅ Request assignment by predicted output length
   - ✅ K-sensitivity analysis (throughput vs K ∈ {1,2,4,8})

2. **SLA-Constrained Dynamic Batching**
   - ✅ Memory constraint: b_mem from token capacity
   - ✅ SLA feedback controller: adaptive batch size window [b_low, b_high]
   - ✅ Final batch size: min(b_mem, b_SLA)
   - ⚠️ Controller uses window-based heuristic, not exact Algorithm 2 step rule

3. **Discrete-Event Queueing Simulation**
   - ✅ Arrival and GPU_FREE events
   - ✅ Per-request queueing delay and service time tracking
   - ✅ Configurable arrival processes (Poisson, BurstGPT-like)
   - ✅ Parametric latency model T(b,L) = α + β·L·(1 + γ·(b-1)/b)

### Where We Deviate from Original Papers

| Aspect | Multi-Bin Paper | SLA Paper | Our Implementation |
|--------|----------------|-----------|-------------------|
| **Servers** | Single (M/G/1) | Single queue | 1-4 GPUs (configurable) |
| **Arrivals** | Poisson | Not specified | Poisson + BurstGPT-like |
| **Service Times** | Uniform U(l_min, l_max) | Empirical | Fitted T(b,L) model |
| **Batch Size** | Fixed B | Dynamic b_t | Fixed OR dynamic (mode-dependent) |
| **SLA Definition** | N/A | Decode-phase latency | End-to-end latency |
| **Binning** | Yes (K bins) | No | Yes (in multi_bin modes) |

### What Is Novel (Our Contribution)

The `multi_bin_dynamic` policy combines:
- Multi-Bin's equal-mass binning strategy
- SLA-constrained dynamic batching controller
- Multi-GPU scheduling

**Neither original paper analyzes this combination.** Our evaluation shows the hybrid reduces SLA violations by 63% vs dynamic-only under realistic multi-GPU, bursty workloads.

### Scientific Validity for Policy Comparison

**Principle:** Relative performance rankings hold even with model approximations, provided all policies use the same model.

**Example from our results:**
```
Synthetic Model:
  static_fifo:       35.6 req/s, 76.9% violations
  multi_bin_dynamic: 43.6 req/s, 3.9% violations  (→ +22% throughput, -95% violations)
  
GPU-Calibrated Model:
  static_fifo:       28.9 req/s, 93.9% violations
  multi_bin_dynamic: 35.2 req/s, 27.9% violations (→ +22% throughput, -70% violations)

→ Absolute numbers differ
→ Relative improvements consistent
→ Valid for policy comparison (standard in ns-3, DiskSim, CloudSim)
```

---

## 📚 References and Paper Alignment

### Paper-Faithful Components

1. **Multi-Bin Batching (Guldogan et al.)**
   - Implementation: `MultiBinScheduler` with equal-mass boundaries
   - Faithfulness: Binning algorithm and data structures match paper
   - Deviation: Evaluated on multi-GPU, non-Poisson workloads (theory assumes M/G/1)

2. **SLA-Constrained Dynamic Batching**
   - Implementation: `DynamicBatcher` with memory and SLA constraints
   - Faithfulness: Two-layer control (b_mem, b_SLA) matches conceptual design
   - Deviation: Heuristic controller vs exact Algorithm 2; end-to-end SLA vs decode-only

3. **BurstGPT Workload Patterns**
   - Implementation: ON/OFF arrival model and dataset trace support
   - Use: Realistic bursty traffic for stress testing

### Novel Contributions

- **Hybrid Policy:** Multi-Bin + SLA dynamic batching combination
- **Multi-GPU Extension:** K bins serving N GPUs (papers analyze single server)
- **End-to-End SLA:** Queueing + service time (stricter than decode-only)

### To Claim "Paper Reproduction" (Future Work)

To fully reproduce the original papers' theoretical results, we would need:

1. **Multi-Bin Theory Validation**
   - Single GPU (NUM_GPUS=1)
   - Poisson arrivals
   - Uniform service times U(l_min, l_max)
   - Fixed batch size B
   - Compare throughput vs K to Theorem 4.2

2. **SLA Capacity Curves**
   - Sweep arrival rate λ
   - Find maximum λ for 1% or 5% SLA violations
   - Compare capacity gain to paper's 22% improvement

---

*Last Updated: 2025-11-20*
