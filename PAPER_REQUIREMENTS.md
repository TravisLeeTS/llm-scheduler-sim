# Paper Requirements and Design Specification

This document specifies the **exact requirements** from the Multi-Bin Batching and Dynamic Batching papers to ensure our simulator is **faithful to the theory** rather than "paper-inspired fanfic".

---

## 1. Multi-Bin Batching Requirements

**Paper**: Multi-Bin Batching for LLM Inference Throughput Optimization

### 1.1 System Model Assumptions

For the multi-bin layer, we must assume:

* **Single-server queue** (we can simulate multi-GPU later, but theory is single server)
* **Arrival process**: Poisson with rate λ
  * Baseline mode: Pure Poisson arrivals (exponential inter-arrival times)
  * Extended mode: BurstGPT-style ON/OFF for realism
* **Service time per request**:
  * i.i.d. random variable in `[l_min, l_max]`
  * Theory uses uniform distribution `U(l_min, l_max)` for analysis
  * Simulator can generalize to other distributions (gamma, power-law, etc.)

### 1.2 Batch Service Time Model

**CRITICAL**: Batch service time is the **maximum** of service times of its requests.

```
service_time(batch) = max(service_time(r_i) for r_i in batch)
```

This reflects reality: in LLM inference, the batch completes when the **longest sequence** finishes decoding.

The server (GPU) is busy from batch start until the slowest request finishes.

### 1.3 Multi-Bin Algorithm (Paper-Faithful Version)

**Core algorithm**:

1. **Choose K_BINS = k** (number of bins, e.g., 1, 2, 4, 8)

2. **Define bin boundaries**:
   * For **equal-mass bins** (required for optimality proofs):
     * Each bin should have approximately **equal probability mass** w.r.t. predicted output length
     * With `U(l_min, l_max)`: use equally-spaced boundaries
       ```
       l_i = l_min + (i/k) * (l_max - l_min)  for i = 0, 1, ..., k
       ```
     * In practice: use **empirical quantiles** of predicted_output_len distribution
       ```python
       boundaries = np.quantile(predicted_lengths, [i/k for i in range(k+1)])
       ```

3. **Request assignment**:
   * For each incoming request:
     * Predict output length (or service time): `l̂`
     * Assign to bin `i` where `l_{i-1} ≤ l̂ < l_i`
     * Put request in that bin's **FIFO queue**

4. **Batch formation** (FIXED BATCH SIZE):
   * Each bin is a FIFO queue
   * When bin has `≥ B` requests, form a batch of **exactly B** requests
   * Push batch to **central batch queue** (FIFO by batch formation time)
   * **B is FIXED** in pure multi-bin experiments (e.g., 8, 32, 128)

5. **Server (GPU) processing**:
   * Server pops next batch from central batch queue (FIFO order)
   * Processes batch: `service_time = max(request_times in batch)`
   * Schedules GPU_FREE event when batch completes

### 1.4 Theoretical Results to Reproduce

* **Throughput increases with k**:
  * k=1 → normal batching (single FIFO queue)
  * k→∞ → throughput approaches theoretical maximum
  * `c_max = B / E[service_time]` (under uniform distribution)
  
* **Experiment**: Vary k ∈ {1, 2, 4, 8} with fixed B, Poisson arrivals, measure throughput
  * Should see monotonic increase in throughput as k increases

---

## 2. Dynamic Batching Requirements

**Paper**: Memory-Aware and SLA-Constrained Dynamic Batching

### 2.1 System Model

* Batch size `b_t` is a **control variable** (decision at each scheduling point)
* Two constraints:
  1. **Memory constraint**: prevent GPU OOM
  2. **SLA constraint**: meet latency targets
* **Objective**: Maximize throughput `Φ(t) ≈ b_t / τ_step(b_t)`

### 2.2 Memory-Constrained Batch Size (Algorithm 1)

**Goal**: Compute maximum batch size that fits in GPU memory with high probability.

**Model**:

* Let `l_in,i`, `l_out,i` = input and output tokens for request i
* Total tokens in batch: `S = Σ_i (l_in,i + l_out,i)`
* Memory limit: `η` = max tokens capacity from GPU memory
  ```
  η = (M_max - M_model) / kv_mem_per_token
  ```
* Under **CLT assumptions**, S is approximately Gaussian → probabilistic bound

**Algorithm** (simplified for simulation):

```python
def compute_b_mem(stats, memory_config):
    """
    Compute memory-limited batch size.
    
    Args:
        stats: Running statistics (μ_in, μ_out, σ_in, σ_out)
        memory_config: (M_max, M_model, kv_mem_per_token, eps_M)
    
    Returns:
        b_mem: Maximum batch size that fits in memory
    """
    # Estimate token capacity
    eta = (M_max - M_model) / kv_mem_per_token
    
    # Running averages
    mu_in = stats['avg_prompt_len']
    mu_out = stats['avg_output_len']
    
    # Safety buffer (L0) based on variance and eps_M
    # For simulation: use simple heuristic (5-10% of eta)
    L0 = 0.1 * eta
    
    # Compute batch size
    avg_tokens_per_request = mu_in + mu_out
    b_mem = floor((eta - L0) / avg_tokens_per_request)
    
    # Clamp to [B_min, B_max]
    return clamp(b_mem, B_min, B_max)
```

**Implementation notes**:

* Maintain **running averages** of prompt_len and output_len from recent batches
* Update statistics after each batch completes
* Don't need full CLT math in simulator; simplified formula captures essence

### 2.3 SLA-Constrained Batch Size (Algorithm 2)

**Goal**: Dynamically adjust batch size to meet SLA latency targets using feedback control.

**Model**:

* `τ` = recent average decode latency (e.g., ms/token or total batch latency)
* `b̄` = recent average batch size actually used
* `D_SLA` = target maximum decoding time (SLA deadline)
* `ε_D` = tolerance band around D_SLA
* `[B_min, B_max]` = allowed batch size range

**Algorithm** (feedback control loop):

```python
class SLAController:
    def __init__(self, D_SLA, eps_D, B_min, B_max):
        self.D_SLA = D_SLA
        self.eps_D = eps_D
        self.B_min = B_min
        self.B_max = B_max
        
        # Dynamic search interval
        self.b_low = B_min
        self.b_high = B_max
        
        # Moving averages
        self.tau_avg = 0.0  # average latency
        self.b_avg = 0.0    # average batch size
    
    def update(self, recent_latency, recent_batch_size):
        """Update controller state with recent observations."""
        # Update moving averages (exponential moving average)
        alpha = 0.2  # smoothing factor
        self.tau_avg = alpha * recent_latency + (1 - alpha) * self.tau_avg
        self.b_avg = alpha * recent_batch_size + (1 - alpha) * self.b_avg
    
    def compute_b_SLA(self):
        """
        Compute SLA-constrained batch size using adaptive search.
        
        Returns:
            b_SLA: Batch size target from SLA controller
        """
        if self.tau_avg > self.D_SLA + self.eps_D:
            # Latency too high → decrease batch size
            # Shrink upper bound, move lower bound towards b_avg
            self.b_high = min(self.b_high, int(self.b_avg))
            self.b_low = max(self.b_low, int(self.b_avg * 0.8))
        
        elif self.tau_avg < self.D_SLA - self.eps_D:
            # Latency too low (conservative) → increase batch size
            # Expand search interval upwards
            self.b_low = max(self.b_low, int(self.b_avg))
            self.b_high = min(self.b_high + int(0.2 * self.b_avg), self.B_max)
        
        else:
            # Within SLA band → center interval around b_avg
            range_size = self.b_high - self.b_low
            margin = int(0.1 * range_size)
            self.b_low = max(self.B_min, int(self.b_avg) - margin)
            self.b_high = min(self.B_max, int(self.b_avg) + margin)
        
        # Ensure valid interval
        self.b_low = max(self.B_min, self.b_low)
        self.b_high = min(self.B_max, self.b_high)
        if self.b_low > self.b_high:
            self.b_low = self.b_high
        
        # Return midpoint of search interval
        b_SLA = (self.b_low + self.b_high) // 2
        return b_SLA
```

### 2.4 Combined Dynamic Batching

**Final batch size**:

```python
b_mem = compute_b_mem(stats, memory_config)
b_SLA = sla_controller.compute_b_SLA()

b_target = min(b_mem, b_SLA)
```

Then from candidate pool:

```python
batch = candidates[:b_target]  # or until candidates exhausted
```

**Update flow**:

1. GPU becomes free → select candidates from scheduler
2. Compute `b_target = min(b_mem, b_SLA)`
3. Form batch: take first `b_target` requests from candidates
4. Process batch, record latency and size
5. Update SLA controller: `sla_controller.update(latency, batch_size)`
6. Update statistics: running averages of prompt_len, output_len

---

## 3. Combined Policy: Multi-Bin + Dynamic Batching

### 3.1 System Architecture

```
Workload Generator
    ↓
Multi-Bin Layer (Global)
    ├── Bin 0: short requests     (FIFO queue)
    ├── Bin 1: medium requests    (FIFO queue)
    ├── Bin 2: long requests      (FIFO queue)
    └── Bin 3: very long requests (FIFO queue)
    ↓
Bin Selection (round-robin or longest-queue)
    ↓
Candidate Pool (up to MAX_CANDIDATES from selected bin)
    ↓
Dynamic Batcher
    ├── Compute b_mem (memory constraint)
    ├── Compute b_SLA (SLA constraint)
    └── b_target = min(b_mem, b_SLA)
    ↓
Final Batch (up to b_target requests)
    ↓
GPU Processing (service_time = f(batch_size, max_seq_len))
    ↓
Metrics & Feedback
```

### 3.2 Operational Flow

**Request arrival**:
1. Predict output length: `l̂`
2. Assign to bin `i` where `l_{i-1} ≤ l̂ < l_i`
3. Enqueue in bin's FIFO queue

**GPU becomes free**:
1. **Multi-Bin layer**: select bin (round-robin or longest-queue)
2. Pull up to `MAX_CANDIDATES` from selected bin
3. **Dynamic Batcher**:
   * Compute `b_mem` from memory constraint
   * Compute `b_SLA` from SLA controller
   * Set `b_target = min(b_mem, b_SLA)`
   * Build batch: take first `b_target` requests from candidates
   * Return unused candidates to bin
4. **GPU**: process batch
   * `service_time = f(batch_size, max_seq_len)`
   * Schedule GPU_FREE event
5. **On completion**:
   * Record metrics (latency, throughput, etc.)
   * Update SLA controller with `(latency, batch_size)`
   * Update running statistics (avg_prompt_len, avg_output_len)

---

## 4. Experiment Modes

To properly validate against both papers, implement **three experiment modes**:

### 4.1 Mode: "multi_bin_only"

**Purpose**: Reproduce Multi-Bin paper results

**Configuration**:
* `K_BINS ∈ {1, 2, 4, 8}` (vary for experiments)
* `B_FIXED` (e.g., 8, 32, 128) → **fixed batch size**, no dynamic batching
* `ARRIVAL_PROFILE = "poisson"` with rate λ
* Service time distribution: `U(l_min, l_max)` or gamma

**Behavior**:
* Multi-Bin layer forms batches of **exactly B** (when possible)
* No dynamic resizing
* Central batch queue: FIFO by batch formation time
* GPU pops batches from central queue

**Expected results**:
* Throughput increases as K_BINS increases
* k=1 (single queue) < k=2 < k=4 < k=8
* Asymptotically approaches theoretical maximum

### 4.2 Mode: "dynamic_only"

**Purpose**: Reproduce Dynamic Batching paper results

**Configuration**:
* `K_BINS = 1` (single global FIFO queue, no multi-bin)
* Dynamic batching **enabled**
* `b_target = min(b_mem, b_SLA)`
* Compare against baseline: **static fixed B**

**Behavior**:
* Single FIFO queue for all requests
* Dynamic batcher computes `b_mem` and `b_SLA` each scheduling decision
* Batch size varies based on constraints

**Expected results**:
* Dynamic batching achieves higher throughput than static B
* Respects memory constraints (no OOM)
* Meets SLA targets (lower violation rate)
* Batch size adapts to load conditions

### 4.3 Mode: "multi_bin_dynamic"

**Purpose**: Combined approach (our main contribution)

**Configuration**:
* `K_BINS > 1` (e.g., 4)
* Dynamic batching **enabled**
* Equal-mass bin boundaries

**Behavior**:
* Multi-Bin layer provides coarse grouping by predicted length
* Dynamic batcher provides fine-grained control within bin
* `b_target = min(b_mem, b_SLA)` acts as **max candidate pool size** for multi-bin

**Expected results**:
* Best of both worlds: multi-bin efficiency + dynamic adaptability
* Higher throughput than either approach alone
* Lower latency variance
* Better resource utilization

---

## 5. Service Time Model

### 5.1 Batch Service Time

**Paper requirement**: Batch service time dominated by **longest sequence**.

```python
def compute_batch_service_time(batch, model_params):
    """
    Compute service time for a batch.
    
    Batch completes when the longest request finishes.
    """
    max_seq_len = max(req.prompt_len + req.output_len for req in batch)
    batch_size = len(batch)
    
    # Linear model (from Multi-Bin paper, calibrated on Phi-3.5)
    # service_time ≈ a0 + a1 * max_seq_len * h(batch_size)
    
    # Simplified for simulation:
    base = model_params['base_latency']
    alpha = model_params['seq_len_coeff']
    beta = model_params['batch_penalty']
    
    # Batch penalty increases sublinearly with batch size
    batch_factor = 1 + beta * (batch_size - 1) / batch_size
    
    service_time = base + alpha * max_seq_len * batch_factor
    
    return service_time
```

### 5.2 Calibration with Qwen3-0.6B via vLLM

**Future enhancement** (not required for initial simulator):

1. Run micro-benchmarks with vLLM + Qwen3-0.6B
2. Vary `(batch_size, max_seq_len)` systematically
3. Measure actual latency
4. Fit linear model: `latency ~ a0 + a1 * max_seq_len + a2 * batch_size`
5. Use fitted coefficients in simulator

This gives **realistic absolute numbers**, but relative comparisons are valid even with synthetic formula.

---

## 6. Workload Generation

### 6.1 Poisson Mode (Paper-Faithful)

```python
if ARRIVAL_PROFILE == "poisson":
    # Pure Poisson arrivals
    inter_arrivals = np.random.exponential(scale=1.0/lambda_rate, size=num_requests)
    arrival_times = np.cumsum(inter_arrivals)
```

### 6.2 BurstGPT Mode (Realistic)

Keep existing ON/OFF process for production-like experiments:

```python
if ARRIVAL_PROFILE == "burstgpt_like":
    arrival_times = generate_burstgpt_arrivals(...)
```

### 6.3 Service Time Distribution

**For multi_bin_only experiments** (theory validation):

```python
if mode == "multi_bin_only":
    # Use uniform distribution as in paper
    service_times = np.random.uniform(l_min, l_max, size=num_requests)
```

**For dynamic experiments** (realistic):

```python
else:
    # Use gamma or power-law distribution
    service_times = np.random.gamma(shape, scale, size=num_requests)
```

---

## 7. Equal-Mass Bin Boundaries

**Critical for Multi-Bin optimality**: bins should have **equal probability mass**.

### Implementation:

```python
def compute_equal_mass_boundaries(predicted_lengths, K_BINS):
    """
    Compute bin boundaries such that each bin has ~equal number of requests.
    
    Args:
        predicted_lengths: Array of all predicted output lengths
        K_BINS: Number of bins
    
    Returns:
        List of (min, max) tuples for each bin
    """
    # Compute quantiles
    quantiles = np.linspace(0, 1, K_BINS + 1)
    boundaries_points = np.quantile(predicted_lengths, quantiles)
    
    # Convert to (min, max) pairs
    bin_boundaries = []
    for i in range(K_BINS):
        min_len = boundaries_points[i]
        max_len = boundaries_points[i + 1]
        bin_boundaries.append((min_len, max_len))
    
    # Adjust last bin to include everything
    bin_boundaries[-1] = (bin_boundaries[-1][0], float('inf'))
    
    return bin_boundaries
```

**Usage**:

```python
# Pre-sample predicted lengths from workload
predicted_lengths = [predict_output_len(p) for p in prompt_lengths]

# Compute equal-mass boundaries
bin_boundaries = compute_equal_mass_boundaries(predicted_lengths, K_BINS)

# Use these boundaries in scheduler config
cfg.BIN_BOUNDARIES = bin_boundaries
```

---

## 8. Metrics to Track

### 8.1 Core Metrics

* **Throughput**: requests/second (overall and per-bin)
* **Latency**: P50, P95, P99 latency distributions
* **SLA violation rate**: percentage of requests exceeding D_SLA
* **GPU utilization**: busy_time / total_time
* **Batch size distribution**: mean, std, histogram

### 8.2 Multi-Bin Specific

* **Throughput vs K_BINS**: plot showing throughput increasing with k
* **Theoretical maximum**: compare against `c_max = B / E[service_time]`
* **Per-bin statistics**: throughput, latency, queue length for each bin

### 8.3 Dynamic Batching Specific

* **Memory utilization**: actual vs limit
* **SLA adherence**: latency tracking over time
* **Batch size adaptation**: plot of batch size over time showing controller response
* **Controller convergence**: time to reach steady state

---

## 9. Summary Checklist

To ensure simulator is **paper-faithful**:

- [ ] **Multi-Bin layer**:
  - [ ] Equal-mass bin boundaries (empirical quantiles)
  - [ ] Fixed batch size B for "multi_bin_only" mode
  - [ ] Central batch queue (FIFO by batch formation time)
  - [ ] Batch service time = max(request_times)
  - [ ] Throughput vs K_BINS experiments

- [ ] **Dynamic Batching layer**:
  - [ ] Compute b_mem from memory constraint
  - [ ] Implement SLA feedback controller for b_SLA
  - [ ] Final batch size: b_target = min(b_mem, b_SLA)
  - [ ] Update running statistics and controller state after each batch

- [ ] **Workload**:
  - [ ] Poisson arrival mode (for theory validation)
  - [ ] BurstGPT ON/OFF mode (for realism)
  - [ ] Configurable service time distributions

- [ ] **Experiment modes**:
  - [ ] "multi_bin_only": k ∈ {1,2,4,8}, fixed B, Poisson arrivals
  - [ ] "dynamic_only": k=1, dynamic batching vs static B
  - [ ] "multi_bin_dynamic": k>1, dynamic batching enabled

- [ ] **Service time model**:
  - [ ] Batch time = f(max_seq_len, batch_size)
  - [ ] Calibration option with vLLM (future)

This document serves as the **ground truth specification** for Copilot to implement the simulator correctly.
