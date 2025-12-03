"""
Scheduler implementations: multi-bin, dynamic batching, and baselines (paper-faithful).

Key Insight from Multi-Bin Paper:
- Binning controls batch composition (WHO gets batched together)
- FIFO within bins maintains fairness
- Bins reduce E[max(t_j) | bin] by narrowing length distributions
- Throughput = B / E[batch_service_time] improves as k increases
"""

from typing import List, Callable, Deque
from collections import deque
import numpy as np

from .config import SchedulerConfig
from .workload import Request


class BatchCompositionTracker:
    """
    Track batch composition efficiency - the key Multi-Bin paper contribution.
    
    The Multi-Bin paper shows that binning improves throughput NOT by changing
    FIFO ordering, but by controlling which requests share a batch.
    
    Key Metric: Length variance within batches
    - Lower variance = better composition
    - Narrower ranges = lower E[max(t_j) | bin]
    - Better composition = higher throughput
    """
    
    def __init__(self, k_bins: int):
        self.k_bins = k_bins
        self.batches_per_bin = [0] * k_bins
        self.avg_length_variance_per_bin = [0.0] * k_bins
        self.avg_length_range_per_bin = [0.0] * k_bins
        self.total_batches = 0
    
    def record_batch(self, batch: List[Request], bin_idx: int) -> dict:
        """
        Record batch composition statistics for Multi-Bin analysis.
        
        Args:
            batch: The batch that was formed
            bin_idx: Which bin the batch came from
            
        Returns:
            Dictionary with composition metrics for this batch
        """
        if not batch or bin_idx < 0:
            return {}
        
        output_lengths = [r.output_len for r in batch]
        
        stats = {
            "bin_idx": bin_idx,
            "batch_size": len(batch),
            "min_length": min(output_lengths),
            "max_length": max(output_lengths),
            "length_range": max(output_lengths) - min(output_lengths),
            "length_variance": float(np.var(output_lengths)) if len(output_lengths) > 1 else 0.0,
            "length_std": float(np.std(output_lengths)) if len(output_lengths) > 1 else 0.0,
            "max_over_mean": max(output_lengths) / np.mean(output_lengths) if np.mean(output_lengths) > 0 else 1.0,
        }
        
        # Update running averages
        self.batches_per_bin[bin_idx] += 1
        self.total_batches += 1
        alpha = 0.2
        self.avg_length_variance_per_bin[bin_idx] = (
            alpha * stats["length_variance"] + 
            (1 - alpha) * self.avg_length_variance_per_bin[bin_idx]
        )
        self.avg_length_range_per_bin[bin_idx] = (
            alpha * stats["length_range"] +
            (1 - alpha) * self.avg_length_range_per_bin[bin_idx]
        )
        
        return stats
    
    def get_composition_summary(self) -> dict:
        """
        Get summary of batch composition efficiency across all bins.
        
        This is the key metric proving Multi-Bin's contribution:
        - Lower variance per bin = better composition
        - Tighter ranges = lower E[max(t_j) | bin]
        - Better composition = improved throughput
        
        Returns:
            Dictionary with composition efficiency metrics
        """
        return {
            "total_batches": self.total_batches,
            "batches_per_bin": self.batches_per_bin,
            "avg_variance_per_bin": self.avg_length_variance_per_bin,
            "avg_range_per_bin": self.avg_length_range_per_bin,
            "overall_avg_variance": float(np.mean(self.avg_length_variance_per_bin)),
            "overall_avg_range": float(np.mean(self.avg_length_range_per_bin)),
        }


class BatchStatistics:
    """Running statistics for dynamic batcher (Algorithm 1 support)."""
    
    def __init__(self, bin_idx: int = -1):
        self.bin_idx = bin_idx  # -1 for global, >= 0 for specific bin
        self.avg_prompt_len = 300.0  # Initial estimate
        self.avg_output_len = 200.0
        self.alpha = 0.2  # EMA smoothing factor
    
    def update(self, batch: List[Request]) -> None:
        """Update statistics from completed batch."""
        if not batch:
            return
        
        batch_avg_prompt = sum(r.prompt_len for r in batch) / len(batch)
        batch_avg_output = sum(r.output_len for r in batch) / len(batch)
        
        self.avg_prompt_len = self.alpha * batch_avg_prompt + (1 - self.alpha) * self.avg_prompt_len
        self.avg_output_len = self.alpha * batch_avg_output + (1 - self.alpha) * self.avg_output_len


class SLAController:
    """
    SLA-constrained batch size controller using adaptive search (Algorithm 2 from paper).
    
    Paper-Faithful Implementation:
    - D(b_t) is the mean Time Between Tokens (TBT) during the decode phase
    - Maintain adaptive interval [b_low, b_high] via feedback control
    - Parameters from paper:
      * D_SLA: target maximum decoding time per token
      * eps_D: tolerance around the SLA
      * alpha (α): step that controls expansion of interval
      * delta (δ): small corrective shrink adjustment
    
    Algorithm 2 Logic:
    - Case 1: τ_avg > D_SLA + eps_D (too slow) → shrink interval to left
    - Case 2: τ_avg < D_SLA - eps_D (too fast) → expand interval to right
    - Case 3: τ_avg ≈ D_SLA (within band) → center interval around b_avg
    - Return b_SLA = floor((b_low + b_high) / 2)
    
    NEW: Bin-specific controllers leverage narrower length distributions within bins.
    """
    
    def __init__(self, D_SLA: float, eps_D: float, B_min: int, B_max: int, bin_idx: int = -1,
                 alpha_step: int = 4, delta_step: int = 2):
        self.D_SLA = D_SLA
        self.eps_D = eps_D
        self.B_min = B_min
        self.B_max = B_max
        self.bin_idx = bin_idx  # -1 for global, >= 0 for specific bin
        
        # Paper hyperparameters for interval adjustment
        self.alpha_step = alpha_step  # α: controls expansion/contraction rate
        self.delta_step = delta_step  # δ: small corrective adjustment
        
        # Adaptive search interval - initialized to full range [B_min, B_max]
        self.b_low = B_min
        self.b_high = B_max
        
        # Moving averages for feedback
        self.tau_avg = 0.0  # average per-token decode latency (TBT)
        self.b_avg = float((B_min + B_max) // 2)  # Start at midpoint
        self.ema_alpha = 0.2  # EMA smoothing factor for averaging
        self.update_count = 0  # Track number of updates
        
        # Track N_decode (number of decode requests currently running)
        self.N_decode = 0
    
    def update(self, recent_tbt: float, recent_batch_size: int, N_decode: int = 0) -> None:
        """
        Update controller state with recent per-token latency (TBT).
        
        Args:
            recent_tbt: Recent per-token decode latency (TBT)
            recent_batch_size: Batch size that was used
            N_decode: Number of decode requests currently in system
        """
        if recent_batch_size <= 0:
            return
        
        self.tau_avg = self.ema_alpha * recent_tbt + (1 - self.ema_alpha) * self.tau_avg
        self.b_avg = self.ema_alpha * recent_batch_size + (1 - self.ema_alpha) * self.b_avg
        self.N_decode = N_decode
        self.update_count += 1
    
    def compute_b_SLA(self) -> int:
        """
        Compute SLA-constrained batch size using adaptive search (Algorithm 2).
        
        Paper-Faithful Implementation:
        
        Case 1: τ_avg > D_SLA + eps_D (latency too high)
            - Shrink window to the left (reduce batch size)
            - b_high_t ≈ max(b_avg, b_low_{t-1} + α)
            - b_low_t ≈ max(b_low_{t-1} - δ, B_min)
        
        Case 2: τ_avg < D_SLA - eps_D (latency too low, can increase)
            - Shift window to the right (increase batch size)
            - b_low_t ≈ min(b_avg, b_high_{t-1} - α)
            - b_high_t ≈ min(b_high_{t-1} + δ, B_max)
        
        Case 3: SLA approximately satisfied
            - Center window around current average
            - b_high_t = min(b_avg + α/2, B_max)
            - b_low_t = max(b_avg - α/2, B_min)
        
        Final: b_SLA = floor((b_low + b_high) / 2), clamped by N_decode and B_max
        
        Returns:
            b_SLA: Batch size target from SLA controller
        """
        # During warmup, use a reasonable starting batch size
        if self.tau_avg == 0.0 or self.update_count < 3:
            return (self.b_low + self.b_high) // 2
        
        α = self.alpha_step
        δ = self.delta_step
        
        if self.tau_avg > self.D_SLA + self.eps_D:
            # Case 1: Latency too high → shrink interval to left
            # We need smaller batch sizes
            new_b_high = max(int(self.b_avg), self.b_low + α)
            new_b_low = max(self.b_low - δ, self.B_min)
            self.b_high = min(self.b_high, new_b_high)  # Only shrink, never expand
            self.b_low = new_b_low
            
        elif self.tau_avg < self.D_SLA - self.eps_D:
            # Case 2: Latency comfortably below SLA → expand interval to right
            # We can increase batch size for more throughput
            new_b_low = min(int(self.b_avg), self.b_high - α)
            new_b_high = min(self.b_high + δ, self.B_max)
            self.b_low = max(self.b_low, new_b_low)  # Only shift right, never left
            self.b_high = new_b_high
            
        else:
            # Case 3: SLA approximately satisfied → center around current
            half_α = α // 2
            self.b_high = min(int(self.b_avg) + half_α, self.B_max)
            self.b_low = max(int(self.b_avg) - half_α, self.B_min)
        
        # Ensure valid interval
        self.b_low = max(self.B_min, self.b_low)
        self.b_high = min(self.B_max, self.b_high)
        if self.b_low > self.b_high:
            self.b_low = self.b_high
        
        # Compute midpoint as paper algorithm
        b_SLA = (self.b_low + self.b_high) // 2
        
        # Paper requirement: b_SLA >= N_decode (don't starve decode requests)
        # and b_SLA <= B_max
        b_SLA = max(b_SLA, self.N_decode) if self.N_decode > 0 else b_SLA
        b_SLA = min(b_SLA, self.B_max)
        
        return max(self.B_min, b_SLA)


def compute_b_mem(stats: BatchStatistics, cfg: SchedulerConfig, bin_idx: int = -1, 
                  N_decode: int = 0, N_prefill: int = 0) -> int:
    """
    Compute memory-constrained batch size (Algorithm 1 from paper).
    
    Paper-Faithful Implementation:
    
    Memory Model (KV cache):
        S = Σ(l_in,i + l_out,i) for all requests in batch
        E[S] = b_t × E[l_in + l_out]
        Var(S) = b_t × (Var(l_in) + Var(l_out))
    
    CLT approximation: S ~ N(μ_S, σ_S²)
    
    Probabilistic constraint: Pr(S > η) ≤ ε_M
    
    Safety buffer (precomputed):
        L₀ = η - (θ × σ_S + μ_S)  where θ = Φ⁻¹(1 - ε_M)
    
    Runtime formula:
        b_mem = floor((η - L₀) / E[l_in + l_out])
    
    Paper requirement: Only update b_mem if both N_prefill > 0 AND N_decode > 0
    Otherwise, keep previous batch size.
    
    Args:
        stats: Running statistics (bin-specific or global)
        cfg: Scheduler configuration
        bin_idx: Bin index for per-bin batch limits (-1 for global)
        N_decode: Number of decode requests (from last interval)
        N_prefill: Number of prefill requests (from last interval)
    
    Returns:
        Maximum batch size that fits in memory
    """
    # Token capacity η from GPU memory (how many tokens fit)
    # η = (M_max - M_model) / kv_mem_per_token
    eta = (cfg.M_MAX_GB - cfg.M_MODEL_GB) / cfg.KV_MEM_PER_TOKEN_GB
    
    # Average tokens per request: E[l_in + l_out]
    E_l_total = stats.avg_prompt_len + stats.avg_output_len
    
    # Safety: if no stats yet, use conservative estimate
    if E_l_total <= 0:
        E_l_total = 500.0  # Conservative default
    
    # Safety buffer L₀ (paper uses CLT-derived value)
    # Simplified: use 10% of capacity as safety buffer
    # This corresponds to ε_M ≈ 0.05 (5% OOM probability)
    L0 = 0.1 * eta
    
    # Paper Algorithm 1 logic:
    # Only recompute b_mem if both types of work are present
    # This prevents oscillation when workload is unbalanced
    # Note: In simulation we may not track N_prefill/N_decode explicitly,
    # so we default to always computing (caller can pass 0 to skip check)
    if N_decode > 0 and N_prefill > 0:
        # Paper formula: b_mem = floor((η - L₀) / E[l_in + l_out])
        b_raw = int((eta - L0) / E_l_total) if E_l_total > 0 else cfg.B_MAX
        
        # Paper requirement: b_mem >= N_decode (can't serve fewer than active decodes)
        b_mem = max(b_raw, N_decode)
    elif N_decode > 0 or N_prefill > 0:
        # Only one type of work present: compute but don't require N_decode minimum
        b_mem = int((eta - L0) / E_l_total) if E_l_total > 0 else cfg.B_MAX
    else:
        # No work tracked yet: compute anyway (simulation startup)
        b_mem = int((eta - L0) / E_l_total) if E_l_total > 0 else cfg.B_MAX
    
    # Overflow protection - ensure b_mem is a valid numeric type for isfinite check
    # This handles edge cases where int() on extreme float values produces non-standard types
    try:
        b_mem_float = float(b_mem)
        if not np.isfinite(b_mem_float) or b_mem < 0:
            b_mem = cfg.B_MAX
    except (ValueError, OverflowError, TypeError):
        b_mem = cfg.B_MAX
    
    # Apply per-bin batch limit if available
    # Longer bins get smaller B_MAX (memory + latency constrained)
    if bin_idx >= 0 and cfg.BIN_B_MAX is not None and bin_idx < len(cfg.BIN_B_MAX):
        bin_b_max = cfg.BIN_B_MAX[bin_idx]
        b_mem = min(b_mem, bin_b_max)
    
    # Clamp to [1, B_MAX]
    return max(1, min(cfg.B_MAX, b_mem))


class MultiBinScheduler:
    """
    Multi-bin scheduler implementing Guldogan et al. approach.
    
    Key Principle from Multi-Bin Paper:
    - Bins partition requests by predicted output length (matchmaking)
    - FIFO within each bin (fairness / queuing etiquette)
    - Batches formed from ONE bin only (batch composition control)
    - Bins reduce E[max(t_j) | bin] by narrowing length distributions
    - Throughput_k = B / E[T_batch,k] increases with k
    
    Why binning matters (example):
    - Without bins: batch [1 token, 100 tokens, 2 tokens, 3 tokens]
      → batch time = 100 (dominated by longest)
    - With bins: batch from short bin [1, 2, 3, 2]
      → batch time ≈ 3 (much better!)
    
    Binning controls WHO is married together in a batch.
    FIFO controls the order of service (fairness).
    Both are needed.
    """
    
    def __init__(self, cfg: SchedulerConfig):
        """
        Initialize multi-bin scheduler.
        
        Args:
            cfg: Scheduler configuration
        """
        self.cfg = cfg
        self.bins: List[Deque[Request]] = [deque() for _ in range(cfg.K_BINS)]
        self.current_bin_index = 0  # For round-robin
        
        # Track batch composition efficiency (Multi-Bin paper contribution)
        self.composition_tracker = BatchCompositionTracker(cfg.K_BINS)
        
    def enqueue_request(self, req: Request) -> None:
        """
        Add a request to the appropriate bin based on predicted output length.
        
        This is the "matchmaking" step from Multi-Bin paper:
        Decide which requests can be batched together based on predicted length.
        
        Args:
            req: Request to enqueue
        """
        bin_idx = self._select_bin(req.predicted_output_len)
        self.bins[bin_idx].append(req)
    
    def _select_bin(self, predicted_output_len: int) -> int:
        """
        Select which bin a request should go to based on predicted output length.
        
        Args:
            predicted_output_len: Predicted output length in tokens
        
        Returns:
            Bin index (0 to K_BINS-1)
        """
        # Fallback if boundaries not yet computed
        if self.cfg.BIN_BOUNDARIES is None:
            return 0  # Put everything in bin 0
        
        for i, (min_len, max_len) in enumerate(self.cfg.BIN_BOUNDARIES):
            if min_len <= predicted_output_len < max_len:
                return i
        # If no match, put in last bin
        return self.cfg.K_BINS - 1
    
    def get_candidates_for_gpu(self, gpu_id: int, max_candidates: int) -> tuple[List[Request], int]:
        """
        Get candidate requests for a GPU to process.
        
        CRITICAL (Multi-Bin Paper Requirement):
        - Returns candidates from ONE bin only (batch composition control)
        - This is how binning improves throughput: by controlling who shares a batch
        - FIFO within bin maintains fairness
        - Global bin selection (round-robin/longest-queue) maintains batch-level FIFO
        
        Args:
            gpu_id: ID of the GPU requesting work
            max_candidates: Maximum number of candidates to return
        
        Returns:
            Tuple of (candidate_requests, bin_idx)
            - candidate_requests: List of requests from one bin
            - bin_idx: Which bin they came from (-1 if empty)
        """
        # Select bin based on policy (FIFO at batch level)
        if self.cfg.BIN_SELECTION_POLICY == "round_robin":
            bin_idx = self._select_bin_round_robin()
        elif self.cfg.BIN_SELECTION_POLICY == "longest_queue":
            bin_idx = self._select_bin_longest_queue()
        else:
            bin_idx = self._select_bin_round_robin()
        
        if bin_idx is None:
            return [], -1
        
        # Pop up to max_candidates from the selected bin (FIFO within bin)
        candidates = []
        for _ in range(min(max_candidates, len(self.bins[bin_idx]))):
            candidates.append(self.bins[bin_idx].popleft())
        
        return candidates, bin_idx
    
    def _select_bin_round_robin(self) -> int | None:
        """
        Select the next non-empty bin using round-robin policy.
        
        Returns:
            Bin index, or None if all bins are empty
        """
        # Try each bin starting from current_bin_index
        for offset in range(self.cfg.K_BINS):
            idx = (self.current_bin_index + offset) % self.cfg.K_BINS
            if len(self.bins[idx]) > 0:
                self.current_bin_index = (idx + 1) % self.cfg.K_BINS
                return idx
        return None
    
    def _select_bin_longest_queue(self) -> int | None:
        """
        Select the bin with the most requests.
        
        Returns:
            Bin index, or None if all bins are empty
        """
        bin_lengths = [len(b) for b in self.bins]
        max_len = max(bin_lengths)
        if max_len == 0:
            return None
        return bin_lengths.index(max_len)
    
    def record_batch_composition(self, batch: List[Request], bin_idx: int) -> dict:
        """
        Record batch composition statistics for Multi-Bin analysis.
        
        This tracks the key contribution of the Multi-Bin paper:
        How binning improves batch composition and reduces E[max(t_j) | bin].
        
        Args:
            batch: The batch that was formed
            bin_idx: Which bin the batch came from
        
        Returns:
            Composition statistics for this batch
        """
        return self.composition_tracker.record_batch(batch, bin_idx)
    
    def get_composition_summary(self) -> dict:
        """
        Get overall batch composition efficiency summary.
        
        This provides the evidence for Multi-Bin's contribution:
        - Lower variance per bin = better composition
        - Tighter ranges = lower E[max(t_j) | bin]
        - Better composition = improved throughput
        
        Returns:
            Dictionary with composition efficiency metrics across all bins
        """
        return self.composition_tracker.get_composition_summary()
    
    def total_queued(self) -> int:
        """Return total number of requests across all bins."""
        return sum(len(b) for b in self.bins)


class DynamicBatcher:
    """
    Dynamic batcher using paper algorithms (Algorithm 1 + Algorithm 2).
    
    Implements:
    - Algorithm 1: Memory-constrained batch size (b_mem)
    - Algorithm 2: SLA-constrained batch size (b_SLA)
    - Final: b_target = min(b_mem, b_SLA)
    
    NEW: Bin-specific controllers leverage narrower length distributions:
    - Jobs in bin [10, 20] tokens have smaller E[max(t_j)] than [10, 200]
    - Each bin learns its own statistics and SLA constraints
    - Throughput_k = B / E[T_batch,k] improves as k increases (paper proof)
    """
    
    def __init__(
        self,
        cfg: SchedulerConfig,
        service_time_fn: Callable[[int, int], float]
    ):
        """
        Initialize dynamic batcher.
        
        Args:
            cfg: Scheduler configuration
            service_time_fn: Function to estimate service time given (batch_size, max_seq_len)
        """
        self.cfg = cfg
        self.service_time_fn = service_time_fn
        
        # Global statistics (for non-binned schedulers)
        self.global_stats = BatchStatistics(bin_idx=-1)
        self.global_sla_controller = SLAController(
            D_SLA=cfg.D_SLA,
            eps_D=cfg.LATENCY_EPSILON,
            B_min=cfg.B_MIN,
            B_max=cfg.B_MAX,
            bin_idx=-1,
        )
        
        # Per-bin statistics and controllers (for multi-bin scheduler)
        # Key insight: bins have narrower [L_min, L_max] ranges
        # → smaller E[max(t_j) | bin] → better throughput
        # → longer bins get smaller B_MAX (memory + latency constrained)
        self.bin_stats = [BatchStatistics(bin_idx=i) for i in range(cfg.K_BINS)]
        
        # Get per-bin batch limits (longer bins → smaller max batch)
        bin_b_max = getattr(cfg, 'BIN_B_MAX', None)
        if bin_b_max is None:
            bin_b_max = [cfg.B_MAX] * cfg.K_BINS
        
        self.bin_sla_controllers = [
            SLAController(
                D_SLA=cfg.D_SLA,
                eps_D=cfg.LATENCY_EPSILON,
                B_min=cfg.B_MIN,
                B_max=bin_b_max[i] if i < len(bin_b_max) else cfg.B_MAX,  # Per-bin max
                bin_idx=i,
            )
            for i in range(cfg.K_BINS)
        ]
    
    def make_batch(
        self,
        now: float,
        candidates: List[Request],
        bin_idx: int = -1,
        N_decode: int = 0,
        N_prefill: int = 0,
    ) -> tuple[List[Request], float]:
        """
        Construct batch using b_target = min(b_mem, b_SLA) (paper algorithm).
        
        NEW: Uses bin-specific statistics when bin_idx >= 0.
        
        Key insight from Multi-Bin paper:
        - Bin k has narrower length range [L_min_k, L_max_k]
        - E[max(t_j) | bin_k] < E[max(t_j) | all]
        - Can support larger batches with same SLA
        - Throughput_k = B / E[T_batch,k] increases with k
        
        Paper requirement (Algorithm 1):
        - b_mem only tightens when both N_decode > 0 AND N_prefill > 0
        - This prevents oscillation when workload is unbalanced
        - N_decode/N_prefill track concurrent decode/prefill requests
        
        Args:
            now: Current simulation time
            candidates: List of candidate requests to batch from
            bin_idx: Which bin these candidates are from (-1 for no bin)
            N_decode: Number of decode requests currently in system (from GPUState)
            N_prefill: Number of prefill requests currently in system (from GPUState)
        
        Returns:
            Tuple of (selected_requests, predicted_service_time)
        """
        if not candidates:
            return [], 0.0
        
        # Select bin-specific or global statistics
        if bin_idx >= 0 and bin_idx < len(self.bin_stats):
            stats = self.bin_stats[bin_idx]
            sla_controller = self.bin_sla_controllers[bin_idx]
        else:
            stats = self.global_stats
            sla_controller = self.global_sla_controller
        
        # Compute target batch size using paper algorithms
        # With bin-specific stats, b_mem and b_SLA are tuned to the bin's characteristics
        # b_mem respects per-bin batch limits for longer sequences
        # Pass N_decode/N_prefill so b_mem can enforce paper's concurrency-aware constraint
        b_mem = compute_b_mem(stats, self.cfg, bin_idx=bin_idx, 
                              N_decode=N_decode, N_prefill=N_prefill)
        b_SLA = sla_controller.compute_b_SLA()
        b_target = min(b_mem, b_SLA)
        
        # Sort by arrival time (FIFO within bin)
        sorted_candidates = sorted(candidates, key=lambda r: r.arrival_time)
        
        # Take first b_target requests
        batch = sorted_candidates[:b_target]
        
        # Double-check memory constraint (safety)
        while batch and not self._check_memory_constraint(batch):
            batch.pop()
        
        if not batch:
            return [], 0.0
        
        # Compute service time
        service_time = self._estimate_service_time(batch)
        
        return batch, service_time
    
    def update_after_batch(self, batch: List[Request], service_time: float, bin_idx: int = -1,
                           N_decode: int = 0, N_prefill: int = 0, decode_tbt: float = None) -> None:
        """
        Update statistics and controllers after batch completion (Feedback Loop).
        
        Paper-Faithful Implementation:
        This implements the feedback loop from both papers:
        - Algorithm 1 (Memory): Updates E[l_in + l_out] estimates
        - Algorithm 2 (SLA): Updates τ_avg and b_avg for interval adjustment
        
        v2 SLA Model (TTFT/TBT Separation):
        - Token SLA applies ONLY to decode TBT (β * h(b)), NOT TTFT
        - decode_tbt parameter provides the actual decode-only TBT for SLA feedback
        - This allows the controller to properly learn and maximize batch size
        
        NEW: Updates bin-specific controllers when bin_idx >= 0.
        This allows each bin to learn from its narrower length distribution.
        
        Args:
            batch: Completed batch
            service_time: Actual service time
            bin_idx: Which bin the batch came from (-1 for no bin)
            N_decode: Number of decode requests currently in system
            N_prefill: Number of prefill requests currently in system
            decode_tbt: Decode-only TBT (β * h(b)) for SLA feedback (v2 model)
        """
        if not batch:
            return
        
        # Update bin-specific or global statistics
        if bin_idx >= 0 and bin_idx < len(self.bin_stats):
            stats = self.bin_stats[bin_idx]
            sla_controller = self.bin_sla_controllers[bin_idx]
        else:
            stats = self.global_stats
            sla_controller = self.global_sla_controller
        
        # Update running statistics (for Algorithm 1)
        # This updates E[l_in], E[l_out] using EMA
        stats.update(batch)
        
        # Update SLA controller (for Algorithm 2)
        # v2 model: Use decode-only TBT (excludes TTFT)
        # This allows controller to learn correct batch sizes since decode TBT << D_SLA
        if decode_tbt is not None:
            # v2: Use provided decode-only TBT (β * h(b))
            recent_tbt = decode_tbt
        else:
            # Legacy fallback: service_time / max_output_len (includes TTFT)
            max_output_len = max(r.output_len for r in batch) if batch else 1
            recent_tbt = service_time / max_output_len if max_output_len > 0 else service_time

        # With bin-specific feedback, each bin learns its own latency characteristics
        # Pass N_decode so controller can ensure b_SLA >= N_decode (paper requirement)
        sla_controller.update(recent_tbt, len(batch), N_decode=N_decode)
    
    def _check_memory_constraint(self, batch: List[Request]) -> bool:
        """
        Check if a batch fits in GPU memory.
        
        Memory model:
        M_batch = M_model + KV_cache_per_token * total_tokens
        
        Args:
            batch: List of requests in the batch
        
        Returns:
            True if batch fits in memory, False otherwise
        """
        total_tokens = sum(req.prompt_len + req.output_len for req in batch)
        M_batch = self.cfg.M_MODEL_GB + self.cfg.KV_MEM_PER_TOKEN_GB * total_tokens
        
        available_memory = self.cfg.M_MAX_GB - self.cfg.MEMORY_MARGIN_GB
        return M_batch <= available_memory
    
    def _check_latency_constraint(self, service_time: float) -> bool:
        """
        Check if per-token decode latency (TBT) is within SLA bounds.
        
        Args:
            service_time: Estimated service time in seconds
            (callers should divide by max output length when using this constraint)
        
        Returns:
            True if within bounds, False otherwise
        """
        max_allowed = self.cfg.D_SLA + self.cfg.LATENCY_EPSILON
        return service_time <= max_allowed
    
    def _estimate_service_time(self, batch: List[Request]) -> float:
        """
        Estimate service time for a batch.
        
        Args:
            batch: List of requests in the batch
        
        Returns:
            Estimated service time in seconds
        """
        if not batch:
            return 0.0
        
        batch_size = len(batch)
        max_seq_len = max(req.prompt_len + req.output_len for req in batch)
        
        return self.service_time_fn(batch_size, max_seq_len)


class StaticFIFOScheduler:
    """
    Baseline: static FIFO scheduler with fixed batch size.
    """
    
    def __init__(self, cfg: SchedulerConfig, fixed_batch_size: int = 8):
        """
        Initialize static FIFO scheduler.
        
        Args:
            cfg: Scheduler configuration
            fixed_batch_size: Fixed batch size to use
        """
        self.cfg = cfg
        self.fixed_batch_size = fixed_batch_size
        self.queue: Deque[Request] = deque()
    
    def enqueue_request(self, req: Request) -> None:
        """Add request to FIFO queue."""
        self.queue.append(req)
    
    def get_candidates_for_gpu(self, gpu_id: int, max_candidates: int) -> List[Request]:
        """
        Get fixed number of requests from FIFO queue.
        
        Args:
            gpu_id: ID of the GPU requesting work
            max_candidates: Maximum number of candidates (ignored, uses fixed_batch_size)
        
        Returns:
            List of requests (up to fixed_batch_size)
        """
        candidates = []
        for _ in range(min(self.fixed_batch_size, len(self.queue))):
            candidates.append(self.queue.popleft())
        return candidates
    
    def total_queued(self) -> int:
        """Return total number of queued requests."""
        return len(self.queue)


class DynamicNoBinsScheduler:
    """
    Baseline: dynamic batching without bins (single FIFO queue).
    """
    
    def __init__(self, cfg: SchedulerConfig):
        """
        Initialize dynamic no-bins scheduler.
        
        Args:
            cfg: Scheduler configuration
        """
        self.cfg = cfg
        self.queue: Deque[Request] = deque()
    
    def enqueue_request(self, req: Request) -> None:
        """Add request to FIFO queue."""
        self.queue.append(req)
    
    def get_candidates_for_gpu(self, gpu_id: int, max_candidates: int) -> List[Request]:
        """
        Get candidates from FIFO queue.
        
        CRITICAL FIX: For dynamic batching, we need to fetch enough candidates
        for the batcher to choose from. The batcher will select b_target <= max_candidates,
        but we should give it a reasonable pool to work with.
        
        Args:
            gpu_id: ID of the GPU requesting work
            max_candidates: Maximum number of candidates to return
        
        Returns:
            List of candidate requests
        """
        # Fetch candidates up to max_candidates OR queue size, whichever is smaller
        # This ensures the dynamic batcher has options without depleting the queue
        num_to_fetch = min(max_candidates, len(self.queue))
        candidates = []
        for _ in range(num_to_fetch):
            candidates.append(self.queue.popleft())
        return candidates
    
    def total_queued(self) -> int:
        """Return total number of queued requests."""
        return len(self.queue)


class FixedBatchSizer:
    """
    Fixed batch size sizer for multi_bin_only mode (paper validation).
    
    This implements the pure Multi-Bin Batching paper approach:
    - Fixed batch size B
    - No dynamic resizing
    - Used for K_BINS sensitivity experiments
    """
    
    def __init__(self, cfg: SchedulerConfig, service_time_fn: Callable[[int, int], float]):
        """
        Initialize fixed batch sizer.
        
        Args:
            cfg: Scheduler configuration
            service_time_fn: Function to estimate service time
        """
        self.batch_size = cfg.B_FIXED
        self.service_time_fn = service_time_fn
    
    def make_batch(self, now: float, candidates: List[Request], 
                   bin_idx: int = -1, N_decode: int = 0, N_prefill: int = 0) -> tuple[List[Request], float]:
        """
        Take exactly batch_size requests (or all if fewer available).
        
        Args:
            now: Current simulation time
            candidates: List of candidate requests
            bin_idx: Bin index (ignored for fixed batch sizing, kept for interface compatibility)
            N_decode: Number of decode requests (ignored, kept for interface compatibility)
            N_prefill: Number of prefill requests (ignored, kept for interface compatibility)
        
        Returns:
            Tuple of (batch, service_time)
        """
        if not candidates:
            return [], 0.0
        
        # Sort by arrival time (FIFO)
        sorted_candidates = sorted(candidates, key=lambda r: r.arrival_time)
        
        # Take exactly batch_size requests (or all available)
        batch = sorted_candidates[:self.batch_size]
        
        if not batch:
            return [], 0.0
        
        # Service time = f(batch_size, max_seq_len)
        max_seq_len = max(r.prompt_len + r.output_len for r in batch)
        service_time = self.service_time_fn(len(batch), max_seq_len)
        
        return batch, service_time
    
    def update_after_batch(self, batch: List[Request], service_time: float, bin_idx: int = -1,
                           N_decode: int = 0, N_prefill: int = 0, decode_tbt: float = None) -> None:
        """No-op for fixed batcher (no feedback loop)."""
        pass
