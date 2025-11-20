"""
Scheduler implementations: multi-bin, dynamic batching, and baselines (paper-faithful).
"""

from typing import List, Callable, Deque
from collections import deque
import numpy as np

from .config import SchedulerConfig
from .workload import Request


class BatchStatistics:
    """Running statistics for dynamic batcher (Algorithm 1 support)."""
    
    def __init__(self):
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
    
    Implements the feedback control loop from Dynamic Batching paper:
    - Maintain adaptive interval [b_low, b_high]
    - If latency > D_SLA: shrink interval (reduce batch size)
    - If latency < D_SLA: expand interval (increase batch size)
    - Return b_SLA as midpoint of interval
    """
    
    def __init__(self, D_SLA: float, eps_D: float, B_min: int, B_max: int):
        self.D_SLA = D_SLA
        self.eps_D = eps_D
        self.B_min = B_min
        self.B_max = B_max
        
        # Adaptive search interval
        self.b_low = B_min
        self.b_high = B_max
        
        # Moving averages
        self.tau_avg = 0.0  # average latency
        self.b_avg = float(B_min)  # average batch size
        self.alpha = 0.2  # EMA smoothing factor
    
    def update(self, recent_latency: float, recent_batch_size: int) -> None:
        """Update controller state with recent observations."""
        if recent_batch_size <= 0:
            return
        
        self.tau_avg = self.alpha * recent_latency + (1 - self.alpha) * self.tau_avg
        self.b_avg = self.alpha * recent_batch_size + (1 - self.alpha) * self.b_avg
    
    def compute_b_SLA(self) -> int:
        """
        Compute SLA-constrained batch size using adaptive search.
        
        Returns:
            b_SLA: Batch size target from SLA controller
        """
        # Only adjust if we have some history
        if self.tau_avg == 0.0:
            return (self.b_low + self.b_high) // 2
        
        if self.tau_avg > self.D_SLA + self.eps_D:
            # Latency too high → decrease batch size
            self.b_high = min(self.b_high, int(self.b_avg))
            self.b_low = max(self.b_low, int(self.b_avg * 0.8))
        elif self.tau_avg < self.D_SLA - self.eps_D:
            # Latency too low → increase batch size
            self.b_low = max(self.b_low, int(self.b_avg))
            self.b_high = min(self.b_high + int(0.2 * self.b_avg), self.B_max)
        else:
            # Within SLA band → center around b_avg
            range_size = self.b_high - self.b_low
            margin = max(1, int(0.1 * range_size))
            self.b_low = max(self.B_min, int(self.b_avg) - margin)
            self.b_high = min(self.B_max, int(self.b_avg) + margin)
        
        # Ensure valid interval
        self.b_low = max(self.B_min, self.b_low)
        self.b_high = min(self.B_max, self.b_high)
        if self.b_low > self.b_high:
            self.b_low = self.b_high
        
        b_SLA = (self.b_low + self.b_high) // 2
        return max(self.B_min, min(self.B_max, b_SLA))


def compute_b_mem(stats: BatchStatistics, cfg: SchedulerConfig) -> int:
    """
    Compute memory-constrained batch size (Algorithm 1 from paper).
    
    Based on:
    η = (M_max - M_model) / kv_mem_per_token  (token capacity)
    μ = avg(prompt_len + output_len)          (avg tokens/req)
    b_mem = floor((η - L₀) / μ)               (max batch size)
    
    Args:
        stats: Running statistics
        cfg: Scheduler configuration
    
    Returns:
        Maximum batch size that fits in memory
    """
    # Token capacity from GPU memory
    eta = (cfg.M_MAX_GB - cfg.M_MODEL_GB) / cfg.KV_MEM_PER_TOKEN_GB
    
    # Average tokens per request
    avg_tokens = stats.avg_prompt_len + stats.avg_output_len
    
    # Safety buffer (10% of capacity)
    L0 = 0.1 * eta
    
    # Compute batch size
    b_mem = int((eta - L0) / avg_tokens)
    
    # Clamp to reasonable range
    return max(1, min(cfg.B_MAX, b_mem))


class MultiBinScheduler:
    """
    Global multi-bin scheduler that assigns requests to bins based on
    predicted output length, then serves GPUs from bins.
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
        
    def enqueue_request(self, req: Request) -> None:
        """
        Add a request to the appropriate bin based on predicted output length.
        
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
        for i, (min_len, max_len) in enumerate(self.cfg.BIN_BOUNDARIES):
            if min_len <= predicted_output_len < max_len:
                return i
        # If no match, put in last bin
        return self.cfg.K_BINS - 1
    
    def get_candidates_for_gpu(self, gpu_id: int, max_candidates: int) -> List[Request]:
        """
        Get candidate requests for a GPU to process.
        
        This implementation chooses one bin to serve based on the policy
        (round-robin or longest-queue) and returns up to max_candidates
        requests from that bin.
        
        Args:
            gpu_id: ID of the GPU requesting work
            max_candidates: Maximum number of candidates to return
        
        Returns:
            List of candidate requests (may be empty)
        """
        # Select bin based on policy
        if self.cfg.BIN_SELECTION_POLICY == "round_robin":
            bin_idx = self._select_bin_round_robin()
        elif self.cfg.BIN_SELECTION_POLICY == "longest_queue":
            bin_idx = self._select_bin_longest_queue()
        else:
            bin_idx = self._select_bin_round_robin()
        
        if bin_idx is None:
            return []
        
        # Pop up to max_candidates from the selected bin
        candidates = []
        for _ in range(min(max_candidates, len(self.bins[bin_idx]))):
            candidates.append(self.bins[bin_idx].popleft())
        
        return candidates
    
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
        
        # Running statistics for Algorithm 1
        self.stats = BatchStatistics()
        
        # SLA controller for Algorithm 2
        self.sla_controller = SLAController(
            D_SLA=cfg.D_SLA,
            eps_D=cfg.LATENCY_EPSILON,
            B_min=cfg.B_MIN,
            B_max=cfg.B_MAX,
        )
    
    def make_batch(
        self,
        now: float,
        candidates: List[Request],
    ) -> tuple[List[Request], float]:
        """
        Construct batch using b_target = min(b_mem, b_SLA) (paper algorithm).
        
        Args:
            now: Current simulation time
            candidates: List of candidate requests to batch from
        
        Returns:
            Tuple of (selected_requests, predicted_service_time)
        """
        if not candidates:
            return [], 0.0
        
        # Compute target batch size using paper algorithms
        b_mem = compute_b_mem(self.stats, self.cfg)
        b_SLA = self.sla_controller.compute_b_SLA()
        b_target = min(b_mem, b_SLA)
        
        # Sort by arrival time (FIFO)
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
    
    def update_after_batch(self, batch: List[Request], service_time: float) -> None:
        """
        Update statistics and controllers after batch completion.
        
        This implements the feedback loop from the papers.
        
        Args:
            batch: Completed batch
            service_time: Actual service time
        """
        if not batch:
            return
        
        # Update running statistics (for Algorithm 1)
        self.stats.update(batch)
        
        # Update SLA controller (for Algorithm 2)
        self.sla_controller.update(service_time, len(batch))
    
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
        Check if service time is within SLA bounds.
        
        Args:
            service_time: Estimated service time in seconds
        
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
        
        Args:
            gpu_id: ID of the GPU requesting work
            max_candidates: Maximum number of candidates to return
        
        Returns:
            List of candidate requests
        """
        candidates = []
        for _ in range(min(max_candidates, len(self.queue))):
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
    
    def make_batch(self, now: float, candidates: List[Request]) -> tuple[List[Request], float]:
        """
        Take exactly batch_size requests (or all if fewer available).
        
        Args:
            now: Current simulation time
            candidates: List of candidate requests
        
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
    
    def update_after_batch(self, batch: List[Request], service_time: float) -> None:
        """No-op for fixed batcher (no feedback loop)."""
        pass
