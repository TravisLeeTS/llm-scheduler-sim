"""
Discrete-event simulator for LLM inference scheduling.
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Any
from enum import Enum

from .config import SchedulerConfig
from .workload import Request
from .schedulers import (
    MultiBinScheduler,
    DynamicBatcher,
    StaticFIFOScheduler,
    DynamicNoBinsScheduler,
    FixedBatchSizer,
)
from .model_calibration import get_service_time_function, LatencyModel


class EventType(Enum):
    """Types of simulation events."""
    ARRIVAL = "arrival"
    GPU_FREE = "gpu_free"


@dataclass(order=True)
class Event:
    """Simulation event with priority queue ordering."""
    time: float
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False)


@dataclass
class GPUState:
    """State of a single GPU."""
    gpu_id: int
    busy: bool = False
    free_at: float = 0.0
    current_batch: List[Request] = field(default_factory=list)
    
    # Statistics
    total_batches: int = 0
    total_requests: int = 0
    total_busy_time: float = 0.0


class Simulator:
    """
    Discrete-event simulator for LLM inference scheduling.
    
    Supports arbitrary number of GPUs and different scheduler types.
    """
    
    def __init__(
        self,
        cfg: SchedulerConfig,
        requests: List[Request],
        scheduler_type: str = "multi_bin_dynamic",
    ):
        """
        Initialize simulator.
        
        Args:
            cfg: Scheduler configuration
            requests: List of requests to process
            scheduler_type: Type of scheduler to use:
                - "static_fifo": Fixed batch size FIFO
                - "dynamic_no_bins": Dynamic batching without bins
                - "multi_bin_dynamic": Multi-bin with dynamic batching
                - "multi_bin_only": Multi-bin with fixed batch size (paper validation)
        """
        self.cfg = cfg
        self.requests = sorted(requests, key=lambda r: r.arrival_time)
        self.scheduler_type = scheduler_type
        
        # Simulation state
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        
        # GPU states (OPTIMIZED: pre-allocate list for faster access)
        self.gpus = [GPUState(gpu_id=i) for i in range(cfg.NUM_GPUS)]
        self._idle_gpus = set(range(cfg.NUM_GPUS))  # Track idle GPUs for O(1) lookup
        
        # Initialize service time function
        if cfg.USE_REAL_CALIBRATION and cfg.CALIBRATION_CSV_PATH:
            print(f"Using real GPU calibration from: {cfg.CALIBRATION_CSV_PATH}")
            latency_model = LatencyModel(cfg.CALIBRATION_CSV_PATH)
            self.service_time_fn = latency_model.predict
            print(f"LatencyModel info: {latency_model.get_info()}")
        else:
            self.service_time_fn = get_service_time_function()
        
        # Initialize scheduler and batcher based on scheduler_type
        # Each scheduler mode should have DISTINCT behavior
        
        if scheduler_type == "static_fifo":
            # Mode 1: Static FIFO with FIXED batch size (no SLA controller, no binning)
            print(f"[Scheduler Init] static_fifo: B_FIXED=8, NO dynamic batching, NO bins")
            self.scheduler = StaticFIFOScheduler(cfg, fixed_batch_size=8)
            self.batcher = None  # No dynamic batching
            
        elif scheduler_type in ["dynamic_no_bins", "dynamic_only"]:
            # Mode 2: Single queue + Dynamic batching (SLA controller + b_mem)
            print(f"[Scheduler Init] dynamic_no_bins: Single FIFO queue + DynamicBatcher (SLA controller)")
            self.scheduler = DynamicNoBinsScheduler(cfg)
            self.batcher = DynamicBatcher(cfg, self.service_time_fn)
            
        elif scheduler_type == "multi_bin_dynamic":
            # Mode 3: Multi-Bin queues + Dynamic batching (our contribution)
            print(f"[Scheduler Init] multi_bin_dynamic: K_BINS={cfg.K_BINS} + DynamicBatcher (SLA controller)")
            self.scheduler = MultiBinScheduler(cfg)
            self.batcher = DynamicBatcher(cfg, self.service_time_fn)
            
        elif scheduler_type == "multi_bin_only":
            # Mode 4: Multi-Bin with FIXED batch size (for paper validation)
            print(f"[Scheduler Init] multi_bin_only: K_BINS={cfg.K_BINS}, B_FIXED={cfg.B_FIXED}")
            self.scheduler = MultiBinScheduler(cfg)
            self.batcher = FixedBatchSizer(cfg, self.service_time_fn)
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        # Statistics
        self.completed_requests: List[Request] = []
    
    def run(self) -> List[Request]:
        """
        Run the simulation until all requests are processed.
        
        Returns:
            List of completed requests with timing information
        """
        # Schedule all arrival events
        for req in self.requests:
            self._schedule_event(Event(
                time=req.arrival_time,
                event_type=EventType.ARRIVAL,
                payload=req,
            ))
        
        # Process events
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            if event.event_type == EventType.ARRIVAL:
                self._handle_arrival(event.payload)
            elif event.event_type == EventType.GPU_FREE:
                self._handle_gpu_free(event.payload)
        
        return self.completed_requests
    
    def _schedule_event(self, event: Event) -> None:
        """Add an event to the priority queue."""
        heapq.heappush(self.event_queue, event)
    
    def _handle_arrival(self, req: Request) -> None:
        """
        Handle a request arrival event (OPTIMIZED).
        
        Args:
            req: The arriving request
        """
        # Enqueue request in scheduler
        self.scheduler.enqueue_request(req)
        
        # OPTIMIZATION: Only check idle GPUs (O(idle) instead of O(total))
        for gpu_id in list(self._idle_gpus):
            self._try_schedule_gpu(self.gpus[gpu_id])
    
    def _handle_gpu_free(self, gpu_id: int) -> None:
        """
        Handle a GPU becoming free.
        
        Args:
            gpu_id: ID of the GPU that became free
        """
        gpu = self.gpus[gpu_id]
        
        # Calculate service time for feedback
        if gpu.current_batch:
            service_time = self.current_time - min(r.start_service_time for r in gpu.current_batch)
            
            # Determine which bin this batch came from (for bin-specific feedback)
            # All requests in a batch come from the same bin (Multi-Bin invariant)
            if gpu.current_batch and hasattr(gpu.current_batch[0], 'predicted_output_len'):
                # Determine bin from first request
                req = gpu.current_batch[0]
                bin_idx = self._get_bin_idx(req.predicted_output_len)
            else:
                bin_idx = -1
        else:
            service_time = 0.0
            bin_idx = -1
        
        # Update batcher statistics (feedback loop for Algorithm 1 & 2)
        # Only multi_bin uses bin-specific learning, dynamic_no_bins uses global
        if self.batcher and hasattr(self.batcher, 'update_after_batch'):
            if self.scheduler_type == "dynamic_no_bins":
                # Force global statistics for dynamic_no_bins
                self.batcher.update_after_batch(gpu.current_batch, service_time, bin_idx=-1)
            else:
                # multi_bin_dynamic uses bin-specific learning
                self.batcher.update_after_batch(gpu.current_batch, service_time, bin_idx=bin_idx)
        
        # Mark completed requests
        for req in gpu.current_batch:
            req.completion_time = self.current_time
            req.assigned_gpu = gpu_id
            self.completed_requests.append(req)
        
        # Update GPU state
        gpu.busy = False
        gpu.current_batch = []
        self._idle_gpus.add(gpu.gpu_id)  # Mark as idle
        
        # Try to schedule new work on this GPU
        self._try_schedule_gpu(gpu)
    
    def _try_schedule_gpu(self, gpu: GPUState) -> None:
        """
        Try to schedule work on an idle GPU.
        
        Args:
            gpu: The GPU to schedule work on
        """
        if gpu.busy:
            return
        
        # Get candidates from scheduler
        # For MultiBinScheduler, this returns (candidates, bin_idx)
        # For other schedulers, returns just candidates (need to handle both)
        result = self.scheduler.get_candidates_for_gpu(
            gpu.gpu_id,
            self.cfg.MAX_CANDIDATES
        )
        
        # Handle both tuple (MultiBin) and list (other schedulers) returns
        if isinstance(result, tuple):
            candidates, bin_idx = result
        else:
            candidates = result
            bin_idx = -1
        
        if not candidates:
            return  # No work available
        
        # Create batch based on scheduler type
        if self.batcher is None:
            # Static FIFO: use all candidates (fixed batch size already enforced by scheduler)
            batch = candidates
            # Estimate service time
            if batch:
                max_seq_len = max(r.prompt_len + r.output_len for r in batch)
                service_time = self.service_time_fn(len(batch), max_seq_len)
            else:
                service_time = 0.0
        else:
            # Dynamic batching (dynamic_no_bins or multi_bin_dynamic)
            # Only multi_bin_dynamic uses bin-specific statistics (bin_idx >= 0)
            # dynamic_no_bins always uses global statistics (bin_idx = -1)
            if self.scheduler_type == "dynamic_no_bins":
                # Force global statistics for dynamic_no_bins (no bin information)
                batch, service_time = self.batcher.make_batch(
                    self.current_time,
                    candidates,
                    bin_idx=-1  # Always use global statistics
                )
            else:
                # multi_bin_dynamic uses bin-specific statistics
                batch, service_time = self.batcher.make_batch(
                    self.current_time,
                    candidates,
                    bin_idx=bin_idx  # Enables bin-specific statistics and SLA control
                )
            
            
            # Put unused candidates back at the FRONT of the queue to maintain FIFO ordering
            # CRITICAL FIX: Re-enqueuing at the back causes queue explosion for dynamic batching
            unused = [c for c in candidates if c not in batch]
            if unused:
                # Put back in reverse order so FIFO is preserved
                if isinstance(self.scheduler, DynamicNoBinsScheduler):
                    # For single queue, put unused candidates back at front
                    for req in reversed(unused):
                        self.scheduler.queue.appendleft(req)
                elif isinstance(self.scheduler, MultiBinScheduler):
                    # For multi-bin, put back in appropriate bins at front
                    for req in reversed(unused):
                        bin_idx = self._get_bin_idx(req.predicted_output_len)
                        self.scheduler.bins[bin_idx].appendleft(req)
                else:
                    # Fallback: re-enqueue normally (shouldn't happen for dynamic schedulers)
                    for req in unused:
                        self.scheduler.enqueue_request(req)
        
        if not batch:
            return  # No valid batch could be formed
        
        # Record batch composition (Multi-Bin paper contribution)
        if hasattr(self.scheduler, 'record_batch_composition') and bin_idx >= 0:
            self.scheduler.record_batch_composition(batch, bin_idx)
        
        # Mark service start time for all requests in batch
        for req in batch:
            req.start_service_time = self.current_time
            req.assigned_gpu = gpu.gpu_id
        
        # Update GPU state
        gpu.busy = True
        gpu.current_batch = batch
        gpu.free_at = self.current_time + service_time
        gpu.total_batches += 1
        gpu.total_requests += len(batch)
        gpu.total_busy_time += service_time
        self._idle_gpus.discard(gpu.gpu_id)  # Remove from idle set
        
        # Schedule GPU_FREE event
        self._schedule_event(Event(
            time=gpu.free_at,
            event_type=EventType.GPU_FREE,
            payload=gpu.gpu_id,
        ))
    
    def get_gpu_stats(self) -> List[dict]:
        """
        Get statistics for each GPU.
        
        Returns:
            List of dictionaries with per-GPU statistics
        """
        total_time = max(gpu.free_at for gpu in self.gpus) if self.gpus else 0.0
        
        stats = []
        for gpu in self.gpus:
            utilization = gpu.total_busy_time / total_time if total_time > 0 else 0.0
            stats.append({
                'gpu_id': gpu.gpu_id,
                'total_batches': gpu.total_batches,
                'total_requests': gpu.total_requests,
                'total_busy_time': gpu.total_busy_time,
                'utilization': utilization,
            })
        
        return stats
    
    def _get_bin_idx(self, predicted_output_len: int) -> int:
        """
        Determine which bin a request belongs to based on predicted output length.
        
        Args:
            predicted_output_len: Predicted output length in tokens
        
        Returns:
            Bin index (0 to K_BINS-1), or -1 if not using bins
        """
        if not hasattr(self.cfg, 'BIN_BOUNDARIES'):
            return -1
        
        for i, (min_len, max_len) in enumerate(self.cfg.BIN_BOUNDARIES):
            if min_len <= predicted_output_len < max_len:
                return i
        return len(self.cfg.BIN_BOUNDARIES) - 1
    
    def get_batch_composition_stats(self) -> dict:
        """
        Get batch composition statistics from Multi-Bin scheduler.
        
        This provides evidence for the Multi-Bin paper's contribution:
        - How binning improves batch composition
        - Reduced E[max(t_j) | bin] via narrower length distributions
        - Improved throughput via better composition
        
        Returns:
            Dictionary with composition efficiency metrics, or empty dict if not Multi-Bin
        """
        if hasattr(self.scheduler, 'get_composition_summary'):
            return self.scheduler.get_composition_summary()
        return {}