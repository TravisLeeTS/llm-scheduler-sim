"""
Configuration for Multi-Bin + Dynamic Batching scheduler.

Level 4 Production Configuration: Real BurstGPT workload + GPU-calibrated latency
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class SchedulerConfig:
    """
    Configuration for Multi-Bin + Dynamic Batching scheduler.
    
    This implements the hybrid approach combining:
    - Multi-Bin Batching (Guldogan et al.): Bins for batch composition efficiency
    - SLA-Constrained Dynamic Batching: Adaptive batch sizing with feedback control
    
    Level 4 Production Mode (Default):
    - BurstGPT dataset arrivals (real Azure traces)
    - GPU-calibrated latency model (Qwen3 1.7B on RTX 4080 12GB)
    - Stress testing mode: 200x RPS scaling (0.27→54 req/s) for high-pressure evaluation
    - Real timestamps available via USE_REAL_TIMESTAMPS=True for realistic benchmarking
    - Demonstrates clear scheduler differences under load
    """
    
    # ===== GPU Infrastructure =====
    NUM_GPUS: int = 4              # Number of parallel GPUs (4 for high-pressure testing)
    M_MAX_GB: float = 12.0         # GPU memory capacity (RTX 4080: 12GB)
    M_MODEL_GB: float = 4.0        # Model VRAM footprint (Qwen3 1.7B FP16: ~4.0GB)
    KV_MEM_PER_TOKEN_GB: float = 1.875e-4  # KV cache memory per token (1.7B: 2×24 layers×2048 hidden×2 bytes = 196,608 bytes = 0.1875 MB/token)
    
    # ===== Multi-Bin Configuration (Paper-Faithful) =====
    K_BINS: int = 4                # Number of bins for length-based batching
    USE_EQUAL_MASS_BINS: bool = True  # Equal probability mass per bin (Lemma 4.1)
    
    # Bin boundaries: None = computed dynamically from workload (equal-mass/quantile-based)
    # If provided, must be a list of (min, max) tuples with length == K_BINS
    BIN_BOUNDARIES: List[Tuple[int, int]] = None  # Computed from workload data
    
    # Per-bin batch size limits: None = computed dynamically based on bin characteristics
    # Formula: b_max_k = min(B_MAX, M_avail / (avg_seq_len_k × kv_mem_per_token))
    # Longer bins → smaller batch limits (memory + latency constrained)
    BIN_B_MAX: List[int] = None  # Computed from bin boundaries
    
    BIN_SELECTION_POLICY: str = "round_robin"  # "round_robin" or "longest_queue"
    
    # ===== Dynamic Batching Parameters =====
    B_MIN: int = 1                 # Minimum batch size
    B_MAX: int = 128               # Maximum batch size
    MAX_CANDIDATES: int = 128      # Candidate pool size per GPU scheduling (match B_MAX)
    
    # ===== SLA Constraints (v2: TTFT/TBT Separation) =====
    # Calibrated for RTX 4080 12GB + Qwen3 1.7B FP16
    # 
    # Latency model: t(b, L) = α + β × L × h(b)
    #   α = 60ms (TTFT - Time To First Token / prefill latency)
    #   β = 5.74ms/token (decode TBT - per-token generation)
    #   γ = 0.316 (batch penalty: h(b) = 1 + γ(b-1)/b)
    #
    # v2 SLA Model (TTFT/TBT Separation):
    # - Token SLA applies ONLY to decode TBT (β × h(b)), NOT TTFT
    # - This eliminates structural violations where TTFT/L dominates
    # - At β=5.74ms and b=8, decode TBT ≈ 5.74×1.25 = 7.2ms
    # - Token SLA of 10ms gives ~38% headroom for batching overhead
    # - Token SLA of 30ms gives ~4× headroom (for stress testing)
    #
    # Per-Token SLA (decode TBT only - excludes TTFT)
    # STRICT: 10ms (decode only, ~38% headroom over baseline β=5.74ms)
    D_SLA_TOKEN: float = 0.010     # 10ms per-token decode latency (STRICT)
    D_SLA_TOKEN_LOOSE: float = 0.030  # 30ms for stress test comparisons
    
    # Per-Request SLA (total response latency including queueing + TTFT)
    # DEFAULT: 10s = realistic user-facing latency target
    D_SLA_REQUEST: float = 10.0    # 10s end-to-end request latency (DEFAULT)
    D_SLA_REQUEST_STRICT: float = 5.0   # 5s for tight SLA testing
    D_SLA_REQUEST_LOOSE: float = 20.0   # 20s for long generation stress tests
    
    # Legacy alias for backward compatibility (uses per-token TBT)
    D_SLA: float = 0.010           # Alias for D_SLA_TOKEN (backward compatibility)
    
    LATENCY_EPSILON: float = 0.005 # TBT tolerance band (5ms)
    MEMORY_MARGIN_GB: float = 1.0  # Safety margin for memory constraint
    
    # ===== Workload Configuration (High-Pressure Production) =====
    NUM_REQUESTS: int = 10000      # Number of requests to simulate (10K for realistic testing)
    SEED: int = 42                 # Random seed for reproducibility
    
    # Level 4: BurstGPT dataset only (real Azure ChatGPT traces)
    WORKLOAD_SOURCE: str = "burstgpt_dataset"  # Level 4 production mode
    DATASET_PATH: str = "data/BurstGPT_sample.csv"
    USE_REAL_TIMESTAMPS: bool = False  # False=RPS scaling (stress testing), True=real timestamps (realistic benchmarking)
    RPS_SCALING: float = 200.0         # RPS scaling factor (200x = 0.27→54 req/s for stress testing)
    
    # Experiment mode (for compatibility)
    EXPERIMENT_MODE: str = "multi_bin_dynamic"
    B_FIXED: int = 32              # Fixed batch size (for multi_bin_only mode)
    USE_REAL_MODEL: bool = False   # Use vLLM (deprecated - use USE_REAL_CALIBRATION)
    
    # ===== GPU Calibration (Level 4 Production) =====
    USE_REAL_CALIBRATION: bool = True  # Use real GPU calibration data (Level 4 default)
    CALIBRATION_CSV_PATH: str = "data/qwen3_1_7b_latency_grid.csv"  # RTX 4080 12GB measurements (Qwen3 1.7B FP16)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # BIN_BOUNDARIES will be computed dynamically from workload if None
        # This validation only applies if boundaries are explicitly provided
        if self.BIN_BOUNDARIES is not None:
            if self.K_BINS != len(self.BIN_BOUNDARIES):
                self.BIN_BOUNDARIES = self.BIN_BOUNDARIES[:self.K_BINS]
                if len(self.BIN_BOUNDARIES) < self.K_BINS:
                    raise ValueError(
                        f"K_BINS ({self.K_BINS}) must match BIN_BOUNDARIES length "
                        f"({len(self.BIN_BOUNDARIES)})"
                    )
        
        if self.NUM_GPUS < 1:
            raise ValueError("NUM_GPUS must be >= 1")
        
        if self.K_BINS < 1:
            raise ValueError("K_BINS must be >= 1")
    
    def compute_bin_boundaries(self, predicted_lengths: List[int]) -> List[Tuple[int, int]]:
        """
        Compute equal-mass bin boundaries from predicted output lengths.
        
        Uses quantile-based boundaries so each bin has approximately 
        equal number of requests (paper requirement: Lemma 4.1).
        
        Args:
            predicted_lengths: List of predicted output lengths from workload
            
        Returns:
            List of (min, max) tuples for each bin
        """
        boundaries = compute_equal_mass_boundaries(predicted_lengths, self.K_BINS)
        self.BIN_BOUNDARIES = boundaries
        
        # Also compute per-bin batch limits based on boundaries
        self.BIN_B_MAX = self._compute_bin_b_max(boundaries)
        
        return boundaries
    
    def _compute_bin_b_max(self, boundaries: List[Tuple[int, int]]) -> List[int]:
        """
        Compute per-bin maximum batch sizes based on bin characteristics.
        
        Formula: b_max_k = min(B_MAX, floor(M_avail / (avg_seq_len_k × kv_mem_per_token)))
        
        Longer bins → smaller batch limits due to:
        1. Higher memory consumption (KV cache scales with sequence length)
        2. Higher per-token latency (larger max in batch)
        
        Args:
            boundaries: List of (min, max) tuples for each bin
            
        Returns:
            List of maximum batch sizes for each bin
        """
        M_avail = self.M_MAX_GB - self.M_MODEL_GB - self.MEMORY_MARGIN_GB
        bin_b_max = []
        
        for min_len, max_len in boundaries:
            # Use midpoint of bin as representative sequence length
            # Cap max_len at 4096 for practical purposes
            effective_max = min(max_len, 4096)
            avg_seq_len = (min_len + effective_max) // 2
            
            # Assume average prompt length is similar to output length
            # Total tokens per request ≈ 2 × avg_output_len
            total_tokens_per_req = avg_seq_len * 2
            
            if total_tokens_per_req > 0 and self.KV_MEM_PER_TOKEN_GB > 0:
                # Memory-based batch limit
                mem_per_req = total_tokens_per_req * self.KV_MEM_PER_TOKEN_GB
                b_max_mem = int(M_avail / mem_per_req) if mem_per_req > 0 else self.B_MAX
            else:
                b_max_mem = self.B_MAX
            
            # Clamp to [1, B_MAX]
            b_max_k = max(1, min(self.B_MAX, b_max_mem))
            bin_b_max.append(b_max_k)
        
        return bin_b_max
    
    def get_batch_composition_stats(self) -> dict:
        """
        Get statistics about batch composition efficiency from Multi-Bin paper.
        
        Returns:
            Dictionary with bin range statistics for throughput analysis
        """
        if self.BIN_BOUNDARIES is None:
            return {"status": "Bin boundaries not yet computed from workload"}
        
        stats = {}
        for i, (min_len, max_len) in enumerate(self.BIN_BOUNDARIES):
            range_size = max_len - min_len if max_len != 10000 else "unbounded"
            b_max = self.BIN_B_MAX[i] if self.BIN_B_MAX and i < len(self.BIN_B_MAX) else self.B_MAX
            stats[f"bin_{i}"] = {
                "range": (min_len, max_len),
                "range_size": range_size,
                "b_max": b_max,
                "purpose": f"Reduce E[max(t_j) | bin] via narrower intervals"
            }
        return stats


def compute_equal_mass_boundaries(predicted_lengths: List[int], k_bins: int) -> List[Tuple[int, int]]:
    """
    Compute bin boundaries with equal probability mass (paper requirement).
    
    Each bin should have approximately equal number of requests based on
    the empirical distribution of predicted output lengths.
    
    Args:
        predicted_lengths: Array of predicted output lengths from workload
        k_bins: Number of bins
    
    Returns:
        List of (min, max) tuples for each bin
    """
    if k_bins == 1:
        return [(0, 10000)]
    
    # Compute quantiles
    quantiles = np.linspace(0, 1, k_bins + 1)
    boundary_points = np.quantile(predicted_lengths, quantiles)
    
    bin_boundaries = []
    for i in range(k_bins):
        min_len = int(boundary_points[i])
        max_len = int(boundary_points[i + 1]) if i < k_bins - 1 else 10000
        bin_boundaries.append((min_len, max_len))
    
    # Ensure last bin captures everything
    bin_boundaries[-1] = (bin_boundaries[-1][0], 10000)
    
    return bin_boundaries


def default_config() -> SchedulerConfig:
    """Return a default configuration."""
    return SchedulerConfig()
