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
    - GPU-calibrated latency model (Qwen3 1.7B on RTX 4080)
    - Stress testing mode: 200x RPS scaling (0.27→54 req/s) for high-pressure evaluation
    - Real timestamps available via USE_REAL_TIMESTAMPS=True for realistic benchmarking
    - Demonstrates clear scheduler differences under load
    """
    
    # ===== GPU Infrastructure =====
    NUM_GPUS: int = 4              # Number of parallel GPUs (4 for high-pressure testing)
    M_MAX_GB: float = 12.0         # GPU memory capacity (RTX 4080: 12GB)
    M_MODEL_GB: float = 2.0        # Model VRAM footprint (Qwen 1.7B)
    KV_MEM_PER_TOKEN_GB: float = 5e-6  # KV cache memory per token
    
    # ===== Multi-Bin Configuration (Paper-Faithful) =====
    K_BINS: int = 4                # Number of bins for length-based batching (modest)
    USE_EQUAL_MASS_BINS: bool = True  # Equal probability mass per bin (Lemma 4.1)
    BIN_BOUNDARIES: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 64),        # Bin 0: short outputs
        (64, 256),      # Bin 1: medium outputs
        (256, 1024),    # Bin 2: long outputs
        (1024, 10000),  # Bin 3: very long outputs
    ])
    BIN_SELECTION_POLICY: str = "round_robin"  # "round_robin" or "longest_queue"
    
    # ===== Dynamic Batching Parameters =====
    B_MIN: int = 1                 # Minimum batch size
    B_MAX: int = 128               # Maximum batch size
    MAX_CANDIDATES: int = 128      # Candidate pool size per GPU scheduling (match B_MAX)
    
    # ===== SLA Constraints (Realistic for LLM Inference) =====
    D_SLA: float = 1.0             # SLA deadline (seconds) - realistic for production LLM inference
    LATENCY_EPSILON: float = 0.1   # SLA tolerance band
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
    CALIBRATION_CSV_PATH: str = "data/qwen3_1_7b_latency_grid.csv"  # RTX 4080 measurements
    
    def __post_init__(self):
        """Validate configuration after initialization."""
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
    
    def get_batch_composition_stats(self) -> dict:
        """
        Get statistics about batch composition efficiency from Multi-Bin paper.
        
        Returns:
            Dictionary with bin range statistics for throughput analysis
        """
        stats = {}
        for i, (min_len, max_len) in enumerate(self.BIN_BOUNDARIES):
            range_size = max_len - min_len if max_len != 10000 else "unbounded"
            stats[f"bin_{i}"] = {
                "range": (min_len, max_len),
                "range_size": range_size,
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
