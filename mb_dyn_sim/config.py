"""
Configuration for the multi-bin dynamic scheduler (paper-faithful).
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class SchedulerConfig:
    """Configuration for the multi-bin dynamic scheduler (paper-faithful)."""
    
    # Logical cloud-style number of GPUs (simulated)
    NUM_GPUS: int = 1      # Configurable X, default 1 for local RTX 4080
    
    # Number of multi-bin queues (global bins)
    K_BINS: int = 4        # Configurable X (e.g., 1, 2, 4, 8)
    
    # Bin boundaries for predicted output length (tokens)
    # Each tuple is (min_tokens, max_tokens) for that bin
    BIN_BOUNDARIES: List[Tuple[int, int]] = field(default_factory=lambda: [
        (0, 64),        # short
        (64, 256),      # medium
        (256, 1024),    # long
        (1024, 10000),  # very long
    ])
    
    # Equal-mass bin boundaries (paper requirement)
    USE_EQUAL_MASS_BINS: bool = True  # Use empirical quantiles for bin boundaries
    
    # SLA and resource parameters
    D_SLA: float = 1.0             # seconds
    M_MAX_GB: float = 12.0         # RTX 4080 12GB
    M_MODEL_GB: float = 2.0        # approx VRAM footprint for Qwen3-0.6B
    KV_MEM_PER_TOKEN_GB: float = 5e-6  # rough KV cache per token
    
    # Dynamic batching behavior
    MAX_CANDIDATES: int = 64
    BASE_LATENCY: float = 0.01     # base compute latency in seconds
    MEMORY_MARGIN_GB: float = 1.0
    LATENCY_EPSILON: float = 0.05
    
    # Dynamic batching constraints (Algorithm 1 & 2 from paper)
    B_MIN: int = 1                 # Minimum batch size
    B_MAX: int = 128               # Maximum batch size
    
    # Experiment mode (paper-faithful)
    EXPERIMENT_MODE: str = "multi_bin_dynamic"  # "multi_bin_only", "dynamic_only", "multi_bin_dynamic"
    B_FIXED: int = 32              # Fixed batch size for multi_bin_only mode
    
    # Workload parameters
    NUM_REQUESTS: int = 20000
    SEED: int = 42
    ARRIVAL_PROFILE: str = "burstgpt_like"  # "poisson", "burstgpt_like", "burstgpt_dataset"
    
    # Poisson arrival parameters (for poisson mode)
    POISSON_LAMBDA: float = 50.0   # requests/second
    
    # BurstGPT dataset parameters
    DATASET_PATH: str = ""         # Path to BurstGPT dataset CSV
    WORKLOAD_SOURCE: str = "synthetic"  # "synthetic" or "burstgpt_dataset"
    RPS_SCALING: float = 1.0       # RPS scaling factor for BurstGPT dataset
    
    # Bin selection policy
    BIN_SELECTION_POLICY: str = "round_robin"  # or "longest_queue"
    
    # Model calibration
    USE_REAL_MODEL: bool = False   # Use vLLM with Qwen3-0.6B for real latency
    CALIBRATE_ON_STARTUP: bool = False  # Run calibration on startup
    USE_REAL_CALIBRATION: bool = False  # Use real GPU calibration data
    CALIBRATION_CSV_PATH: str = ""  # Path to calibration CSV (if USE_REAL_CALIBRATION=True)
    
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
