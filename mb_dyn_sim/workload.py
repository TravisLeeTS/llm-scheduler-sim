"""
Workload generation for LLM inference scheduling (Level 4 Production Mode).

Level 4 uses only BurstGPT dataset - real Azure ChatGPT production traces.
"""

import numpy as np
from dataclasses import dataclass
from typing import List
from .config import SchedulerConfig


@dataclass
class Request:
    """Represents a single LLM inference request."""
    id: int
    arrival_time: float
    prompt_len: int
    output_len: int
    predicted_output_len: int
    deadline: float
    
    # Fields filled during simulation
    start_service_time: float = -1.0
    completion_time: float = -1.0
    assigned_gpu: int = -1
    
    @property
    def latency(self) -> float:
        """Total latency from arrival to completion."""
        if self.completion_time < 0:
            return -1.0
        return self.completion_time - self.arrival_time
    
    @property
    def queueing_delay(self) -> float:
        """Time spent waiting before service starts."""
        if self.start_service_time < 0:
            return -1.0
        return self.start_service_time - self.arrival_time
    
    @property
    def service_time(self) -> float:
        """Time spent in actual service."""
        if self.completion_time < 0 or self.start_service_time < 0:
            return -1.0
        return self.completion_time - self.start_service_time
    
    @property
    def violates_sla(self) -> bool:
        """Whether this request violated its SLA deadline."""
        if self.completion_time < 0:
            return False
        return self.completion_time > self.deadline


def predict_output_len(prompt_len: int, alpha: float = 1.5, noise_std: float = 0.1) -> int:
    """
    Predict output length from prompt length.
    
    Uses a simple power-law relationship with Gaussian noise.
    
    Args:
        prompt_len: Input prompt length in tokens
        alpha: Scaling exponent
        noise_std: Standard deviation of multiplicative noise
    
    Returns:
        Predicted output length in tokens
    """
    # Base prediction using power law
    base_prediction = (prompt_len ** 0.7) * alpha
    
    # Add multiplicative noise
    noise_factor = np.random.lognormal(mean=0, sigma=noise_std)
    prediction = base_prediction * noise_factor
    
    # Clamp to reasonable range
    prediction = max(1, min(int(prediction), 4096))
    
    return prediction





def load_burstgpt_dataset(dataset_path: str, num_requests: int, d_sla: float = 1.0, 
                          use_real_timestamps: bool = True, rps_scaling: float = 1.0) -> List[Request]:
    """
    Load BurstGPT dataset from CSV file (real Azure ChatGPT traces).
    
    Expected CSV format:
    - arrival_time (seconds) or timestamp
    - prompt_length (tokens)
    - output_length (tokens) or completion_length
    
    Args:
        dataset_path: Path to CSV file
        num_requests: Number of requests to load
        d_sla: SLA deadline in seconds
        use_real_timestamps: If True, use actual timestamps from dataset (preserves real arrival patterns).
                           If False, use RPS scaling factor to compress/stretch times.
        rps_scaling: RPS scaling factor (only used if use_real_timestamps=False)
                     Example: rps_scaling=2.0 doubles the arrival rate
    
    Returns:
        List of Request objects
    """
    import pandas as pd
    import os
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"BurstGPT dataset not found: {dataset_path}")
    
    print(f"Loading BurstGPT dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path, nrows=num_requests)
    
    # Detect column names (handle different formats)
    arrival_col = None
    prompt_col = None
    output_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'arrival' in col_lower or 'timestamp' in col_lower:
            arrival_col = col
        elif 'request' in col_lower and 'token' in col_lower:
            prompt_col = col
        elif 'prompt' in col_lower:
            prompt_col = col
        elif 'response' in col_lower and 'token' in col_lower:
            output_col = col
        elif 'output' in col_lower or 'completion' in col_lower:
            output_col = col
    
    if not all([arrival_col, prompt_col, output_col]):
        raise ValueError(f"Dataset must have arrival_time/Timestamp, prompt/request tokens, and output/response tokens columns. Found: {df.columns.tolist()}")
    
    print(f"Loaded {len(df)} requests from dataset")
    
    if use_real_timestamps:
        print(f"Using REAL timestamps from dataset (preserves actual arrival patterns)")
    elif rps_scaling != 1.0:
        print(f"Applying RPS scaling factor: {rps_scaling:.2f}x")
    
    requests = []
    for i, row in df.iterrows():
        prompt_len = int(row[prompt_col])
        output_len = int(row[output_col])
        arrival_time = float(row[arrival_col])
        
        # Normalize arrival times to start from 0
        if i == 0:
            start_time = arrival_time
        arrival_time -= start_time
        
        # Apply RPS scaling only if not using real timestamps
        if not use_real_timestamps:
            arrival_time /= rps_scaling
        
        predicted_len = predict_output_len(prompt_len)
        
        req = Request(
            id=i,
            arrival_time=arrival_time,
            prompt_len=prompt_len,
            output_len=output_len,
            predicted_output_len=predicted_len,
            deadline=arrival_time + d_sla,
        )
        requests.append(req)
    
    return requests


def generate_workload(cfg: SchedulerConfig) -> List[Request]:
    """
    Generate workload from BurstGPT dataset (Level 4 production mode).
    
    Args:
        cfg: Scheduler configuration
    
    Returns:
        List of Request objects with arrival times, lengths, and deadlines
    """
    np.random.seed(cfg.SEED)
    
    # Level 4: Only BurstGPT dataset supported
    if cfg.WORKLOAD_SOURCE != "burstgpt_dataset":
        raise ValueError(
            f"Level 4 production mode only supports WORKLOAD_SOURCE='burstgpt_dataset', "
            f"got '{cfg.WORKLOAD_SOURCE}'"
        )
    
    if not cfg.DATASET_PATH:
        raise ValueError("DATASET_PATH must be set when using BurstGPT dataset")
    
    # Check if we should use real timestamps
    use_real_timestamps = getattr(cfg, 'USE_REAL_TIMESTAMPS', True)
    rps_scaling = getattr(cfg, 'RPS_SCALING', 1.0)
    
    return load_burstgpt_dataset(
        cfg.DATASET_PATH, 
        cfg.NUM_REQUESTS, 
        cfg.D_SLA, 
        use_real_timestamps=use_real_timestamps,
        rps_scaling=rps_scaling
    )
