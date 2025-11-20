"""
Workload generation for BurstGPT-style bursty traffic.
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


def generate_burstgpt_arrivals(
    num_requests: int,
    seed: int,
    lambda_on: float = 100.0,  # arrival rate during ON periods (requests/sec)
    mean_on_duration: float = 2.0,  # seconds
    mean_off_duration: float = 5.0,  # seconds
    on_duration_shape: float = 0.5,  # lognormal shape parameter
    off_duration_shape: float = 0.8,  # lognormal shape parameter
) -> List[float]:
    """
    Generate BurstGPT-style bursty arrival times using ON/OFF process.
    
    Args:
        num_requests: Total number of requests to generate
        seed: Random seed
        lambda_on: Arrival rate during ON periods (requests/sec)
        mean_on_duration: Mean duration of ON periods (seconds)
        mean_off_duration: Mean duration of OFF periods (seconds)
        on_duration_shape: Shape parameter for ON duration lognormal
        off_duration_shape: Shape parameter for OFF duration lognormal
    
    Returns:
        List of arrival times in seconds
    """
    rng = np.random.RandomState(seed)
    arrival_times = []
    current_time = 0.0
    
    while len(arrival_times) < num_requests:
        # OFF period: no arrivals
        off_duration = rng.lognormal(
            mean=np.log(mean_off_duration),
            sigma=off_duration_shape
        )
        current_time += off_duration
        
        # ON period: Poisson arrivals
        on_duration = rng.lognormal(
            mean=np.log(mean_on_duration),
            sigma=on_duration_shape
        )
        on_end_time = current_time + on_duration
        
        # Generate Poisson arrivals during ON period
        while current_time < on_end_time and len(arrival_times) < num_requests:
            inter_arrival = rng.exponential(1.0 / lambda_on)
            current_time += inter_arrival
            if current_time < on_end_time:
                arrival_times.append(current_time)
    
    return sorted(arrival_times[:num_requests])


def generate_prompt_lengths(
    num_requests: int,
    seed: int,
    short_prob: float = 0.6,
    medium_prob: float = 0.3,
    long_prob: float = 0.1,
) -> List[int]:
    """
    Generate prompt lengths following a mixture distribution.
    
    Args:
        num_requests: Number of prompt lengths to generate
        seed: Random seed
        short_prob: Probability of short prompts (16-128 tokens)
        medium_prob: Probability of medium prompts (128-512 tokens)
        long_prob: Probability of long prompts (512-2048 tokens)
    
    Returns:
        List of prompt lengths
    """
    rng = np.random.RandomState(seed + 1)
    prompt_lengths = []
    
    for _ in range(num_requests):
        category = rng.choice(
            ['short', 'medium', 'long'],
            p=[short_prob, medium_prob, long_prob]
        )
        
        if category == 'short':
            length = int(rng.uniform(16, 128))
        elif category == 'medium':
            length = int(rng.uniform(128, 512))
        else:  # long
            length = int(rng.uniform(512, 2048))
        
        prompt_lengths.append(length)
    
    return prompt_lengths


def generate_output_lengths(
    prompt_lengths: List[int],
    seed: int,
    correlation: float = 0.3,
) -> List[int]:
    """
    Generate output lengths correlated with prompt lengths.
    
    Args:
        prompt_lengths: List of prompt lengths
        seed: Random seed
        correlation: Correlation strength between prompt and output length
    
    Returns:
        List of output lengths
    """
    rng = np.random.RandomState(seed + 2)
    output_lengths = []
    
    for prompt_len in prompt_lengths:
        # Base output length from a gamma distribution
        base_output = rng.gamma(shape=2.0, scale=50.0)
        
        # Add correlation with prompt length
        correlated_component = correlation * prompt_len * rng.uniform(0.2, 0.8)
        
        output_len = int(base_output + correlated_component)
        
        # Clamp to reasonable range
        output_len = max(1, min(output_len, 4096))
        output_lengths.append(output_len)
    
    return output_lengths


def generate_poisson_arrivals(num_requests: int, lambda_rate: float, seed: int) -> List[float]:
    """
    Generate pure Poisson arrivals (paper requirement for theory validation).
    
    Args:
        num_requests: Number of requests
        lambda_rate: Arrival rate (requests/second)
        seed: Random seed
    
    Returns:
        List of arrival times
    """
    rng = np.random.RandomState(seed)
    inter_arrivals = rng.exponential(scale=1.0/lambda_rate, size=num_requests)
    arrival_times = np.cumsum(inter_arrivals)
    return arrival_times.tolist()


def load_burstgpt_dataset(dataset_path: str, num_requests: int, d_sla: float = 1.0, rps_scaling: float = 1.0) -> List[Request]:
    """
    Load BurstGPT dataset from CSV file (Level 2 fidelity).
    
    Expected CSV format:
    - arrival_time (seconds) or timestamp
    - prompt_length (tokens)
    - output_length (tokens) or completion_length
    
    Args:
        dataset_path: Path to CSV file
        num_requests: Number of requests to load
        d_sla: SLA deadline in seconds
        rps_scaling: RPS scaling factor (>1 = compress time, <1 = stretch time)
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
    if rps_scaling != 1.0:
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
        
        # Apply RPS scaling: compress/stretch inter-arrival times
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
    Generate a complete workload of requests.
    
    Supports three modes:
    1. Synthetic (Poisson or BurstGPT-style)
    2. BurstGPT dataset (real traces)
    
    Args:
        cfg: Scheduler configuration
    
    Returns:
        List of Request objects with arrival times, lengths, and deadlines
    """
    np.random.seed(cfg.SEED)
    
    # Check if using real dataset
    if cfg.WORKLOAD_SOURCE == "burstgpt_dataset":
        if not cfg.DATASET_PATH:
            raise ValueError("DATASET_PATH must be set when WORKLOAD_SOURCE='burstgpt_dataset'")
        rps_scaling = getattr(cfg, 'RPS_SCALING', 1.0)
        return load_burstgpt_dataset(cfg.DATASET_PATH, cfg.NUM_REQUESTS, cfg.D_SLA, rps_scaling)
    
    # Generate synthetic workload
    # Generate arrival times
    if cfg.ARRIVAL_PROFILE == "burstgpt_like":
        arrival_times = generate_burstgpt_arrivals(cfg.NUM_REQUESTS, cfg.SEED)
    elif cfg.ARRIVAL_PROFILE == "poisson":
        lambda_rate = cfg.POISSON_LAMBDA if hasattr(cfg, 'POISSON_LAMBDA') else 50.0
        arrival_times = generate_poisson_arrivals(cfg.NUM_REQUESTS, lambda_rate, cfg.SEED)
    else:
        raise ValueError(f"Unknown arrival profile: {cfg.ARRIVAL_PROFILE}")
    
    # Generate prompt lengths
    prompt_lengths = generate_prompt_lengths(cfg.NUM_REQUESTS, cfg.SEED)
    
    # Generate actual output lengths
    output_lengths = generate_output_lengths(prompt_lengths, cfg.SEED)
    
    # Generate requests
    requests = []
    for i in range(cfg.NUM_REQUESTS):
        prompt_len = prompt_lengths[i]
        output_len = output_lengths[i]
        predicted_len = predict_output_len(prompt_len)
        
        req = Request(
            id=i,
            arrival_time=arrival_times[i],
            prompt_len=prompt_len,
            output_len=output_len,
            predicted_output_len=predicted_len,
            deadline=arrival_times[i] + cfg.D_SLA,
        )
        requests.append(req)
    
    return requests


class BurstGPTWorkload:
    """
    BurstGPT workload loader with RPS scaling support (Level 2 fidelity).
    
    This class provides a clean interface for loading BurstGPT traces
    with optional filtering and RPS scaling.
    """
    
    def __init__(
        self,
        csv_path: str,
        service_type: str = "api",  # "api" or "conversation"
        max_requests: int = None,
        rps_scaling: float = 1.0,
        d_sla: float = 1.0,
    ):
        """
        Initialize BurstGPT workload loader.
        
        Args:
            csv_path: Path to BurstGPT CSV file
            service_type: Filter by service type ("api" or "conversation")
            max_requests: Maximum number of requests to load (None = all)
            rps_scaling: RPS scaling factor (>1 = compress time, <1 = stretch)
            d_sla: SLA deadline in seconds
        """
        self.csv_path = csv_path
        self.service_type = service_type
        self.max_requests = max_requests
        self.rps_scaling = rps_scaling
        self.d_sla = d_sla
    
    def sample_requests(self) -> List[Request]:
        """
        Load and return Request objects from BurstGPT dataset.
        
        Returns:
            List of Request objects with:
            - arrival_time (scaled by rps_scaling)
            - prompt_len (request_length)
            - output_len (response_length)
            - predicted_output_len (predictor function)
        """
        import pandas as pd
        
        # Load dataset
        print(f"Loading BurstGPT workload from: {self.csv_path}")
        df = pd.read_csv(self.csv_path, nrows=self.max_requests)
        
        # Filter by service type if column exists
        if 'Log Type' in df.columns:
            service_map = {
                'api': 'API',
                'conversation': 'Conversation',
            }
            if self.service_type in service_map:
                df = df[df['Log Type'] == service_map[self.service_type]]
                print(f"Filtered to {len(df)} {self.service_type} requests")
        
        # Detect column names
        arrival_col = None
        prompt_col = None
        output_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'timestamp' in col_lower:
                arrival_col = col
            elif 'request' in col_lower and 'token' in col_lower:
                prompt_col = col
            elif 'response' in col_lower and 'token' in col_lower:
                output_col = col
        
        if not all([arrival_col, prompt_col, output_col]):
            raise ValueError(
                f"Dataset must have Timestamp, Request tokens, and Response tokens columns. "
                f"Found: {df.columns.tolist()}"
            )
        
        # Convert to requests
        requests = []
        start_time = None
        
        for i, row in df.iterrows():
            prompt_len = int(row[prompt_col])
            output_len = int(row[output_col])
            arrival_time = float(row[arrival_col])
            
            # Normalize to start from 0
            if start_time is None:
                start_time = arrival_time
            arrival_time -= start_time
            
            # Apply RPS scaling
            arrival_time /= self.rps_scaling
            
            # Predict output length
            predicted_len = predict_output_len(prompt_len)
            
            req = Request(
                id=len(requests),
                arrival_time=arrival_time,
                prompt_len=prompt_len,
                output_len=output_len,
                predicted_output_len=predicted_len,
                deadline=arrival_time + self.d_sla,
            )
            requests.append(req)
        
        print(f"Loaded {len(requests)} requests")
        if self.rps_scaling != 1.0:
            original_duration = requests[-1].arrival_time * self.rps_scaling
            scaled_duration = requests[-1].arrival_time
            print(f"RPS scaling {self.rps_scaling:.2f}x: {original_duration:.1f}s → {scaled_duration:.1f}s")
            print(f"Effective arrival rate: {len(requests)/scaled_duration:.2f} req/s")
        
        return requests
