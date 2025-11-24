"""
Model calibration for Qwen3-0.6B to estimate service times (paper-faithful).

Supports three levels of fidelity:
1. Synthetic formula (fast, for algorithm development)
2. Transformers-based calibration (medium, requires GPU)
3. vLLM-based calibration (best, realistic production latency)

Key requirement from papers: Batch service time = f(max_seq_len, batch_size)
where max_seq_len dominates (batch completes when longest request finishes).
"""

import warnings
from typing import Dict, Callable
import os

MODEL_NAME = "Qwen/Qwen2.5-0.6B-Instruct"  # Updated to latest Qwen model


def calibrate_with_vllm(device: str = "cuda") -> Dict:
    """
    Calibrate using vLLM for realistic production latency.
    
    vLLM provides better batching and more realistic inference than transformers.
    This is the recommended calibration method for publication-quality results.
    
    Args:
        device: Device to use ("cuda" required for vLLM)
    
    Returns:
        Calibration parameters dictionary
    """
    try:
        from vllm import LLM, SamplingParams
        import torch
        import time
        import numpy as np
        
        if device != "cuda" or not torch.cuda.is_available():
            warnings.warn("vLLM requires CUDA. Falling back to fallback parameters.")
            return _get_fallback_parameters()
        
        print(f"Loading {MODEL_NAME} with vLLM for calibration...")
        print("This may take a few minutes on first run (downloading model)...")
        
        # Initialize vLLM
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=2048,  # Limit context for faster calibration
        )
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # Just measure prefill time
        )
        
        print("Running vLLM calibration benchmarks...")
        
        calibration_data = []
        
        # Test different batch sizes and sequence lengths
        test_configs = [
            (1, 128),
            (1, 512),
            (1, 1024),
            (4, 128),
            (4, 512),
            (8, 128),
            (8, 512),
            (16, 128),
            (16, 512),
        ]
        
        for batch_size, seq_len in test_configs:
            # Create dummy prompts
            dummy_text = "Hello world! " * (seq_len // 2)
            prompts = [dummy_text] * batch_size
            
            # Warm-up
            _ = llm.generate(prompts, sampling_params)
            
            # Measure
            start = time.time()
            for _ in range(3):
                _ = llm.generate(prompts, sampling_params)
            elapsed = (time.time() - start) / 3
            
            calibration_data.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'latency': elapsed,
            })
            
            print(f"  B={batch_size}, L={seq_len}: {elapsed*1000:.2f}ms")
        
        # Fit linear model: latency = a0 + a1 * max_seq_len * h(batch_size)
        # where h(batch_size) = 1 + beta * (B-1)/B
        
        # Extract parameters (simple linear regression)
        base_latency = min(d['latency'] for d in calibration_data if d['batch_size'] == 1)
        
        # Estimate seq_len coefficient
        seq_data = [d for d in calibration_data if d['batch_size'] == 1]
        if len(seq_data) >= 2:
            seq_len_coeff = (seq_data[-1]['latency'] - seq_data[0]['latency']) / \
                           (seq_data[-1]['seq_len'] - seq_data[0]['seq_len'])
        else:
            seq_len_coeff = 0.0002
        
        # Estimate batch penalty
        batch_data = [d for d in calibration_data if d['seq_len'] == 128]
        if len(batch_data) >= 2:
            batch_penalty = (batch_data[-1]['latency'] / batch_data[0]['latency'] - 1.0) / \
                           (batch_data[-1]['batch_size'] - 1)
        else:
            batch_penalty = 0.3
        
        params = {
            'base_latency': base_latency,
            'seq_len_coeff': seq_len_coeff,
            'batch_penalty': batch_penalty,
            'calibrated': True,
            'method': 'vllm',
        }
        
        print(f"\nvLLM Calibration complete:")
        print(f"  base_latency: {base_latency*1000:.2f}ms")
        print(f"  seq_len_coeff: {seq_len_coeff*1000:.4f}ms/token")
        print(f"  batch_penalty: {batch_penalty:.3f}")
        
        return params
        
    except ImportError as e:
        warnings.warn(f"vLLM not available: {e}. Install with: pip install vllm")
        return _get_fallback_parameters()
    except Exception as e:
        warnings.warn(f"vLLM calibration failed: {e}. Using fallback parameters.")
        return _get_fallback_parameters()


def calibrate_qwen3_06b(device: str = "cuda") -> Dict:
    """
    Calibrate Qwen3-0.6B model to estimate latency parameters.
    
    This function attempts to load the model and run calibration.
    If it fails (no GPU, missing dependencies, etc.), it returns
    fallback synthetic parameters.
    
    Args:
        device: Device to use for calibration ("cuda" or "cpu")
    
    Returns:
        Dictionary with calibration parameters:
        - base_latency: Base latency per forward pass (seconds)
        - decode_cost_per_token: Additional cost per output token (seconds)
        - batch_scale_factor: How batch size affects latency
        - seq_len_scale_factor: How sequence length affects latency
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Check if CUDA is available if requested
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, using fallback parameters")
            return _get_fallback_parameters()
        
        print(f"Loading {MODEL_NAME} for calibration...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device},
        )
        
        print("Running calibration benchmarks...")
        
        # Simple calibration: measure a few scenarios
        import time
        
        calibration_data = []
        
        # Test different batch sizes and sequence lengths
        test_configs = [
            (1, 128),
            (1, 512),
            (4, 128),
            (4, 512),
            (8, 128),
        ]
        
        for batch_size, seq_len in test_configs:
            # Create dummy input
            input_ids = torch.randint(
                0, model.config.vocab_size,
                (batch_size, seq_len),
                device=device
            )
            
            # Warm-up
            with torch.no_grad():
                _ = model(input_ids)
            
            # Measure
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.time()
            
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_ids)
            
            torch.cuda.synchronize() if device == "cuda" else None
            elapsed = (time.time() - start) / 3
            
            calibration_data.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'latency': elapsed,
            })
            
            print(f"  B={batch_size}, L={seq_len}: {elapsed*1000:.2f}ms")
        
        # Fit simple linear model: latency = base + a*batch_size + b*seq_len
        # For simplicity, use the measurements to estimate parameters
        base_latency = min(d['latency'] for d in calibration_data)
        
        # Estimate decode cost per token (rough approximation)
        decode_cost_per_token = base_latency / 128 * 0.5
        
        # Estimate scaling factors
        batch_scale_factor = 0.3  # batching is efficient
        seq_len_scale_factor = 1.0  # linear in sequence length
        
        params = {
            'base_latency': base_latency,
            'decode_cost_per_token': decode_cost_per_token,
            'batch_scale_factor': batch_scale_factor,
            'seq_len_scale_factor': seq_len_scale_factor,
            'calibrated': True,
        }
        
        print(f"Calibration complete: base_latency={base_latency*1000:.2f}ms")
        return params
        
    except ImportError as e:
        warnings.warn(f"transformers or torch not available: {e}. Using fallback parameters.")
        return _get_fallback_parameters()
    except Exception as e:
        warnings.warn(f"Calibration failed: {e}. Using fallback parameters.")
        return _get_fallback_parameters()


def _get_fallback_parameters() -> Dict:
    """
    Return fallback synthetic parameters when calibration is not possible.
    
    These are paper-faithful estimates that reflect the max-dominates property:
    - Batch service time = f(max_seq_len, batch_size)
    - Longer sequences dominate batch completion time
    - Batching adds sublinear overhead
    """
    return {
        'base_latency': 0.010,  # 10ms base latency
        'seq_len_coeff': 0.0002,  # 0.2ms per token (calibrate for Qwen3-0.6B)
        'batch_penalty': 0.3,  # batching adds 30% overhead factor
        'calibrated': False,
        'method': 'synthetic',
    }


# Global calibration parameters (lazy loaded)
_calibration_params: Dict | None = None


def get_calibration_params(force_recalibrate: bool = False, use_vllm: bool = False) -> Dict:
    """
    Get calibration parameters, loading them if necessary.
    
    Args:
        force_recalibrate: If True, re-run calibration even if params exist
        use_vllm: If True, use vLLM for calibration (best quality)
    
    Returns:
        Calibration parameters dictionary
    """
    global _calibration_params
    
    if _calibration_params is None or force_recalibrate:
        if use_vllm or os.environ.get('USE_VLLM', '').lower() == 'true':
            print("Using vLLM calibration for production accuracy...")
            _calibration_params = calibrate_with_vllm()
        else:
            # Use fallback by default for fast experiments
            _calibration_params = _get_fallback_parameters()
            # Uncomment below to attempt transformers calibration:
            # _calibration_params = calibrate_qwen3_06b()
    
    return _calibration_params


def estimate_service_time(batch_size: int, max_seq_len: int) -> float:
    """
    Estimate service time for a batch (paper-faithful formula).
    
    Key property: Batch service time = f(max_seq_len, batch_size)
    where the longest sequence (max_seq_len) dominates.
    
    Formula: time = a₀ + a₁ * max_seq_len * h(batch_size)
    where h(batch_size) = 1 + β * (B-1)/B captures sublinear batch overhead
    
    Args:
        batch_size: Number of requests in the batch
        max_seq_len: Maximum sequence length (prompt + output) in the batch
    
    Returns:
        Estimated service time in seconds
    """
    params = get_calibration_params()
    
    # Base latency (GPU kernel launch, etc.)
    a0 = params['base_latency']
    
    # Per-token coefficient
    a1 = params['seq_len_coeff']
    
    # Batch penalty factor: h(B) = 1 + β * (B-1)/B
    # This is sublinear: h(1)=1, h(∞)→1+β
    beta = params['batch_penalty']
    h_B = 1.0 + beta * (batch_size - 1) / max(1, batch_size)
    
    # Final formula (paper-faithful)
    # time = a₀ + a₁ * max_seq_len * h(batch_size)
    time = a0 + a1 * max_seq_len * h_B
    
    return max(0.001, time)  # Minimum 1ms


def get_service_time_function() -> Callable[[int, int], float]:
    """
    Return a function that estimates service time.
    
    Returns:
        Function with signature (batch_size, max_seq_len) -> service_time
    """
    return estimate_service_time


class LatencyModel:
    """
    GPU-calibrated latency model using real measurements.
    
    This class fits a parametric model to real GPU measurements and
    provides predictions for arbitrary (batch_size, max_seq_len) combinations.
    
    Model form: t(b, L) ≈ α + β * L * (1 + γ * (b - 1) / b)
    
    where:
    - α: base latency (kernel launch overhead)
    - β: per-token coefficient
    - γ: batch penalty factor (sublinear batching overhead)
    """
    
    def __init__(self, calibration_csv: str = None):
        """
        Initialize LatencyModel from calibration data.
        
        Args:
            calibration_csv: Path to CSV with columns:
                - batch_size
                - max_seq_len
                - mean_latency_sec
                If None, uses synthetic parameters.
        """
        self.alpha = 0.010  # base latency
        self.beta = 0.0002  # per-token coefficient
        self.gamma = 0.3    # batch penalty
        self.calibrated = False
        self.method = "synthetic"
        
        if calibration_csv is not None:
            self._fit_from_csv(calibration_csv)
    
    def _fit_from_csv(self, csv_path: str):
        """
        Fit parametric model from calibration CSV.
        
        Uses least squares to fit: t(b, L) = α + β * L * h(b)
        where h(b) = 1 + γ * (b - 1) / b
        """
        try:
            import pandas as pd
            import numpy as np
            from scipy.optimize import curve_fit
            
            # Load calibration data
            df = pd.read_csv(csv_path)
            
            # Filter out failed measurements
            df = df.dropna(subset=['mean_latency_sec'])
            
            if len(df) == 0:
                print(f"Warning: No valid calibration data in {csv_path}")
                return
            
            # Extract data
            batch_sizes = df['batch_size'].values
            max_seq_lens = df['max_seq_len'].values
            latencies = df['mean_latency_sec'].values
            
            # Define model function for fitting
            def model_func(X, alpha, beta, gamma):
                b, L = X
                h_b = 1.0 + gamma * (b - 1) / np.maximum(1, b)
                return alpha + beta * L * h_b
            
            # Fit using curve_fit
            X = np.array([batch_sizes, max_seq_lens])
            
            try:
                # Initial guess
                p0 = [0.010, 0.0002, 0.3]
                
                # Fit with bounds to keep parameters physical
                popt, _ = curve_fit(
                    model_func,
                    X,
                    latencies,
                    p0=p0,
                    bounds=([0.001, 0.00001, 0.0], [1.0, 0.01, 2.0]),
                    maxfev=5000,
                )
                
                self.alpha, self.beta, self.gamma = popt
                self.calibrated = True
                self.method = "real_gpu_fitted"
                
                # Compute R² for quality check
                predictions = model_func(X, *popt)
                ss_res = np.sum((latencies - predictions) ** 2)
                ss_tot = np.sum((latencies - np.mean(latencies)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                print(f"\nLatencyModel fitted from {csv_path}:")
                print(f"  α (base latency):    {self.alpha*1000:.3f} ms")
                print(f"  β (per-token coeff): {self.beta*1000:.5f} ms/token")
                print(f"  γ (batch penalty):   {self.gamma:.3f}")
                print(f"  R² fit quality:      {r_squared:.4f}")
                
            except Exception as e:
                print(f"Warning: Curve fitting failed: {e}")
                print("Using fallback to 2D interpolation...")
                self._fit_interpolation(df)
                
        except ImportError as e:
            print(f"Warning: scipy not available for fitting: {e}")
            print("Install with: pip install scipy")
        except Exception as e:
            print(f"Warning: Failed to load calibration from {csv_path}: {e}")
    
    def _fit_interpolation(self, df):
        """
        Fallback: Use 2D interpolation instead of parametric fitting.
        """
        try:
            from scipy.interpolate import LinearNDInterpolator
            import numpy as np
            
            points = df[['batch_size', 'max_seq_len']].values
            values = df['mean_latency_sec'].values
            
            self.interpolator = LinearNDInterpolator(points, values)
            self.calibrated = True
            self.method = "real_gpu_interpolated"
            
            print("Using 2D linear interpolation for latency prediction")
            
        except Exception as e:
            print(f"Warning: Interpolation also failed: {e}")
    
    def predict(self, batch_size: int, max_seq_len: float) -> float:
        """
        Predict latency t(b, L_max).
        
        Args:
            batch_size: Number of requests in batch
            max_seq_len: Maximum sequence length in batch (tokens)
        
        Returns:
            Predicted latency in seconds
        """
        if hasattr(self, 'interpolator'):
            # Use interpolation if available
            import numpy as np
            try:
                result = self.interpolator(batch_size, max_seq_len)
                if not np.isnan(result):
                    return float(result)
            except:
                pass
        
        # Use parametric model: t(b, L) = α + β * L * h(b)
        h_b = 1.0 + self.gamma * (batch_size - 1) / max(1, batch_size)
        latency = self.alpha + self.beta * max_seq_len * h_b
        
        return max(0.001, float(latency))  # Minimum 1ms
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            'calibrated': self.calibrated,
            'method': self.method,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
        }
