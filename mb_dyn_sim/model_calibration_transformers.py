"""
Real GPU Calibration Module using Hugging Face Transformers

Alternative to vLLM for Windows users with CUDA support.
Measures actual batch latency on RTX 4080 12GB using HuggingFace Transformers.
Default model: Qwen3 1.7B FP16 (~4.0GB VRAM)
"""

from typing import Optional, List
import time
import numpy as np
import torch
from pathlib import Path


def measure_batch_latency(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    batch_size: int = 1,
    max_seq_len: int = 512,
    num_trials: int = 5,
) -> tuple[float, float]:
    """
    Measure batch latency using Hugging Face Transformers.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
        batch_size: Number of requests in the batch
        max_seq_len: Maximum sequence length (prompt + generation)
        num_trials: Number of trials to average over

    Returns:
        (mean_latency_sec, std_latency_sec)
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "transformers is required for GPU calibration. "
            "Install with: pip install transformers"
        )

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU calibration requires CUDA.")

    device = "cuda:0"
    print(f"Loading {model_name} on {torch.cuda.get_device_name(0)}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 for speed
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate synthetic prompts
    prompt_len = max(32, max_seq_len // 4)  # 25% prompt, 75% generation
    generation_len = max_seq_len - prompt_len

    # Create dummy prompts
    dummy_text = "This is a test prompt. " * (prompt_len // 6 + 1)
    prompts = [dummy_text[:200]] * batch_size  # Limit prompt text length

    # Tokenize batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_len,
    ).to(device)

    # Warmup run
    print(f"Warmup: batch_size={batch_size}, max_seq_len={max_seq_len}")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=min(generation_len, 100),  # Limit for warmup
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    torch.cuda.empty_cache()

    # Benchmark runs
    latencies = []
    print(f"Running {num_trials} trials...")
    for trial in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=generation_len,
                do_sample=False,  # Deterministic
                pad_token_id=tokenizer.pad_token_id,
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        latency = end_time - start_time
        latencies.append(latency)
        print(f"  Trial {trial + 1}/{num_trials}: {latency:.4f}s")

    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))

    # Cleanup
    del model, tokenizer, inputs
    torch.cuda.empty_cache()

    return mean_latency, std_latency


def calibrate_latency_grid(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    batch_sizes: Optional[List[int]] = None,
    max_seq_lens: Optional[List[int]] = None,
    num_trials: int = 3,
    output_csv: str = "data/qwen_latency_grid.csv",
) -> str:
    """
    Calibrate latency across a grid of (batch_size, max_seq_len) combinations.

    Args:
        model_name: HuggingFace model name
        batch_sizes: List of batch sizes to test
        max_seq_lens: List of max sequence lengths to test
        num_trials: Trials per configuration
        output_csv: Path to save results

    Returns:
        Path to saved CSV file
    """
    import pandas as pd

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]  # Smaller batch sizes for Transformers
    if max_seq_lens is None:
        max_seq_lens = [128, 256, 512, 1024]  # Moderate lengths

    # Ensure output directory exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    results = []
    total_configs = len(batch_sizes) * len(max_seq_lens)
    current = 0

    print(f"\n{'='*60}")
    print(f"GPU CALIBRATION: {model_name}")
    print(f"Using Hugging Face Transformers")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Testing {total_configs} configurations...")
    print(f"{'='*60}\n")

    for batch_size in batch_sizes:
        for max_seq_len in max_seq_lens:
            current += 1
            print(f"\n[{current}/{total_configs}] batch_size={batch_size}, max_seq_len={max_seq_len}")

            try:
                mean_latency, std_latency = measure_batch_latency(
                    model_name=model_name,
                    batch_size=batch_size,
                    max_seq_len=max_seq_len,
                    num_trials=num_trials,
                )

                results.append({
                    "batch_size": batch_size,
                    "max_seq_len": max_seq_len,
                    "mean_latency_sec": mean_latency,
                    "std_latency_sec": std_latency,
                    "num_trials": num_trials,
                })

                print(f"  ✓ Mean: {mean_latency:.4f}s, Std: {std_latency:.4f}s")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                # Record NaN for failed measurements
                results.append({
                    "batch_size": batch_size,
                    "max_seq_len": max_seq_len,
                    "mean_latency_sec": np.nan,
                    "std_latency_sec": np.nan,
                    "num_trials": 0,
                })

            # Clean up between runs
            torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Calibration complete! Results saved to: {output_csv}")
    print(f"{'='*60}\n")
    print(df.to_string(index=False))

    return output_csv


if __name__ == "__main__":
    # Quick test
    print("Testing GPU calibration with Transformers...")
    mean, std = measure_batch_latency(
        model_name="Qwen/Qwen2.5-0.5B",
        batch_size=1,
        max_seq_len=128,
        num_trials=2,
    )
    print(f"\nTest result: {mean:.4f}s ± {std:.4f}s")
