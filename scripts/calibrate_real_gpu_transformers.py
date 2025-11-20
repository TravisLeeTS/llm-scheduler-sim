"""
Real GPU Calibration Script (Transformers version)

Windows-compatible GPU calibration using Hugging Face Transformers.
Measures Qwen2.5 latency on RTX 4080 for discrete-event simulator.

This script calibrates the latency model: T(b, L) = α + β·L·(1 + γ·(b-1)/b)
where:
  - α: base latency (overhead)
  - β: per-token coefficient  
  - γ: batch penalty factor

Usage:
    # Quick test (2-3 minutes)
    python scripts/calibrate_real_gpu_transformers.py --model Qwen/Qwen2.5-0.5B --batch-sizes 1 2 --max-seq-lens 128 256 --trials 2
    
    # Full calibration for Qwen2.5-1.5B (~30-45 minutes)
    python scripts/calibrate_real_gpu_transformers.py --model Qwen/Qwen2.5-1.5B --batch-sizes 1 2 4 8 --max-seq-lens 128 256 512 1024 --trials 3
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mb_dyn_sim.model_calibration_transformers import calibrate_latency_grid


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate GPU latency for LLM scheduler simulation (Transformers version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with Qwen2.5-0.5B
  python %(prog)s --model Qwen/Qwen2.5-0.5B --batch-sizes 1 2 --max-seq-lens 128 256 --trials 2
  
  # Full calibration with Qwen2.5-1.5B  
  python %(prog)s --model Qwen/Qwen2.5-1.5B --trials 3
  
  # Target model (may require quantization on 12GB GPU)
  python %(prog)s --model Qwen/Qwen3-1.7B --batch-sizes 1 2 4 --max-seq-lens 128 256 512
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model name (default: Qwen/Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to test (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--max-seq-lens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Max sequence lengths to test (default: 128 256 512 1024)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials per configuration (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: data/<model_name>_latency_grid.csv)",
    )

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output is None:
        model_shortname = args.model.split("/")[-1].lower().replace(".", "_").replace("-", "_")
        args.output = f"data/{model_shortname}_latency_grid.csv"

    # Validate CUDA availability
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("Please ensure:")
        print("  1. PyTorch with CUDA is installed:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
        print("  2. NVIDIA GPU drivers are installed")
        print("  3. GPU is properly detected (run: nvidia-smi)")
        print(f"\nCurrent PyTorch version: {torch.__version__}")
        sys.exit(1)

    print("="*70)
    print("REAL GPU CALIBRATION")
    print("="*70)
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Configurations: {len(args.batch_sizes)} batch sizes × {len(args.max_seq_lens)} seq lens × {args.trials} trials")
    print(f"Total measurements: {len(args.batch_sizes) * len(args.max_seq_lens)}")
    print(f"Estimated time: {len(args.batch_sizes) * len(args.max_seq_lens) * args.trials * 10 / 60:.0f}-{len(args.batch_sizes) * len(args.max_seq_lens) * args.trials * 20 / 60:.0f} minutes")
    print("="*70 + "\n")

    # Run calibration
    csv_path = calibrate_latency_grid(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        max_seq_lens=args.max_seq_lens,
        num_trials=args.trials,
        output_csv=args.output,
    )

    print(f"\n✓ Calibration complete!")
    print(f"\nNext steps:")
    print(f"  1. Run calibrated simulation:")
    print(f"     python scripts/run_mb_dynamic.py --use-real-calibration \\")
    print(f"            --calibration-csv {csv_path} --compare")
    print(f"\n  2. Test with BurstGPT workload:")
    print(f"     python scripts/run_mb_dynamic.py --use-real-calibration \\")
    print(f"            --calibration-csv {csv_path} \\")
    print(f"            --arrival-profile burstgpt_dataset \\")
    print(f"            --num-requests 20000 --rps-scaling 50.0 --compare")
    print(f"\n  3. Multi-GPU scaling test:")
    print(f"     python scripts/run_mb_dynamic.py --use-real-calibration \\")
    print(f"            --calibration-csv {csv_path} \\")
    print(f"            --num-gpus 4 --compare")


if __name__ == "__main__":
    main()
