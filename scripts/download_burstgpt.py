#!/usr/bin/env python3
"""
Download BurstGPT dataset from official GitHub repository.

Source: https://github.com/HPMLL/BurstGPT
Paper: https://arxiv.org/pdf/2401.17644.pdf

Usage:
    python scripts/download_burstgpt.py [--version VERSION] [--no-fails]
    
Options:
    --version: 1 or 2 (default: 1)
               1 = First 2 months (~1.43M lines)
               2 = Second 2 months (~3.86M lines)
    --no-fails: Download version without failed requests (Response tokens = 0)
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path


# Official BurstGPT release URLs
BURSTGPT_RELEASES = {
    # Version 1 (first 2 months)
    (1, False): "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv",
    (1, True): "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_1.csv",
    # Version 2 (second 2 months)
    (2, False): "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_2.csv",
    (2, True): "https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv",
}

BURSTGPT_INFO = {
    (1, False): {"lines": "~1.43M", "size": "~50MB", "desc": "First 2 months with failures"},
    (1, True): {"lines": "~1.40M", "size": "~49MB", "desc": "First 2 months without failures"},
    (2, False): {"lines": "~3.86M", "size": "~135MB", "desc": "Second 2 months with failures"},
    (2, True): {"lines": "~3.78M", "size": "~132MB", "desc": "Second 2 months without failures"},
}


def download_progress(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    else:
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_downloaded:.1f} MB")
    sys.stdout.flush()


def download_burstgpt(version: int = 1, no_fails: bool = False, output_dir: str = "data"):
    """
    Download BurstGPT dataset from official GitHub releases.
    
    Args:
        version: 1 or 2 (first or second 2-month period)
        no_fails: If True, download version without failed requests
        output_dir: Directory to save the dataset
    """
    key = (version, no_fails)
    
    if key not in BURSTGPT_RELEASES:
        print(f"Error: Invalid combination (version={version}, no_fails={no_fails})")
        sys.exit(1)
    
    url = BURSTGPT_RELEASES[key]
    info = BURSTGPT_INFO[key]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Output filename
    output_file = output_path / "BurstGPT_sample.csv"
    
    print("=" * 60)
    print("BurstGPT Dataset Downloader")
    print("=" * 60)
    print(f"Source: https://github.com/HPMLL/BurstGPT")
    print(f"Version: {version} ({info['desc']})")
    print(f"Expected: {info['lines']} lines, {info['size']}")
    print(f"URL: {url}")
    print(f"Output: {output_file}")
    print()
    
    # Check if file already exists
    if output_file.exists():
        response = input(f"File {output_file} already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return
    
    print("Starting download...")
    try:
        urllib.request.urlretrieve(url, output_file, download_progress)
        print("\n")  # New line after progress
        
        # Verify download
        import pandas as pd
        df = pd.read_csv(output_file, nrows=5)
        expected_columns = ['Timestamp', 'Model', 'Request tokens', 'Response tokens', 'Total tokens', 'Log Type']
        
        if list(df.columns) == expected_columns:
            print("✓ Download successful!")
            print("✓ Schema verified (matches official BurstGPT format)")
            
            # Count actual lines
            with open(output_file, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # Subtract header
            print(f"✓ Total rows: {line_count:,}")
        else:
            print("⚠ Warning: Schema mismatch. File may be corrupted.")
            print(f"  Expected: {expected_columns}")
            print(f"  Got: {list(df.columns)}")
            
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Citation (please cite if using this dataset):")
    print("=" * 60)
    print("""
@inproceedings{BurstGPT,
  author    = {Yuxin Wang and Yuhan Chen and Zeyu Li and ...},
  title     = {{BurstGPT}: A Real-World Workload Dataset to Optimize LLM Serving Systems},
  booktitle = {KDD '25},
  year      = {2025},
}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download BurstGPT dataset from official GitHub repository."
    )
    parser.add_argument(
        "--version", type=int, default=1, choices=[1, 2],
        help="Dataset version: 1 (first 2 months) or 2 (second 2 months)"
    )
    parser.add_argument(
        "--no-fails", action="store_true",
        help="Download version without failed requests (Response tokens = 0)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available BurstGPT datasets:")
        print()
        for key, info in BURSTGPT_INFO.items():
            version, no_fails = key
            suffix = " (no failures)" if no_fails else ""
            print(f"  Version {version}{suffix}:")
            print(f"    {info['desc']}")
            print(f"    {info['lines']} lines, {info['size']}")
            print()
        return
    
    download_burstgpt(
        version=args.version,
        no_fails=args.no_fails,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
