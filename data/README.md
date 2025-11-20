# Example BurstGPT Dataset Format

This directory should contain BurstGPT trace files (if available).

## Expected CSV Format

The dataset should be a CSV file with the following columns:

```csv
arrival_time,prompt_length,output_length
0.0,128,256
0.05,64,128
0.12,512,384
...
```

### Column Descriptions

- **arrival_time**: Timestamp or time in seconds when request arrived
- **prompt_length**: Number of tokens in the input prompt
- **output_length**: Number of tokens in the output/completion

## How to Use

### Option 1: Download BurstGPT Dataset

If you have access to the BurstGPT dataset from the paper:

1. Download the trace files
2. Convert to the CSV format above
3. Save to this directory (e.g., `burstgpt_trace.csv`)

### Option 2: Create Synthetic Dataset

Run the provided script to create a synthetic dataset:

```bash
python scripts/create_synthetic_dataset.py --num-requests 10000 --output data/synthetic_trace.csv
```

## Running with Dataset

Once you have a dataset file:

```bash
# Use BurstGPT dataset
python scripts/run_mb_dynamic.py \
    --arrival-profile burstgpt_dataset \
    --dataset-path data/burstgpt_trace.csv \
    --num-requests 10000 \
    --compare

# Or use synthetic arrivals (default)
python scripts/run_mb_dynamic.py \
    --arrival-profile burstgpt_like \
    --num-requests 10000 \
    --compare
```

## Dataset Sources

The BurstGPT paper: [Reference to paper]

If the dataset is not publicly available, use the synthetic workload generator
which produces statistically similar arrival patterns and length distributions.
