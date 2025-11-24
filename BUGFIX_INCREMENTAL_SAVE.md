# Bug Fix: Individual Steps Overwriting All Results

## Issue Description

When running individual test steps (e.g., `--step3-only`), all result CSV files were being **overwritten** with only the new step's data, losing results from previous runs.

### Example of the Problem:

```bash
# Run Step 1 - creates results with Step 1 data ✓
python scripts/comprehensive_stress_test_optimized.py --step1-only

# Run Step 2 - adds Step 2 data ✓
python scripts/comprehensive_stress_test_optimized.py --step2-only

# Run Step 3 - OVERWRITES Step 1 & 2 files! ✗
python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 32
```

**Result after Step 3:**
- `step1_request_scaling.csv` - **EMPTY** (data lost!)
- `step2_gpu_scaling.csv` - **EMPTY** (data lost!)
- `step3_kbins_sensitivity.csv` - Has Step 3 data only
- `all_results.csv` - Has Step 3 data only

## Root Cause

Two problems in the optimized script:

### Problem 1: Incremental Save Overwrites
```python
# OLD CODE (broken):
def save_incremental_results(all_results, output_path):
    df = pd.DataFrame(all_results)  # Only current step's results
    df.to_csv(temp_path, index=False)  # OVERWRITES incremental file ✗
```

**Issue:** Each step overwrites the incremental file instead of appending.

### Problem 2: Final Save Creates Empty Files
```python
# OLD CODE (borrowed from original script):
def save_results_to_csv(all_results, output_path):
    df = pd.DataFrame(all_results)  # Only current step's results
    
    # Filter into steps
    step1_df = df[...]  # Empty if running step2-only or step3-only
    step2_df = df[...]  # Empty if running step1-only or step3-only
    step3_df = df[...]  # Empty if running step1-only or step2-only
    
    # OVERWRITES all files, even empty ones! ✗
    step1_df.to_csv(step1_path, index=False)  
    step2_df.to_csv(step2_path, index=False)
    step3_df.to_csv(step3_path, index=False)
```

**Issue:** Always saves all 4 files, overwriting existing data with empty dataframes.

## Solution

### Fix 1: Incremental Save Appends Instead of Overwrites

```python
def save_incremental_results(all_results, output_path):
    """FIXED: Append to existing incremental file"""
    import os
    
    df = pd.DataFrame(all_results)
    temp_path = output_path.replace('.csv', '_incremental.csv')
    
    # Load existing incremental file if it exists
    if os.path.exists(temp_path):
        try:
            existing_df = pd.read_csv(temp_path)
            # Combine: existing + new, remove duplicates
            df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates()
        except Exception:
            pass
    
    df.to_csv(temp_path, index=False)  # Now includes all accumulated results ✓
```

### Fix 2: Smart Save Merges with Existing Results

```python
def save_results_to_csv_smart(all_results, output_path):
    """
    FIXED: Merge with existing results instead of overwriting.
    """
    import os
    
    # Load existing "all_results.csv" if it exists
    all_path = output_path.replace('.csv', '_all_results.csv')
    existing_all_df = pd.DataFrame()
    
    if os.path.exists(all_path):
        existing_all_df = pd.read_csv(all_path)
        print(f"Loaded {len(existing_all_df)} existing results")
    
    # Combine with new results
    new_df = pd.DataFrame(all_results)
    if not existing_all_df.empty:
        combined_df = pd.concat([existing_all_df, new_df], ignore_index=True)
        # Remove duplicates based on key columns
        combined_df = combined_df.drop_duplicates(
            subset=['scheduler_type', 'num_requests', 'num_gpus', 'k_bins', 'rps_scaling'],
            keep='last'  # Keep most recent result
        )
    else:
        combined_df = new_df
    
    # Separate steps from COMBINED data (includes old + new)
    step1_df = combined_df[...]  # Now includes Step 1 from previous runs ✓
    step2_df = combined_df[...]  # Now includes Step 2 from previous runs ✓
    step3_df = combined_df[...]  # Now includes Step 3 from previous runs ✓
    
    # Save all files (now with complete data)
    step1_df.to_csv(step1_path, index=False)
    step2_df.to_csv(step2_path, index=False)
    step3_df.to_csv(step3_path, index=False)
    combined_df.to_csv(all_path, index=False)
```

## How It Works Now

### Scenario: Running Steps Individually

```bash
# Step 1: Run request scaling tests
python scripts/comprehensive_stress_test_optimized.py --step1-only

# Results after Step 1:
#   step1_request_scaling.csv - 25 tests ✓
#   step2_gpu_scaling.csv - 0 tests (empty, expected)
#   step3_kbins_sensitivity.csv - 0 tests (empty, expected)
#   all_results.csv - 25 tests ✓

# Step 2: Run GPU scaling tests
python scripts/comprehensive_stress_test_optimized.py --step2-only

# Results after Step 2:
#   step1_request_scaling.csv - 25 tests ✓ (PRESERVED from Step 1!)
#   step2_gpu_scaling.csv - 8 tests ✓
#   step3_kbins_sensitivity.csv - 0 tests (empty, expected)
#   all_results.csv - 33 tests ✓ (25 from Step 1 + 8 from Step 2)

# Step 3: Run K-bins sensitivity tests
python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 32

# Results after Step 3:
#   step1_request_scaling.csv - 25 tests ✓ (PRESERVED!)
#   step2_gpu_scaling.csv - 8 tests ✓ (PRESERVED!)
#   step3_kbins_sensitivity.csv - 6 tests ✓
#   all_results.csv - 39 tests ✓ (25 + 8 + 6)
```

### Scenario: Re-running Same Step (Updates Results)

```bash
# First run Step 3
python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 32

# Fix a bug, re-run Step 3 with same parameters
python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 32

# Result: Step 3 tests are UPDATED (not duplicated)
# Duplicate detection based on: scheduler_type, num_requests, num_gpus, k_bins, rps_scaling
# Keep: 'last' (most recent run)
```

## Benefits

### Before Fix:
- ✗ Running individual steps lost previous results
- ✗ Had to run full test suite every time
- ✗ Couldn't iterate on specific steps
- ✗ Wasted hours re-running completed tests

### After Fix:
- ✓ Can run steps individually and accumulate results
- ✓ Previous results are preserved
- ✓ Can iterate/debug specific steps
- ✓ Re-running same step updates results (no duplicates)
- ✓ Incremental workflow supported

## Files Modified

- `scripts/comprehensive_stress_test_optimized.py`
  - `save_incremental_results()` - Now appends instead of overwrites
  - `save_results_to_csv_smart()` - New function that merges with existing data
  - Main function - Uses `save_results_to_csv_smart()` instead of original

## How to Use

### Incremental Workflow (Recommended)

```bash
# Run each step separately as you develop/test
python scripts/comprehensive_stress_test_optimized.py --step1-only
python scripts/comprehensive_stress_test_optimized.py --step2-only
python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 32

# Results accumulate automatically!
# Final files have all 3 steps combined
```

### Full Suite (Still Works)

```bash
# Run everything at once (if you prefer)
python scripts/comprehensive_stress_test_optimized.py

# Same result as incremental, just takes longer
```

### Re-run Specific Step

```bash
# Made changes to K-bins logic? Re-run Step 3 only
python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 32

# Old Step 1 and Step 2 results preserved ✓
# Step 3 results updated with new run ✓
```

### Clear Results and Start Fresh

```bash
# Delete result files to start from scratch
rm comprehensive_stress_results_optimized*.csv

# Then run tests normally
python scripts/comprehensive_stress_test_optimized.py --step1-only
```

## Duplicate Detection

Results are deduplicated based on:
- `scheduler_type` (e.g., "multi_bin_dynamic")
- `num_requests` (e.g., 1,000,000)
- `num_gpus` (e.g., 32)
- `k_bins` (e.g., 4)
- `rps_scaling` (e.g., 200.0)

If you re-run the same configuration, the **most recent result** is kept (strategy: `keep='last'`).

## Edge Cases Handled

1. **First run (no existing files)** - Works like before ✓
2. **Incremental file missing** - Creates new one ✓
3. **Corrupted CSV file** - Falls back to new results only ✓
4. **Re-running same config** - Updates result, no duplicates ✓
5. **Running full suite after incremental** - Merges correctly ✓

## Testing

Verify the fix works:

```bash
# Test incremental workflow
python scripts/comprehensive_stress_test_optimized.py --step1-only --max-requests 10000
python scripts/comprehensive_stress_test_optimized.py --step2-only --max-gpus 8
python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 8

# Check all_results.csv has data from all 3 steps
# Check step1/step2/step3 CSV files all have data

# Re-run Step 3 with different K values
python scripts/comprehensive_stress_test_optimized.py --step3-only --best-gpu-count 8

# Verify: Step 1 and Step 2 still have data
# Verify: Step 3 results updated
```

## Summary

The fix enables a **true incremental workflow** where you can:
1. Run test steps in any order
2. Re-run specific steps without losing other results
3. Build up comprehensive results over multiple runs
4. Avoid hours of redundant computation

This is especially valuable when:
- Debugging specific scheduler logic
- Tuning parameters for one step
- Running expensive tests in stages
- Recovering from crashes/interruptions
