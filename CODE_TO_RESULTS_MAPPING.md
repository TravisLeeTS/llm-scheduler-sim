# Code-to-Results Mapping

**Repository:** https://github.com/TravisLeeTS/llm-scheduler-sim

This document maps each table/figure in [`RESULTS_AND_DISCUSSION.md`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/RESULTS_AND_DISCUSSION.md) to the specific script that generated it.

---

## Quick Reference Table

| Result | Script | Output | 
|--------|--------|--------|
| **Tables 2.1-2.3** | See [Main Results](#1-tables-21-23-method-comparison) | `step2_comparison.csv` |
| **Table 2.7** | See [Grid Search](#2-table-27-optimal-configuration-grid-search) | `step1_grid_search.csv` |
| **Figures 1-5** | [`generate_analysis_plots.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/generate_analysis_plots.py) | [`figures/`](https://github.com/TravisLeeTS/llm-scheduler-sim/tree/main/figures) |
| **Section 5 Sensitivity** | [`generate_analysis_plots.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/generate_analysis_plots.py) | [`sensitivity_analysis.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/sensitivity_analysis.csv) |
| **Section 5 Multi-Seed** | [`generate_analysis_plots.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/generate_analysis_plots.py) | [`multi_seed_variance.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/multi_seed_variance.csv) |

---

## 1. Tables 2.1-2.3: Method Comparison

| Load Level | Script | Output |
|------------|--------|--------|
| **High (100×)** | [`run_step2.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/run_step2.py) | [`stress_test_final/step2_comparison.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/stress_test_final/step2_comparison.csv) |
| **Medium (10×)** | [`step2_low_load.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/step2_low_load.py) | [`stress_test_low_load/step2_comparison.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/stress_test_low_load/step2_comparison.csv) |
| **Low (1×)** | [`step2_ultra_low_load.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/step2_ultra_low_load.py) | [`stress_test_ultra_low_load/step2_comparison.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/stress_test_ultra_low_load/step2_comparison.csv) |

---

## 2. Table 2.7: Optimal Configuration Grid Search

| Load Level | Script | Output |
|------------|--------|--------|
| **High (100×)** | [`step1_grid_search.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/step1_grid_search.py) | [`stress_test_final/step1_grid_search.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/stress_test_final/step1_grid_search.csv) |
| **Medium (10×)** | [`step1_low_load.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/step1_low_load.py) | [`stress_test_low_load/step1_grid_search.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/stress_test_low_load/step1_grid_search.csv) |
| **Low (1×)** | [`step1_ultra_low_load.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/step1_ultra_low_load.py) | [`stress_test_ultra_low_load/step1_grid_search.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/stress_test_ultra_low_load/step1_grid_search.csv) |

---

## 3. Figures & Sensitivity Analysis

**Script:** [`scripts/generate_analysis_plots.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/scripts/generate_analysis_plots.py)

| Output | GitHub Link |
|--------|-------------|
| Fig. 1: Throughput vs GPU | [`fig1_throughput_vs_gpu.png`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/fig1_throughput_vs_gpu.png) |
| Fig. 2: SLA Pareto Frontier | [`fig2_sla_pareto_frontier.png`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/fig2_sla_pareto_frontier.png) |
| Fig. 3: Latency-Throughput | [`fig3_latency_throughput_tradeoff.png`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/fig3_latency_throughput_tradeoff.png) |
| Fig. 4: Scheduler Comparison | [`fig4_scheduler_comparison.png`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/fig4_scheduler_comparison.png) |
| Fig. 5: Sensitivity & Variance | [`fig5_sensitivity_variance.png`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/fig5_sensitivity_variance.png) |
| K-Bins & GPU Sensitivity | [`sensitivity_analysis.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/sensitivity_analysis.csv) |
| Multi-Seed Variance | [`multi_seed_variance.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/figures/multi_seed_variance.csv) |

---

## 4. Core Library Code

| Module | Purpose | GitHub Link |
|--------|---------|-------------|
| `config.py` | `SchedulerConfig` class | [`mb_dyn_sim/config.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/mb_dyn_sim/config.py) |
| `workload.py` | Workload generation | [`mb_dyn_sim/workload.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/mb_dyn_sim/workload.py) |
| `simulation.py` | Discrete-event simulator | [`mb_dyn_sim/simulation.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/mb_dyn_sim/simulation.py) |
| `schedulers.py` | FIFO, Dynamic, Multi-Bin | [`mb_dyn_sim/schedulers.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/mb_dyn_sim/schedulers.py) |
| `metrics.py` | SLA, throughput, latency | [`mb_dyn_sim/metrics.py`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/mb_dyn_sim/metrics.py) |

---

## 5. Input Data

| Data | GitHub Link |
|------|-------------|
| BurstGPT Workload Traces | [`data/BurstGPT_sample.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/data/BurstGPT_sample.csv) |
| GPU Latency Calibration | [`data/qwen3_1_7b_latency_grid.csv`](https://github.com/TravisLeeTS/llm-scheduler-sim/blob/main/data/qwen3_1_7b_latency_grid.csv) |

---

## 6. Directory Structure

```
https://github.com/TravisLeeTS/llm-scheduler-sim/
├── mb_dyn_sim/                  # Core simulation library
├── scripts/                     # Experiment scripts
├── data/                        # Input datasets
├── figures/                     # Generated figures & CSVs
├── stress_test_final/           # High load (100×) results
├── stress_test_low_load/        # Medium load (10×) results
├── stress_test_ultra_low_load/  # Low load (1×) results
└── RESULTS_AND_DISCUSSION.md    # Paper documentation
```

---

**Document Version**: 1.1  
**Last Updated**: December 4, 2025
