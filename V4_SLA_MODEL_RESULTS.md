# v4 SLA Model: TTFT/TBT Separation Results

## Summary

The v4 (TTFT/TBT Separation) SLA model successfully eliminates structural Token SLA violations by computing decode-only TBT that excludes TTFT.

## Key Changes in v4 Model

### Before (v3 Legacy):
```
per_token_tbt = service_time / max_output_len
             = (α + β × L × h(b)) / L
             = α/L + β × h(b)      ← TTFT/L term dominates for short outputs!
```

### After (v4 Separation):
```
ttft = α                          ← ~60ms (tracked separately)
decode_tbt = β × h(b)             ← ~5.74ms × h(b), used for Token SLA
```

## Results Summary (100 configs tested)

### Token SLA (10ms threshold, decode-only)
| Metric | Value |
|--------|-------|
| Min violations | **0%** |
| Max violations | **0%** |
| Mean violations | **0%** |

### Decode TBT
| Metric | Value |
|--------|-------|
| Min | 5.74 ms |
| Max | 7.46 ms |
| Mean | 6.81 ms |
| Headroom vs 10ms | 31.9% |

### TTFT (Prefill Latency)
| Metric | Value |
|--------|-------|
| Mean | 59.65 ms |
| Expected (α) | ~60 ms |

### Request SLA (20s threshold) by GPU Count
| GPUs | Violation Rate |
|------|----------------|
| 1 | 98.2% |
| 2 | 95.2% |
| 4 | 71.4% |
| 8 | 46.6% |
| 16 | 23.5% |
| 32 | 3.7% |
| 64 | 0.4% |
| 100 | 0.1% |

## Interpretation

1. **Token SLA: SOLVED** - The v4 model eliminates all structural violations
   - Decode TBT (5.74-7.46ms) is consistently below 10ms threshold
   - 31.9% headroom allows for batch size growth under load

2. **Request SLA: Requires capacity** - High queueing delays at low GPU counts
   - At 200× RPS scaling (~54 req/s), need 32+ GPUs for <5% violations
   - This is expected behavior (queueing delay dominates at high load)

3. **TTFT Tracked Separately** - ~60ms prefill latency correctly isolated
   - No longer contaminates Token SLA calculation
   - Can be used for separate TTFT SLA if needed

## Model Parameters

```
Latency Model: t(b, L) = α + β × L × h(b)
  α (TTFT):     59.653 ms
  β (decode):   5.742 ms/token
  γ (batch):    0.316
  R² fit:       0.9995

SLA Thresholds:
  D_SLA_TOKEN:   10 ms (decode TBT only)
  D_SLA_REQUEST: 20 s
  RPS scaling:   200× (~54 req/s)
```

## Conclusion

The TTFT/TBT separation model (v4) successfully addresses the fundamental SLA violation issue. Token SLA violations are now 0% across all tested configurations, confirming the model correctly evaluates streaming decode performance without the confounding effect of prefill latency.
