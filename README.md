# Pumpfun V3 Model

This model forecasts the top 5 largest pump.fun coins by market cap 24 hours out.

## Model Description
- **Horizon-aware Monte Carlo:** Simulates log market caps over time, accounting for decaying growth rates and long-run drift.
- **Newcomer Reserve:** Incorporates probability for coins not yet born.
- **Zero-Fill Rows:** Ensures at least 100 rows per place.
- **Sizes:** Histogram mixed with a newcomer normal distribution, anchored at the last observed cap, with Monte Carlo weight decaying after 2 hours.

## How to Run
Run the main script providing the input and output directories:

```bash
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 repo/main.py
```

## Round 4 resubmission (2026-11-12)
Round 3 (commit f836b1a) was rejected by the server with ALL 24 validity checks passing and no problem given (stage: rejected, problem: null) — the grade phase gave no cause. main.py is byte-identical to round 3; resubmitting the same code to test whether the rejection was a transient grade failure. If round 4 is rejected the same way, the entry needs a hardening rebuild.
