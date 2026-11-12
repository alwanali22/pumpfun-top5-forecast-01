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
