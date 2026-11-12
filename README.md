# pumpfun-v2 competition entry

This entry provides a forecast for the pump.fun top-5 competition.

## Model
The model uses a horizon-aware Monte Carlo simulation over the log10 market cap in SOL. It calculates trajectories for candidates based on recent market behavior, including venue-specific drifts and noise. The final forecast is based on the simulated rank distributions at the specified horizon, with a correction for newcomers.

## How to run
```bash
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

This entry is implemented using only Python standard library modules and is deterministic (fixed random seed, no external dependencies).
