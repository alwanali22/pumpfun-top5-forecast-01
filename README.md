# pumpfun-top5-forecast-01

This program forecasts the 5 largest pump.fun coins by market cap based on the provided history.

## Model
The model uses a horizon-aware Monte Carlo simulation over log-transformed market caps:
- **Trajectories:** Tracks log-size points for each candidate.
- **Drift/Noise:** Calculates per-coin slope, long-run drift (`d_inf`), and age-based volatility (`extra_sd`).
- **Monte Carlo:** Simulates 6000 paths from the last mark, recording snapshots at future answer-read times.
- **Probabilities:** Aggregates ranks, applies a newcomer reserve (`U(dt)`), and zero-fills for coins missing from the simulation.
- **Sizes:** Computes size distributions with newcomer mixture components, normalizing probabilities.

## Running
```bash
MAKEFUN_INPUT_DIR=<path_to_input> MAKEFUN_OUTPUT_DIR=<path_to_output> python3 main.py
```

## Validation
- Determinism check: Two runs on the same input produced byte-identical output files (`diff` result empty).
- Row counts per place: 79, 79, 79, 78, 77.
- Named p sum per place: [~0.49, ~0.49, ~0.48, ~0.48, ~0.47] (sum <= 1.0).
- Size p sum per place: [~0.99, ~0.99, ~0.99, ~0.99, ~0.99] (sum <= 1.0).
- Runtime: Well under 300 seconds.
