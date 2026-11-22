### Validation (model selection) vs held-out test return, and return per scenario

| algorithm | val_best_return | test_return | drop | return_base | return_high_demand | return_high_breakdown | return_heavy_traffic |
|---|---|---|---|---|---|---|---|
| DQN | 60.31 | 54.12 | 6.19 | 54.12 | 34.60 | 32.23 | 39.60 |
| PPO | 56.79 | 51.70 | 5.09 | 51.70 | 29.86 | 35.09 | 39.92 |
| A2C | 51.87 | 48.62 | 3.25 | 48.62 | 48.56 | 33.75 | 37.56 |
| REINFORCE | 39.05 | 34.15 | 4.91 | 34.15 | 30.07 | 18.30 | 21.27 |
| rolling_horizon |  |  |  | 59.89 | 70.59 | 42.27 | 51.85 |
| urgency_first |  |  |  | 57.53 | 67.45 | 42.09 | 48.22 |
| nearest_job_first |  |  |  | 52.08 | 69.54 | 38.72 | 44.22 |
| random |  |  |  | 1.00 | -5.89 | -8.55 | -12.29 |
