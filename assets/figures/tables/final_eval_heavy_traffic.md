### Held-out evaluation, heavy_traffic (50 test seeds; RL pooled over 5 seeds)

| policy | return | ci_low | ci_high | seed_std | tonnes_collected | overflow_bin_minutes | urgent_service_rate | vehicles_not_returned | tonnes_left_on_trucks | terminal_penalty | episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rolling_horizon | 51.85 | 48.62 | 54.57 |  | 18.11 | 102.19 | 0.81 | 0.24 | 0.40 | -2.08 | 50 |
| urgency_first | 48.22 | 43.41 | 52.32 |  | 17.99 | 137.70 | 0.81 | 0.32 | 0.51 | -2.68 | 50 |
| nearest_job_first | 44.22 | 41.79 | 46.47 |  | 18.36 | 99.80 | 0.28 | 0.30 | 0.50 | -2.58 | 50 |
| random | -12.29 | -17.32 | -7.96 |  | 15.23 | 938.65 | 0.30 | 2.82 | 5.11 | -26.10 | 50 |
| DQN (5 seeds) | 39.60 | 37.63 | 41.56 | 2.19 | 18.63 | 268.11 | 0.68 | 1.06 | 2.34 | -11.48 | 250 |
| PPO (5 seeds) | 39.92 | 38.44 | 41.34 | 1.23 | 19.07 | 226.83 | 0.74 | 1.85 | 2.07 | -11.98 | 250 |
| A2C (5 seeds) | 37.56 | 36.00 | 39.06 | 1.05 | 19.14 | 272.86 | 0.70 | 2.26 | 2.37 | -13.99 | 250 |
| REINFORCE (5 seeds) | 21.27 | 19.48 | 23.03 | 6.46 | 17.81 | 470.99 | 0.64 | 2.65 | 3.48 | -19.22 | 250 |
