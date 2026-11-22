### Held-out evaluation, base (50 test seeds; RL pooled over 5 seeds)

| policy | return | ci_low | ci_high | seed_std | tonnes_collected | overflow_bin_minutes | urgent_service_rate | vehicles_not_returned | tonnes_left_on_trucks | terminal_penalty | episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rolling_horizon | 59.89 | 57.91 | 61.74 |  | 19.19 | 25.03 | 0.84 | 0.18 | 0.30 | -1.55 | 50 |
| urgency_first | 57.53 | 55.12 | 59.81 |  | 19.32 | 26.84 | 0.84 | 0.42 | 0.54 | -3.02 | 50 |
| nearest_job_first | 52.08 | 49.62 | 54.37 |  | 19.03 | 43.09 | 0.30 | 0.32 | 0.38 | -2.16 | 50 |
| random | 1.00 | -3.20 | 4.93 |  | 16.33 | 787.83 | 0.33 | 2.74 | 4.29 | -22.65 | 50 |
| DQN (5 seeds) | 54.12 | 52.28 | 55.94 | 3.44 | 19.52 | 135.30 | 0.77 | 0.72 | 1.29 | -6.60 | 250 |
| PPO (5 seeds) | 51.70 | 50.37 | 52.97 | 1.34 | 19.99 | 100.35 | 0.83 | 1.67 | 1.66 | -9.98 | 250 |
| A2C (5 seeds) | 48.62 | 46.82 | 50.27 | 2.08 | 20.15 | 145.45 | 0.79 | 2.11 | 2.07 | -12.48 | 250 |
| REINFORCE (5 seeds) | 34.15 | 32.32 | 35.85 | 5.36 | 19.02 | 281.28 | 0.74 | 2.66 | 3.09 | -17.68 | 250 |
