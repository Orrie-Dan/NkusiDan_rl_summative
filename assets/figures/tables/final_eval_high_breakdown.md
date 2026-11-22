### Held-out evaluation, high_breakdown (50 test seeds; RL pooled over 5 seeds)

| policy | return | ci_low | ci_high | seed_std | tonnes_collected | overflow_bin_minutes | urgent_service_rate | vehicles_not_returned | tonnes_left_on_trucks | terminal_penalty | episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rolling_horizon | 42.27 | 35.13 | 48.69 |  | 17.22 | 263.34 | 0.80 | 0.90 | 1.64 | -8.35 | 50 |
| urgency_first | 42.09 | 36.67 | 47.11 |  | 17.42 | 216.18 | 0.79 | 0.96 | 1.66 | -8.56 | 50 |
| nearest_job_first | 38.72 | 32.97 | 43.80 |  | 17.61 | 222.35 | 0.31 | 0.94 | 1.55 | -8.07 | 50 |
| random | -8.55 | -13.00 | -4.21 |  | 14.36 | 1030.43 | 0.28 | 2.84 | 4.34 | -23.02 | 50 |
| DQN (5 seeds) | 32.23 | 29.16 | 35.17 | 3.46 | 17.42 | 447.25 | 0.61 | 1.70 | 2.54 | -13.55 | 250 |
| PPO (5 seeds) | 35.09 | 32.58 | 37.47 | 1.91 | 17.93 | 363.34 | 0.70 | 2.21 | 2.48 | -14.35 | 250 |
| A2C (5 seeds) | 33.75 | 31.32 | 36.14 | 1.44 | 18.08 | 388.57 | 0.66 | 2.49 | 2.66 | -15.61 | 250 |
| REINFORCE (5 seeds) | 18.30 | 15.68 | 20.86 | 6.03 | 16.57 | 576.56 | 0.62 | 2.84 | 3.52 | -19.77 | 250 |
