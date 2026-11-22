### Held-out evaluation, high_demand (50 test seeds; RL pooled over 5 seeds)

| policy | return | ci_low | ci_high | seed_std | tonnes_collected | overflow_bin_minutes | urgent_service_rate | vehicles_not_returned | tonnes_left_on_trucks | terminal_penalty | episodes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rolling_horizon | 70.59 | 64.52 | 75.84 |  | 24.85 | 521.79 | 0.80 | 0.32 | 0.45 | -2.44 | 50 |
| urgency_first | 67.45 | 61.72 | 72.68 |  | 24.70 | 592.62 | 0.83 | 0.30 | 0.45 | -2.38 | 50 |
| nearest_job_first | 69.54 | 65.73 | 73.05 |  | 25.92 | 358.96 | 0.37 | 0.14 | 0.14 | -0.84 | 50 |
| random | -5.89 | -11.97 | -0.71 |  | 20.38 | 1561.37 | 0.34 | 2.68 | 4.97 | -25.24 | 50 |
| DQN (5 seeds) | 34.60 | 31.69 | 37.36 | 2.35 | 24.65 | 1013.67 | 0.69 | 2.34 | 5.13 | -25.19 | 250 |
| PPO (5 seeds) | 29.86 | 26.68 | 32.91 | 7.92 | 23.81 | 1124.96 | 0.72 | 2.26 | 4.33 | -21.85 | 250 |
| A2C (5 seeds) | 48.56 | 45.50 | 51.41 | 10.80 | 25.90 | 769.24 | 0.72 | 2.32 | 3.77 | -19.73 | 250 |
| REINFORCE (5 seeds) | 30.07 | 27.74 | 32.30 | 3.93 | 24.07 | 1051.88 | 0.67 | 2.57 | 4.29 | -22.29 | 250 |
