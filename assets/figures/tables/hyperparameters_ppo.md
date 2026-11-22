### PPO: Phase 1 configs (200k frames, seed 0)

| config | factor | change | best_val_return | final_val_return | selection_return | selection_ci | final_overflow_bin_minutes | final_urgent_service_rate | picked | rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| ppo_c00 | base | (base) | 44.87 | 35.05 | 43.14 | [40.8, 45.4] | 295.82 | 0.63 |  | Base config: trainer defaults tuned in the single-algorithm cards. |
| ppo_c01 | lr | lr=0.001 | 47.78 | 43.89 | 46.84 | [43.2, 49.8] | 198.17 | 0.79 |  | Higher learning rate: faster progress per rollout, tests the clip's protection. |
| ppo_c02 | clip_epsilon | clip_epsilon=0.1 | 38.89 | 36.43 | 35.84 | [32.1, 39.2] | 197.25 | 0.67 |  | Tighter clip: more conservative policy updates. |
| ppo_c03 | clip_epsilon | clip_epsilon=0.3 | 42.50 | 36.88 | 36.86 | [33.8, 39.8] | 262.41 | 0.59 |  | Looser clip: larger policy changes per rollout. |
| ppo_c04 | epochs | num_epochs=10 | 55.95 | 49.86 | 49.29 | [46.0, 52.3] | 78.89 | 0.87 | yes | More epochs per rollout: better sample use, risk of overfitting a rollout. |
| ppo_c05 | minibatch_size | minibatch_size=256 | 37.15 | 35.66 | 35.17 | [32.0, 38.0] | 267.68 | 0.64 |  | Larger minibatch: half the gradient steps, smoother each. |
| ppo_c06 | rollout_length | frames_per_batch=2048 | 43.71 | 43.71 | 42.14 | [38.9, 45.0] | 186.62 | 0.76 |  | Longer rollout (~36 shifts): fewer, better-averaged updates. |
| ppo_c07 | entropy | entropy_coef=0.05 | 39.33 | 37.84 | 31.72 | [29.2, 34.2] | 296.46 | 0.62 |  | Stronger entropy bonus: more exploration. |
| ppo_c08 | value_coef | value_coef=1.0 | 42.24 | 38.89 | 41.22 | [38.6, 43.8] | 285.87 | 0.69 |  | Heavier critic loss weight. |
| ppo_c09 | gae_lambda | gae_lambda=0.9 | 46.48 | 42.17 | 43.42 | [40.7, 46.0] | 212.60 | 0.64 |  | Lower GAE lambda: lower-variance, more biased advantages. |
