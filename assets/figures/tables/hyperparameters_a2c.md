### A2C: Phase 1 configs (200k frames, seed 0)

| config | factor | change | best_val_return | final_val_return | selection_return | selection_ci | final_overflow_bin_minutes | final_urgent_service_rate | picked | rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| a2c_c00 | base | (base) | 31.15 | 31.15 | 28.02 | [24.9, 30.8] | 414.27 | 0.51 |  | Base config: trainer defaults tuned in the single-algorithm cards. |
| a2c_c01 | lr | lr=0.0003 | 17.38 | 14.87 | 16.93 | [13.6, 20.2] | 513.11 | 0.28 |  | Lower learning rate: steadier but slower actor-critic updates. |
| a2c_c02 | lr | lr=0.003 | 23.12 | 6.33 | 18.77 | [15.4, 21.9] | 1007.66 | 0.52 |  | Higher learning rate: faster, risk of policy collapse with one step per batch. |
| a2c_c03 | rollout_length | frames_per_batch=64 | 31.54 | 19.54 | 32.91 | [29.9, 35.7] | 432.10 | 0.54 | yes | Short rollouts (~1 shift): many small, noisy updates. |
| a2c_c04 | rollout_length | frames_per_batch=1024 | 18.91 | 18.04 | 10.31 | [7.0, 13.7] | 478.60 | 0.39 |  | Long rollouts (~18 shifts): few, low-variance updates. |
| a2c_c05 | entropy | entropy_coef=0.05 | 27.46 | 27.46 | 23.41 | [19.9, 26.5] | 335.50 | 0.44 |  | Stronger entropy bonus: more exploration, slower commitment. |
| a2c_c06 | value_coef | value_coef=0.25 | 36.25 | 34.62 | 29.36 | [25.8, 32.5] | 293.01 | 0.52 |  | Lighter critic loss weight relative to the policy loss. |
| a2c_c07 | value_coef | value_coef=1.0 | 28.29 | 26.50 | 23.72 | [20.4, 26.8] | 467.12 | 0.34 |  | Heavier critic loss weight: faster value fitting. |
| a2c_c08 | gae_lambda | gae_lambda=0.8 | 12.90 | 6.41 | 2.79 | [-0.4, 5.9] | 523.62 | 0.35 |  | Lower GAE lambda: leans on the critic, lower variance, more bias. |
| a2c_c09 | gae_lambda | gae_lambda=1.0 | 26.93 | 26.93 | 20.75 | [17.7, 23.7] | 507.49 | 0.46 |  | GAE lambda 1: Monte-Carlo-like advantages, high variance, no bias. |
