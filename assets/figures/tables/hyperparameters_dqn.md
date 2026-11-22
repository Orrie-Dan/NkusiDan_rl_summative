### DQN: Phase 1 configs (200k frames, seed 0)

| config | factor | change | best_val_return | final_val_return | selection_return | selection_ci | final_overflow_bin_minutes | final_urgent_service_rate | picked | rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| dqn_c00 | base | (base) | 29.54 | 28.15 | 29.34 | [26.3, 32.3] | 212.47 | 0.83 |  | Base config: trainer defaults tuned in the single-algorithm cards. |
| dqn_c01 | lr | lr=0.0001 | -8.73 | -8.73 | -10.48 | [-13.3, -7.6] | 1218.28 | 0.50 |  | Lower learning rate: slower but steadier Q-value fitting. |
| dqn_c02 | lr | lr=0.001 | 32.28 | 32.02 | 30.02 | [25.6, 34.0] | 334.51 | 0.60 |  | Higher learning rate: faster progress, risk of unstable bootstrapped targets. |
| dqn_c03 | gamma | gamma=0.95 | 38.18 | 30.54 | 35.14 | [32.5, 37.6] | 101.54 | 0.68 |  | Shorter horizon (~20 decisions): favours immediate pickups over planning. |
| dqn_c04 | exploration | eps_decay_frames=20000 | 41.39 | 29.29 | 37.64 | [34.6, 40.3] | 364.50 | 0.68 |  | Fast epsilon decay: exploit early, risk locking in a poor route habit. |
| dqn_c05 | exploration | eps_decay_frames=120000 | 42.36 | 37.18 | 44.32 | [40.8, 47.6] | 183.18 | 0.72 |  | Slow epsilon decay: explore for 60% of the budget. |
| dqn_c06 | replay | buffer_size=10000 | 35.86 | 33.93 | 29.89 | [26.4, 33.2] | 177.97 | 0.84 |  | Small replay buffer: fresher, more on-policy data, less diversity. |
| dqn_c07 | replay | batch_size=32 | -17.39 | -44.58 | -13.24 | [-15.7, -10.8] | 1860.47 | 0.49 |  | Small minibatch: noisier gradients, same number of updates. |
| dqn_c08 | replay | learning_starts=20000 | 44.94 | 39.08 | 44.38 | [39.6, 48.2] | 394.12 | 0.76 | yes | Late learning start: a more diverse buffer before the first update. |
| dqn_c09 | target_update | target_update=hard | 39.99 | 18.95 | 36.60 | [33.6, 39.3] | 363.40 | 0.64 |  | Hard target copy every 1k gradient steps (hard_update_interval) instead of soft Polyak updates. |
