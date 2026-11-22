### REINFORCE: Phase 1 configs (200k frames, seed 0)

| config | factor | change | best_val_return | final_val_return | selection_return | selection_ci | final_overflow_bin_minutes | final_urgent_service_rate | picked | rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| reinforce_c00 | base | (base) | 25.86 | 20.75 | 19.86 | [16.8, 22.8] | 473.51 | 0.45 |  | Base config: trainer defaults tuned in the single-algorithm cards. |
| reinforce_c01 | lr | lr=0.001 | 29.66 | 29.29 | 26.21 | [23.2, 29.2] | 406.71 | 0.46 |  | Lower learning rate: smaller steps on high-variance Monte-Carlo gradients. |
| reinforce_c02 | lr | lr=0.01 | 8.60 | -63.76 | 5.03 | [1.0, 8.8] | 2574.81 | 0.38 |  | Higher learning rate: tests whether the policy collapses early (low entropy). |
| reinforce_c03 | gamma | gamma=0.95 | 34.74 | 19.79 | 32.49 | [29.3, 35.7] | 632.21 | 0.55 | yes | Shorter horizon: lower-variance returns, less credit for long-term effects. |
| reinforce_c04 | baseline | baseline=none | 17.34 | -0.67 | 10.01 | [6.1, 13.9] | 810.51 | 0.44 |  | No baseline: plain REINFORCE, the variance-reduction reference. |
| reinforce_c05 | baseline | baseline=moving_average | 22.11 | -0.56 | 12.60 | [9.2, 16.0] | 741.57 | 0.44 |  | Scalar moving-average baseline instead of a learned V(s). |
| reinforce_c06 | normalization | normalize_advantages=False | 29.21 | 25.10 | 23.78 | [21.0, 26.4] | 462.35 | 0.46 |  | Raw advantages: no per-batch scale normalisation. |
| reinforce_c07 | entropy | entropy_coef=0.05 | 29.60 | 29.60 | 24.86 | [21.5, 27.9] | 388.45 | 0.42 |  | Stronger entropy bonus: keeps exploring longer. |
| reinforce_c08 | episodes_per_update | episodes_per_update=2 | 19.50 | -5.96 | 14.14 | [10.4, 17.6] | 1115.24 | 0.32 |  | Fewer episodes per update: more, noisier updates. |
| reinforce_c09 | episodes_per_update | episodes_per_update=10 | 30.03 | 30.03 | 27.67 | [24.7, 30.5] | 365.55 | 0.50 |  | More episodes per update: fewer, lower-variance updates. |
