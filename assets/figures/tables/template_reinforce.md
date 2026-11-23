### REINFORCE: Phase 1 runs in the report-template layout (200k frames, seed 0; * = the value changed from c00; Mean Reward = mean return on 50 selection seeds)

| Run | Learning Rate | Gamma | Baseline | Normalize Advantages | Entropy Coef | Episodes per Update | Mean Reward |
|---|---|---|---|---|---|---|---|
| c00 | 0.003 | 0.99 | value | yes | 0.01 | 5 | 19.9 |
| c01 | 0.001* | 0.99 | value | yes | 0.01 | 5 | 26.2 |
| c02 | 0.01* | 0.99 | value | yes | 0.01 | 5 | 5.0 |
| c03 (picked) | 0.003 | 0.95* | value | yes | 0.01 | 5 | 32.5 |
| c04 | 0.003 | 0.99 | none* | yes | 0.01 | 5 | 10.0 |
| c05 | 0.003 | 0.99 | moving_average* | yes | 0.01 | 5 | 12.6 |
| c06 | 0.003 | 0.99 | value | no* | 0.01 | 5 | 23.8 |
| c07 | 0.003 | 0.99 | value | yes | 0.05* | 5 | 24.9 |
| c08 | 0.003 | 0.99 | value | yes | 0.01 | 2* | 14.1 |
| c09 | 0.003 | 0.99 | value | yes | 0.01 | 10* | 27.7 |
