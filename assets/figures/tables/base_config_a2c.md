### A2C: base config c00

| parameter | value |
|---|---|
| frames_per_batch | 256 |
| hidden_sizes | [128, 128] |
| activation | tanh |
| lr | 0.001 |
| gamma | 0.99 |
| max_grad_norm | 0.5 |
| gae_lambda | 0.95 |
| entropy_coef | 0.01 |
| value_coef | 0.5 |
| loss_critic_type | smooth_l1 |
| normalize_advantages | True |
