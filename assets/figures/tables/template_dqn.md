### DQN: Phase 1 runs in the report-template layout (200k frames, seed 0; * = the value changed from c00; Mean Reward = mean return on 50 selection seeds)

| Run | Learning Rate | Gamma | Replay Buffer Size | Batch Size | Exploration Strategy | Learning Starts | Target Update | Mean Reward |
|---|---|---|---|---|---|---|---|---|
| c00 | 0.0005 | 0.99 | 50000 | 128 | eps-greedy 1->0.05 over 60k frames | 5000 | soft, tau=0.005 | 29.3 |
| c01 | 0.0001* | 0.99 | 50000 | 128 | eps-greedy 1->0.05 over 60k frames | 5000 | soft, tau=0.005 | -10.5 |
| c02 | 0.001* | 0.99 | 50000 | 128 | eps-greedy 1->0.05 over 60k frames | 5000 | soft, tau=0.005 | 30.0 |
| c03 | 0.0005 | 0.95* | 50000 | 128 | eps-greedy 1->0.05 over 60k frames | 5000 | soft, tau=0.005 | 35.1 |
| c04 | 0.0005 | 0.99 | 50000 | 128 | eps-greedy 1->0.05 over 20k frames* | 5000 | soft, tau=0.005 | 37.6 |
| c05 | 0.0005 | 0.99 | 50000 | 128 | eps-greedy 1->0.05 over 120k frames* | 5000 | soft, tau=0.005 | 44.3 |
| c06 | 0.0005 | 0.99 | 10000* | 128 | eps-greedy 1->0.05 over 60k frames | 5000 | soft, tau=0.005 | 29.9 |
| c07 | 0.0005 | 0.99 | 50000 | 32* | eps-greedy 1->0.05 over 60k frames | 5000 | soft, tau=0.005 | -13.2 |
| c08 (picked) | 0.0005 | 0.99 | 50000 | 128 | eps-greedy 1->0.05 over 60k frames | 20000* | soft, tau=0.005 | 44.4 |
| c09 | 0.0005 | 0.99 | 50000 | 128 | eps-greedy 1->0.05 over 60k frames | 5000 | hard / 1k steps* | 36.6 |
