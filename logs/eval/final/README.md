# Final evaluation and generalization (Trello card 19)

The 4 baselines and the 20 Phase 2 agents (best checkpoint `model.pt` of `dqn_c08`, `ppo_c04`, `a2c_c03` and
`reinforce_c03`, seeds 0-4) were scored on the 50 held-out test seeds (`EVAL_SEEDS` 10000-10049). No training run
and no model selection ever saw these seeds. All policies play the same episodes (common random numbers). The RL
agents act in their training eval mode: DQN greedy, policy-gradient agents sampled with a per-episode seed.

Each scenario has its own folder with `episodes.csv` (one row per policy x seed), `kpi_summary.csv` (mean, std,
95% bootstrap CI) and `paired_vs_urgency_first.csv`. `<scenario>.txt` is the console output. To reproduce
(policy list as in the file headers):

    uv run main.py evaluate --policies random,nearest_job_first,urgency_first,rolling_horizon,dqn_s0=models/dqn/dqn_c08_p2_s0/model.pt,... --scenarios <scenario> --out logs/eval/final/<scenario>

The agents were trained only on `base`. The other three scenarios are distribution shifts they never saw.

## Mean return on the 50 test seeds

RL rows pool 5 seeds x 50 episodes (95% bootstrap CI over episodes). The spread between seeds (std of the
per-seed means) is in brackets.

| Policy | base | high_demand | high_breakdown | heavy_traffic |
|---|---|---|---|---|
| random | 1.0 | -5.9 | -8.6 | -12.3 |
| nearest_job_first | 52.1 | 69.5 | 38.7 | 44.2 |
| urgency_first | 57.5 | 67.5 | 42.1 | 48.2 |
| rolling_horizon | **59.9** [57.9, 61.7] | **70.6** | **42.3** | **51.9** |
| DQN | 54.1 [52.3, 55.9] (3.4) | 34.6 (2.4) | 32.2 (3.5) | 39.6 (2.2) |
| PPO | 51.7 [50.4, 53.0] (1.3) | 29.9 (7.9) | 35.1 (1.9) | 39.9 (1.2) |
| A2C | 48.6 [46.8, 50.3] (2.1) | 48.6 (10.8) | 33.8 (1.4) | 37.6 (1.1) |
| REINFORCE | 34.2 [32.3, 35.9] (5.4) | 30.1 (3.9) | 18.3 (6.0) | 21.3 (6.5) |

## Findings

1. **Validation to test seeds (same scenario): returns drop 3-6 points.** Mean drop between the best validation
   return and the test return: DQN 6.2, PPO 5.1, REINFORCE 4.9, A2C 3.3. Part of the drop is selection bias: the
   best checkpoint was picked on the same 10 validation seeds it is scored on, so the validation number is
   optimistic. The algorithm ranking is unchanged: DQN > PPO > A2C > REINFORCE.
2. **On `base`, no RL agent beats the heuristics.** The best agent, DQN (54.1), is below urgency_first (57.5)
   and rolling_horizon (59.9). In the paired per-seed test against urgency_first, 17 of 20 agents are
   significantly worse and 3 of the 5 DQN seeds are not significantly different. None is significantly better.
3. **Why RL loses: it collects MORE waste but pays at the end of the shift and in overflow.** On `base` the agents
   collect 19.0-20.2 t against 19.2-19.3 t for urgency_first and rolling_horizon. But they leave 0.7-2.7 trucks
   away from the depot and 1.3-3.1 t on board (baselines: 0.2-0.4 trucks, 0.3-0.5 t), which costs 6.6-17.7 return
   points in terminal penalties (baselines about 1.5-3). Bin-minutes in overflow are also 4-11x higher (100-281
   against 25-27). This confirms the Phase 1 finding.
4. **Distribution shift hurts RL more than the baselines.**
   - `high_demand` is the biggest failure. The heuristics GAIN 10-17 points (more waste to collect), while DQN
     loses 19.5 and PPO 21.8 against `base`. Overflow explodes (about 1000-1100 bin-minutes) and terminal
     penalties reach about -20. A2C is the exception (-0.1 change, 48.6), though its seeds vary widely (std 10.8).
   - `high_breakdown` and `heavy_traffic` cost every policy 8-22 points, RL and baselines similarly. DQN, PPO
     and A2C stay about 7-11 points behind urgency_first. A2C and PPO vary least across seeds in these two (std 1.0-1.9).
5. **REINFORCE is last everywhere,** with the largest variance between seeds, consistent with its high-variance
   Monte-Carlo gradient.

**For the report:** the RL agents learned to collect, but not to plan the end of the shift or to cope with demand
they never trained on. The heuristics encode both directly (urgency ordering, return-to-depot), and the
rolling-horizon planner looks ahead explicitly.
