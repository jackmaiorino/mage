# Spy Imitation Trajectory BC

date: 2026-05-03

## Question

Can the Spy safe-hand tactic prober generate enough winning prefixes that pure behavior cloning creates a Spy model capable of beating CP1 before any RL fine-tuning?

## Setup

- profile: `Pauper-Spy-Combo-Imitation-20260502`
- agent deck: `Deck - Spy Combo.dek`
- opponent pool: Spy Combo, Jund Wildfire, Mono Red Rally, Grixis Affinity
- trainer: `ActionCounterfactualTrainer` in `SPY_COMBO_MILESTONE_ONLY`
- generation: tactic autopilot, no search-model scoring, random non-target decisions
- reward: terminal/milestone only; no intermittent rewards
- main continuation: `20260502_spy_imitation_trajectory_24h_cont7_b48w48_t120`

## Throughput

The best local setting was 48 scenarios / 48 workers with 120 second scenario timeout. During active search this saturated CPU at roughly 96-100% and produced about 700-800 won trajectories/hour.

## Results

Approximate cumulative won trajectories across all continuations: 10,017.

| Checkpoint | Eval games | Overall CP1 WR | Spy mirror WR | Notes |
| --- | ---: | ---: | ---: | --- |
| ~1.3k | 128 | 20/128 = 15.6% | 15/32 = 46.9% | manual eval before high-throughput continuation |
| ~2.5k | 64 | 2/64 = 3.1% | 2/16 = 12.5% | first cont7 milestone |
| ~5k | 64 | 14/64 = 21.9% | 12/16 = 75.0% | clear transient executable Spy mirror behavior |
| ~10k | 128 | 2/128 = 1.6% | 2/32 = 6.3% | final checkpoint regressed badly |

Final matchup summary:

- vs Spy Combo: 2/32
- vs Jund Wildfire: 0/32
- vs Mono Red Rally: 0/32
- vs Grixis Affinity: 0/32

## Interpretation

This run did not pass the intended full CP1 gate. The 5k checkpoint is still important: imitation alone can temporarily produce executable Spy combo behavior in the mirror. The final collapse means the online form of this experiment is not a clean test of the original hypothesis, because each small batch trained and saved immediately instead of training once on an aggregate dataset.

Two likely confounds remain:

- Online BC can forget earlier trajectory diversity as later batches overwrite the policy.
- The generated trajectories begin from forced safe/reachable hands, while full CP1 eval still depends on the untrained or weak mulligan policy.

## Next Experiment

Run the cleaner version: collect serialized winning trajectory tensors first, then train a fresh model once on the aggregate dataset. This better matches the original experiment: generate trajectories, behavior-clone on the set, then evaluate before RL.

Key code added for that follow-up:

- `--collect-only`
- `--export-training-data-file`
- `--import-training-data-path`

Primary artifacts:

- `local-training/local_pbt/spy_imitation_trajectory/20260502_spy_imitation_trajectory_24h_cont7_b48w48_t120`
- `local-training/local_pbt/cp7_eval_sweeps/20260502_spy_imitation_trajectory_24h_cont7_b48w48_t120_cp1_1046`
- `local-training/local_pbt/cp7_eval_sweeps/20260502_spy_imitation_trajectory_24h_cont7_b48w48_t120_cp1_3560`
- `local-training/local_pbt/cp7_eval_sweeps/20260502_spy_imitation_trajectory_24h_cont7_b48w48_t120_cp1_8556`
