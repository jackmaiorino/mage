# Spy Actual-Win Offline BC

date: 2026-05-04

## Question

Can the Spy safe-hand tactic prober produce actual won trajectories that are good enough for pure behavior cloning to create a CP1-viable Spy model before any RL fine-tuning?

The decision gate was intentionally harsh:

- If pure imitation on about 10k won trajectories reaches more than 70% Spy CP1 winrate, the RL stack mostly needs a better seed.
- If imitation cannot approach that after fitting the labels, the problem is upstream of RL or in the start-state distribution.

## Setup

- agent deck: `Deck - Spy Combo.dek`
- opponent pool: Spy Combo, Jund Wildfire, Mono Red Rally, Grixis Affinity
- generator: `ActionCounterfactualTrainer`
- terminal mode: `WIN`
- search: tactic autopilot, no model scoring, depth-first winning-prefix search
- collection: forced Spy safe/reachable hand pool
- reward shaping: none for RL; this is supervised imitation on terminal-winning prefixes only

Primary data run:

- run: `20260504_spy_actual_win_offline_bc_5k`
- collected winning trajectories: 5,001
- training examples: 109,525
- serialized data size: about 15.3 GB
- batches: 194

## Results

| Model | Train setting | Label score | Overall CP1 WR | Spy mirror WR | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Small | 128 dim, 2 layers, 4 epochs | 29.49% top-1 on 2,048 examples | 24/128 = 18.75% | 15/32 = 46.88% | Did not fit actual-win labels. |
| Small canary | 128 dim, 2 layers, 100 epochs on 1,382 fixed examples | 46.16% top-1 | not evaluated | not evaluated | Capacity/training insufficient. |
| Big canary | 256 dim, 4 layers, 100 epochs on same 1,382 examples | 91.90% top-1 | not evaluated | not evaluated | State/action path can fit. |
| Big full | 256 dim, 4 layers, 4 epochs | 81.13% top-1 on 10,000 examples | 22/128 = 17.19% | 15/32 = 46.88% | Fits labels but does not transfer to random-start CP1. |

Big full CP1 matchup breakdown:

- vs Spy Combo: 15/32 = 46.88%
- vs Jund Wildfire: 3/32 = 9.38%
- vs Mono Red Rally: 3/32 = 9.38%
- vs Grixis Affinity: 1/32 = 3.12%

## Replay Probe

The replay probe reran the big policy from the saved opening hands in `winning_trajectories.csv`.

- replay groups: 28
- exact prefix action match: 29/606 = 4.79%
- scenario wins: 19/28 = 67.86%

This is important: the model does not reproduce the exact found prefix, but it often finds an alternate winning line from the same reachable hands. That makes full-start CP1 failure look less like "Spy cannot execute combo actions" and more like "Spy does not reliably reach or keep combo-capable starts."

## Natural Start Probe

A natural-start London search smoke test tried to find terminal wins without the safe-hand pool.

- scenarios: 48
- action type: `LONDON_MULLIGAN`
- max game turns: 8
- max search nodes: 800
- winning trajectories found: 0

The current prober cannot discover natural-start Spy wins cheaply.

## Mulligan Repair Follow-up

Because full CP1 starts are dominated by the opening-hand decision, a terminal-only London counterfactual repair was run on top of the big full imitation profile.

Small repair:

- run: `20260504_spy_big_terminal_london_mull_cf64`
- selected samples: 122
- train passes: 976
- CP1 after repair: 23/128 = 17.97%

Scaled repair:

- run: `20260504_spy_big_terminal_london_mull_cf512`
- trained scenarios: 385
- selected samples: 1,015
- train pass samples: 101,500
- elapsed: 7,571 sec
- first-hand probe after training: 512/512 keeps, 0 mulligans

The scaled repair produced balanced terminal labels for some first prompts, but the deployed first-hand policy still kept every tested Spy opener.

## Interpretation

This experiment rules out the most pessimistic upstream explanation. The model can fit actual winning action labels when given enough capacity, and from generated reachable hands it wins about two thirds of replayed scenarios.

The failed 70% CP1 gate is therefore not conclusive evidence that the action space or state encoding is broken. The remaining blocker is the start-state layer:

- forced reachable-hand trajectories do not match full random starts;
- pure action BC does not train the first London decision enough;
- the terminal-only mulligan counterfactual pass did not change deployed first-hand decisions.

## Next Experiment

Debug the mulligan repair path as an exact supervised-learning problem.

1. Export or score the selected London counterfactual examples directly.
2. Overfit a fixed first-hand mulligan subset and verify top-1/target-prob fit.
3. Probe the deployed first-hand policy on random Spy openers after exact-fit training.
4. If exact-fit succeeds but random-hand behavior still keeps everything, the data distribution is wrong.
5. If exact-fit fails, the bug is in the mulligan training/head/decision path.

Primary artifacts:

- `local-training/local_pbt/spy_imitation_offline_bc/20260504_spy_actual_win_offline_bc_5k`
- `local-training/local_pbt/spy_line_search/20260504_spy_actual_win_big_score_10000_e4`
- `local-training/local_pbt/cp7_eval_sweeps/20260504_spy_actual_win_big_cp1_5001_e4`
- `local-training/local_pbt/spy_line_replay/20260504_big_actual_win_replay_batch0001_64`
- `local-training/local_pbt/spy_line_search/20260504_spy_natural_london_win_smoke48`
- `local-training/local_pbt/mulligan_counterfactual/20260504_spy_big_terminal_london_mull_cf512`
- `local-training/local_pbt/mulligan_probes/20260504_spy_big_after_mullcf512_probe512`
