# Thesis-Clean Baseline-Alt Branch Trajectory Policy

Date: 2026-05-13

## Question

Can baseline-losing-alternative branch rollouts be used as coherent terminal-return policy data without explicitly teaching Spy lines?

This is thesis-clean because the collector uses:

- terminal `WIN` only;
- generic legal branch candidates;
- no action-text/card-name filters;
- no `SPY_LANDLESS_COMBO_WIN`;
- no heuristic step rewards;
- no selective MCTS keywords;
- no Spy candidate facts.

## Harness Change

Added default-off branch trajectory diagnostics and filtering:

- `--branch-trajectory-require-training-example` only exports branch trajectories from decision points that also produced a baseline-losing-alternative label.
- `action_branch_samples.csv` now includes `trajectory_raw_records`, `trajectory_kept_records`, and `trajectory_drop_reason`.

The first console summary was misleading: `trajectories=0` reports winning-prefix trajectories, not serialized branch trajectory episodes. The `.ser` files were nonempty.

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

## Collection

Main collection:

```text
20260513_baselinealt_branchtraj_policy_collect32
```

Settings:

```text
scenarios=32
opponent=cp7
max_game_turns=12
max_decision_depth=10
top_k=4
random_extra=2
baseline_losing_alternative_only=true
branch_trajectory_mode=true
branch_trajectory_require_training_example=true
generic_branch_order=true
skip_pass_training=true
skip_blank_training=true
skip_mulligan_training=true
terminal_mode=WIN
```

Output:

```text
candidate_examples=16
serialized file=local-training/local_pbt/generic_trajectory_data/20260513_baselinealt_branchtraj_policy_collect32.ser
file size=189,572,969 bytes
```

Debug validation:

```text
20260513_baselinealt_branchtraj_policy_debug8
candidate_examples=6
serialized file size=84,026,641 bytes
trajectory drop reasons:
  blank: 117
  target_not_forced: 22
```

## Low-Dose Policy Import

Profile clone:

```text
Pauper-Spy-Combo-Value-BranchTrajPolicy-20260513
```

Source checkpoint:

```text
Pauper-Spy-Combo-Value model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

Import run:

```text
20260513_baselinealt_branchtraj_policy_import_e1_512
```

Settings:

```text
train_epochs=1
max_train_examples=512
imported_episodes=17
imported_steps=527
policy_loss_coef=0.10
value_loss_coef=0.25
mcts_kl_loss_coef=0
use_mc_returns=true
reference_policy_kl_coef=1.0
reference_model=Pauper-Spy-Combo-Value/model_latest.pt
load_optimizer_state=0
reset_training_state_on_load=1
```

Resulting checkpoint:

```text
model_latest.pt sha256=437DF09ADB3F18EFCA038D16EE596B54FDEE214E10AE00946B7174258BF869BC
```

## Reduced CP7 Gate

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_baselinealt_branchtraj_policy_cp7_g4
```

Result:

```text
Overall: 10/16 = 62.50%
```

Per matchup:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 4 | 4 | 100.00% |
| Jund Wildfire | 2 | 4 | 50.00% |
| Mono Red Rally | 1 | 4 | 25.00% |
| Grixis Affinity | 3 | 4 | 75.00% |

This clears the reduced local gate and is the first baseline-alt trajectory policy result that improves the weak Affinity screen rather than collapsing it.

## Next Gate

Expanded local CP7 run started:

```text
20260513_baselinealt_branchtraj_policy_cp7_g16
games_per_matchup=16
games_per_job=2
parallel=2
```

Decision rule: promote this line to broader CP1/CP3/CP7 evaluation only if the expanded CP7 result remains at or above the accepted anchor without reintroducing an Affinity regression.

## Expanded CP7 Gate

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_baselinealt_branchtraj_policy_cp7_g16
```

Result:

```text
Overall: 29/64 = 45.31%
```

Per matchup:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 7 | 16 | 43.75% |
| Mono Red Rally | 7 | 16 | 43.75% |
| Grixis Affinity | 2 | 16 | 12.50% |

Accepted CP7 reference:

```text
Overall: 108/242 = 44.63%
Grixis Affinity: 13/62 = 20.97%
Mono Red Rally: 16/64 = 25.00%
```

## Verdict

Rejected for promotion and HPC. The aggregate is barely above the accepted CP7 anchor, but the candidate reintroduces the Affinity regression. The useful signal is narrower: baseline-alt branch trajectory policy improved Rally and mirror play, but the collection data was badly underweighted toward Affinity:

```text
selected baseline-alt examples by opponent:
Mono Red Rally: 9
Spy mirror: 4
Jund Wildfire: 2
Grixis Affinity: 1
```

Next experiment: repeat this same thesis-clean mechanism with Affinity-pressure collection from the accepted checkpoint, then gate on Affinity before running another full CP7 screen.

## Affinity-Pressure Repeat

Profile clone:

```text
Pauper-Spy-Combo-Value-BranchTrajPolicyAffinity-20260513
```

Collection:

```text
20260513_baselinealt_branchtraj_policy_affinity_collect128_stop16
opponent=cp7 Grixis Affinity only
completed_scenarios=70
trained_scenarios=11
candidate_examples=16
serialized file=local-training/local_pbt/generic_trajectory_data/20260513_baselinealt_branchtraj_policy_affinity_collect128_stop16.ser
file size=143,403,754 bytes
```

Import:

```text
20260513_baselinealt_branchtraj_policy_affinity_import_e1_512
imported_episodes=24
imported_steps=523
policy_loss_coef=0.10
value_loss_coef=0.25
reference_policy_kl_coef=1.0
sha256=34CBC4DECB7C35BA460B620109F9BE640F36E70A67FFA1E3A2D8AF29BDD92F19
```

Local disk was full during the first atomic save attempt; after moving old generated artifacts to `D:\mage_artifact_archive\20260513_free_space`, the checkpoint loaded successfully with `torch.load`.

Affinity-only CP7 gate:

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_baselinealt_branchtraj_policy_affinity_cp7_affinity_g16
Grixis Affinity: 3/16 = 18.75%
```

Verdict: rejected. The targeted collection did not clear the accepted Affinity anchor (`13/62 = 20.97%`) and remains too weak for broader CP7 evaluation or HPC promotion. This closes the baseline-alt branch-trajectory policy line for now: it can find terminal-winning alternatives, but low-dose imitation from those branch trajectories does not produce a robust policy improvement.

## Checkpoint Interpolation Follow-Up

Question:

Can the branch-trajectory policy checkpoint's Rally/mirror gains be softened into the accepted policy without keeping its Affinity regression?

Profiles:

```text
Pauper-Spy-Combo-Value-BranchTrajSoup025-20260513
Pauper-Spy-Combo-Value-BranchTrajSoup050-20260513
```

Sources:

```text
base: Pauper-Spy-Combo-Value/model_latest.pt
branch source: Pauper-Spy-Combo-Value-BranchTrajPolicy-20260513/model_latest.pt
```

CP7 Rally/Affinity screen:

```text
20260513_branchtraj_soup_rally_affinity_cp7_eval8
```

| Soup | Rally | Affinity |
| --- | ---: | ---: |
| 25% branch | 3/8 | 1/8 |
| 50% branch | 1/8 | 1/8 |

Verdict: rejected. The 25% soup kept some Rally signal but immediately failed Affinity, and the 50% soup failed both pressure matchups. Do not continue branch-trajectory checkpoint interpolation or submit this line to HPC.

## Policy-Miss Trajectory Import Follow-Up

Question:

Can the same branch-trajectory corpus work if trajectory import trains only on
generic policy-miss steps instead of every forced-trajectory step?

Patch:

- `ActionCounterfactualTrainer.filterTrajectoryEpisode` now applies
  `passesTargetMargin` and `passesPolicyMissFilter`, matching the flat import
  path.
- This is default-off through the existing `-PolicyMissOnly` flag.

Profile:

```text
Pauper-Spy-Combo-Value-BranchTrajPolicyMiss-20260513
```

Import:

```text
20260513_branchtraj_policy_miss_import_e1_512
source=20260513_baselinealt_branchtraj_policy_collect32.ser
episodes_kept=21
steps_kept=515
train_pass_samples=515
sha256=4833A2350F418119CD46155F5E54C67B696ACCC216A0A078D39C3A7D1FC1DA5C
```

Small CP7 gate:

```text
20260513_branchtraj_policy_miss_cp7_g4
overall: 10/16 = 62.50%
Spy mirror: 4/4
Jund Wildfire: 2/4
Mono Red Rally: 2/4
Grixis Affinity: 2/4
```

Pressure expansion:

```text
20260513_branchtraj_policy_miss_pressure_cp7_g16
Mono Red Rally: 7/16 = 43.75%
Grixis Affinity: 1/16 = 6.25%
combined pressure: 8/32 = 25.00%
```

Verdict: rejected. The policy-miss trajectory filter preserves the same Rally
lift seen in prior branch-combination experiments, but Affinity collapses even
harder than accepted. This closes the current branch-trajectory family as a
local promotion surface.

Affinity-only control:

```text
profile: Pauper-Spy-Combo-Value-BranchTrajPolicyMissAffinity-20260513
import: 20260513_branchtraj_policy_miss_affinity_import_e1_512
source: 20260513_baselinealt_branchtraj_policy_affinity_collect128_stop16.ser
episodes_kept: 25
steps_kept: 515
```

Eval:

```text
20260513_branchtraj_policy_miss_affinity_cp7_g16_copydb
Grixis Affinity: 1/16 = 6.25%
```

This directly rejects the hypothesis that the previous targeted Affinity import
failed only because solved/noisy downstream trajectory steps were included.

Operational fix: the first retry exposed that DB hardlinking in
`scripts/run_cp7_eval_sweep.py` is unsafe for parallel H2 eval jobs because
separate per-job paths can still point at the same physical database file and
lock each other. DB preparation now defaults to real copies; hardlinking is
available only with `CP7_EVAL_DB_HARDLINK=1`.
