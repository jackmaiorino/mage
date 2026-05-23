# Thesis-Clean Prefix Trajectory RL

Date: 2026-05-12

## Question

Can generic terminal-prefix search be used as trajectory-level RL data instead of direct one-step KL distillation?

This keeps the thesis boundary if the teacher uses:

- terminal `WIN` only;
- legal candidate search only;
- generic branch ordering;
- no action-text/card-name filters;
- no Spy terminal milestone;
- no heuristic rewards;
- no selective MCTS.

## Code Change

`ActionCounterfactualTrainer` now supports:

```text
--export-trajectory-data-file=<path>
--import-trajectory-data-path=<path>
--trajectory-final-reward=<float>
```

Exported trajectory files store episode-shaped `TrainingData` records plus rewards, so import can enqueue them through the normal RL path rather than the MCTS/KL target path.

Because strict generic search did not produce fresh full trajectories locally, a low-cost proxy was also added:

```text
--import-flat-as-terminal-episodes=true
```

That treats existing serialized generic prefix tensors as one-step terminal-return episodes. This is not a full trajectory teacher, but it tests whether the prefix corpus is less harmful through PPO/value terminal credit than through direct KL.

Wrapper support:

```powershell
scripts/run_spy_line_search.ps1 `
  -ExportTrajectoryDataFile ... `
  -ImportTrajectoryDataPath ... `
  -ImportFlatAsTerminalEpisodes `
  -TrajectoryRlLoss
```

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
PowerShell parser check for scripts/run_spy_line_search.ps1
```

## Fresh Clone

Profile:

```text
Pauper-Spy-Combo-Value-TrajRL-20260512
```

Source: accepted `Pauper-Spy-Combo-Value`.

The clone reset optimizer state and training counters. Active accepted hashes were rechecked after the experiment and remained unchanged.

## Strict Full-Trajectory Collection

Three thesis-clean collection attempts produced zero winning trajectories:

```text
20260512_thesis_clean_trajectory_rl_cp7_unique_collect16: 0 trajectories
20260512_thesis_clean_trajectory_rl_cp1_unique_collect16: 0 trajectories
20260512_thesis_clean_trajectory_rl_replay_prior_collect: 0 trajectories
```

The last run replayed the prior successful generic-prefix collection seed/settings, with trajectory export enabled:

```text
terminal_mode=WIN
generic_branch_order=true
tactic_autopilot=false
max_prefix_depth=5
train_prefix_depth=5
max_search_nodes=15
top_k=3
random_extra=1
max_game_turns=12
scenario_timeout_sec=75
skip_pass_training=true
skip_blank_training=true
skip_mulligan_training=true
```

Result: no fresh full-trajectory data to train. This means strict generic search is not currently a reliable trajectory generator at the local budget.

## Flat Terminal-Return Proxy

Source corpus:

```text
local-training/local_pbt/generic_prefix_data/20260511_balanced_cp1_cp7_hardmatchups
```

Training run:

```text
20260512_thesis_clean_prefix_flat_terminal_rl_train124
```

Settings:

```text
import_flat_as_terminal_episodes=true
trajectory_final_reward=1.0
trajectory_rl_loss=true
MCTS_KL_LOSS_COEF=0
POLICY_LOSS_COEF=1.0
VALUE_LOSS_COEF=0.5
ENTROPY_LOSS_MULT=0.005
train_epochs=4
candidate_permutations=2
skip_pass_training=true
```

Training completed cleanly:

```text
importedTrainingExamples=124
trainPassSamples=992
train_steps=992
```

## Reduced CP7 Eval

Run:

```text
20260512_prefix_flat_terminal_rl_train124_spy_cp7_unique_eval16
```

Result:

```text
Overall: 5/64 = 7.81%
```

Per matchup:

```text
Spy mirror:        5/16 = 31.25%
Jund Wildfire:     0/16 = 0.00%
Mono Red Rally:    0/16 = 0.00%
Grixis Affinity:   0/16 = 0.00%
```

Accepted CP7 reference:

```text
108/242 = 44.63%
```

## Verdict

Rejected. Do not run CP1 or HPC.

The full trajectory generator failed to produce data at the local generic-search budget. The flat terminal-return proxy was far worse than direct KL branches and collapsed all non-mirror matchups. The prefix corpus remains useful diagnostically, but neither direct target matching nor positive-only terminal-return training is a viable policy-improvement mechanism.

Next work should move away from positive-only prefix data. Viable thesis-clean directions are either true online policy improvement with a stronger search oracle, or a generic belief/determinization implementation that improves the search oracle before training from it.

## 2026-05-13 Correction

This page's zero-trajectory collection result was invalidated by a harness
default issue: `run_spy_line_search.ps1` was defaulting to `MODEL_D_MODEL=256`
and `MODEL_NUM_LAYERS=4` while the accepted thesis-clean checkpoint uses `128/2`.

After fixing the wrapper default, corrected collection produced local winning
trajectories. The follow-up results are documented in:

```text
docs/mtgrl/experiments/2026-05-13-thesis-clean-corrected-trajectory-teacher/README.md
```
