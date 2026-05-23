# Spy Terminal Branch Counterfactual-Q Handoff

Date: 2026-05-08

## Context

This log captures the work after the zone-count terminal RL experiment and before the session was closed. The goal was to keep the project thesis intact: terminal rewards only, no deck-specific heuristic rewards, but use terminal branch outcomes to improve Spy Combo timing.

Parent checkpoint for these experiments:

- `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Parent full CP1 result from earlier log: 38/128
- Parent logged failure mode: no pass-over-land, but broken mulligans, premature Spy, and premature Dread Return.

No training/eval/collection processes were left running at handoff time.

## Code Changes

Changed files relevant to this experiment:

- `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/ActionCounterfactualTrainer.java`
  - Added strict avoid-losing branch labels.
  - Added mask-baseline-only labels for "baseline action loses, some sibling branch wins".
  - Existing branch-target `mctsVisitTargets` are now the carrier for terminal branch distributions.
- `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/ComputerPlayerRL.java`
  - Added eval-only `RL_VALUE_ACTION_GATE_*` diagnostic gate, default off.
  - This was only used to test whether current-state V could veto low-value Spy/Dread actions.
- `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py`
  - Added `CANDIDATE_Q_FROM_MCTS_TARGETS=1`.
  - When enabled with `CANDIDATE_Q_LOSS_COEF>0`, candidate-Q trains from branch target tensors rather than selected-action-only returns.
- `scripts/run_action_counterfactual.ps1`
  - Passes strict/mask avoid-losing flags through to Java.
- `local-training/local_pbt/action_counterfactual/run_20260508_spy_actioncf_collect_onnx.ps1`
  - Fixed regex argument passing by building an argument array.
  - Added `-AvoidLosingActionTextRegex "__NONE__"` support.
- `local-training/local_pbt/action_counterfactual/run_20260508_spy_softbranch_q_collect64.ps1`
  - Dedicated literal-regex runner for `Balustrade Spy|Dread Return` soft branch-Q collection.
- Eval registries added:
  - `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_actioncf_maskbase_eval_registry.json`
  - `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_cfq_eval_registry.json`
  - `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_cfq300_eval_registry.json`

Compile check run after code edits:

```powershell
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

Result: successful.

## Mask-Baseline Strict Negative

Purpose: avoid strict hard suppression of Spy by preserving baseline distribution over all non-losing alternatives.

Collection:

- Run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_actioncf_maskbase_collect_s12000_w16_stop16`
- Completed: 16 examples after 972 scenarios
- Note: because of the older PowerShell quoting bug, this collection only used `Balustrade`, not full `Balustrade Spy|Dread Return`.

Training:

- Profile: `Pauper-Spy-Combo-ZoneCountsMaskBase-20260508`
- Import run: `20260508_spy_zone_counts_actioncf_maskbase_import16_e8p2_direct05`
- 16 examples, 256 train passes.
- Post-score on those examples: top1 14/16, target-set top1 15/16.

Full CP1 eval:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_actioncf_maskbase_cp1_eval32`
- Result: 41/128
- Matchups:
  - Affinity: 6/32
  - Wildfire: 12/32
  - Rally: 3/32
  - Spy mirror: 20/32

Logged CP1:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_actioncf_maskbase_cp1_logged8`
- Result: 11/32
- Action health output: `local-training/analysis/value_head_20260508_maskbase_logged8/action_health.csv`
- Key failures:
  - Mulligan: 32/32 keep, 30/32 kept 0-1 true lands.
  - Spy: 26 casts, 25/26 with hidden true lands.
  - Dread Return: 18 flashbacks, 17/18 premature/not combo-ready.

Value-head diagnostic:

- Output dir: `local-training/analysis/value_head_20260508_maskbase_logged8`
- Decision-level AUC: 0.647
- Nontrivial decision AUC: 0.688
- Critical-combo AUC: 0.839
- Per-game mean value AUC: 0.918

Conclusion: value signal is present, but policy does not use it well enough.

## Current-State Value Gate Probe

Purpose: test whether current-state V can be used directly to veto low-value Spy/Dread actions.

Eval env:

```powershell
$env:RL_VALUE_ACTION_GATE_ENABLE='1'
$env:RL_VALUE_ACTION_GATE_MIN='0.10'
$env:RL_VALUE_ACTION_GATE_KEYWORDS='Balustrade Spy,Dread Return'
```

Full CP1 eval:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_maskbase_valuegate010_cp1_eval32`
- Result: 33/128
- Matchups:
  - Affinity: 5/32
  - Wildfire: 10/32
  - Rally: 1/32
  - Spy mirror: 17/32

Logged CP1:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_maskbase_valuegate010_cp1_logged4`
- Result: 1/16
- Value gate fired 85 times in 16 games.
- Action health:
  - Spy casts dropped to 7, but 6/7 still had hidden true lands.
  - Dread flashbacks dropped to 4, but 4/4 were still premature.

Conclusion: current-state V is too blunt. It suppresses necessary low-value combo attempts and does not solve timing. Do not promote this gate; leave it default-off as a diagnostic only.

## Counterfactual-Q Plumbing

The earlier candidate-Q experiment was selected-action-only and did not have counterfactual labels. This session added a mode where candidate-Q trains from terminal branch distributions:

```powershell
$env:CANDIDATE_Q_ONLY='1'
$env:CANDIDATE_Q_LOSS_COEF='1.0'
$env:CANDIDATE_Q_FROM_MCTS_TARGETS='1'
$env:CANDIDATE_Q_BLEND='0.0'
```

Eval uses:

```powershell
$env:CANDIDATE_Q_BLEND='0.5' # or 1.0
```

This is still terminal-only: labels come from sibling branch terminal outcomes, not step rewards or Spy-specific rules.

## CFQ 64-Example Dataset

Collection:

- Run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_actioncf_softbranch_q_collect64`
- Completed: 64 examples after 274 scenarios
- Wall clock: 1,107.5 sec
- Regex: `Balustrade Spy|Dread Return`
- Action types: `ACTIVATE_ABILITY_OR_SPELL`

Training:

- Profile: `Pauper-Spy-Combo-ZoneCountsCFQ-20260508`
- Fresh clone from parent `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Import run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_cfq_import64_e16p2`
- 64 examples, 2,048 Q-only train passes

Reduced eval, blend 1.0:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_cfq_blend1_cp1_eval16`
- Result: 29/64
- Matchups:
  - Affinity: 5/16
  - Wildfire: 9/16
  - Rally: 2/16
  - Spy mirror: 13/16

Full eval, blend 1.0:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_cfq_blend1_cp1_eval32`
- Result: 35/128
- Matchups:
  - Affinity: 6/32
  - Wildfire: 13/32
  - Rally: 3/32
  - Spy mirror: 13/32

Reduced eval, blend 0.5:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_cfq_blend05_cp1_eval16`
- Result: 26/64
- Matchups:
  - Affinity: 6/16
  - Wildfire: 5/16
  - Rally: 3/16
  - Spy mirror: 12/16

Conclusion: plumbing works, but 64 examples were too noisy. Reduced blend-1 looked promising but did not hold up in full eval.

## CFQ 300-Example Dataset

Collection:

- Run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_actioncf_softbranch_q_collect300`
- Completed: 300 examples after 1,258 scenarios
- Wall clock: 4,862.1 sec
- Regex: `Balustrade Spy|Dread Return`
- CPU was saturated near 99% for most of collection.
- A few activation failures occurred but the run completed.

Best-label distribution highlights:

- Balustrade Spy cast: 48
- Lead the Stampede: 31
- Winding Way: 27
- Forest mana: 24
- Overgrown Battlement cast: 17
- Swamp mana: 16
- Pass: 15
- Dread Return flashback: 10
- Dread Return cast: 2

Training:

- Profile: `Pauper-Spy-Combo-ZoneCountsCFQ300-20260508`
- Fresh clone from parent `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Import run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_cfq300_import300_e8p2`
- 300 examples, 4,800 Q-only train passes

Reduced eval, blend 0.5:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_cfq300_blend05_cp1_eval16`
- Result: 20/64
- Matchups:
  - Affinity: 2/16
  - Wildfire: 7/16
  - Rally: 2/16
  - Spy mirror: 9/16

Conclusion: the larger Q fit did not improve CP1. It hurt Affinity/Rally and did not rescue mirror enough.

## Current Standing

Best recent full CP1 results:

- Mixed avoid-loss import: 48/128, but action health still bad.
- Mask-baseline import: 41/128, action health still bad.
- Zone-count parent: 38/128.
- CFQ64 blend 1.0 full: 35/128.
- Current-state value gate: 33/128.
- CFQ300 blend 0.5 reduced: 20/64.

Main technical finding:

- The state/value representation is not the primary blocker anymore. Logged value AUC is strong, especially on critical combo states.
- Direct current-state value gating is not enough.
- Candidate-Q from terminal branch distributions is now implemented, but the current datasets/loss/blend did not improve play.

## Branch-Return Target Follow-Up

After this handoff, the next objective change was implemented:

- `ActionCounterfactualTrainer` now supports `--branch-return-targets`.
- In that mode, branch targets are explicit signed terminal returns: winning forced candidates get `+1`, losing forced candidates get `-1`, and unbranched candidates are ignored by the signed-Q loss.
- Python training now supports `CANDIDATE_Q_MCTS_SIGNED_TARGETS=1`, which interprets target tensors as signed Q targets instead of probability mass and disables accidental KL/direct-BC use of those signed rows.
- `scripts/run_action_counterfactual.ps1` exposes `-BranchReturnTargets` / `-CandidateQBranchReturns` plus candidate-Q env switches.

Verification:

```powershell
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
python -m py_compile Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py
```

Smoke collection:

- Run: `local-training/local_pbt/action_counterfactual/20260508_branch_returns_smoke_broad`
- Mode: collect-only, ONNX CPU, `-BranchReturnTargets`
- Result: 2 selected examples after 1 scenario.
- `action_training_samples.csv` confirmed one row with both `-1` and `+1` targets and one all-loss row with two `-1` targets.

## Branch-Return Q128 Cycle

Collection:

- Run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_branchret_q_collect128`
- Parent: `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Mode: ONNX CPU, `Workers=16`, `AiThreads=16`, `-BranchReturnTargets`
- Regex: `Balustrade Spy|Dread Return`
- Completed: 128 examples after 509 scenarios.
- Wall clock: 2,017.3 sec.
- Target distribution:
  - Rows with any positive target: 109/128.
  - Rows with any negative target: 73/128.
  - Mixed positive/negative rows: 54/128.
  - All-loss negative rows: 19/128.
  - Mean observed branch targets per row: 2.73.

Training:

- Profile: `Pauper-Spy-Combo-ZoneCountsBranchQ128-20260508`
- Fresh clone from `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Import run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_branchq128_import_e8p2_signed`
- 128 examples, 2,048 Q-only train passes.
- Training env: `CANDIDATE_Q_ONLY=1`, `CANDIDATE_Q_LOSS_COEF=1.0`, `CANDIDATE_Q_MCTS_SIGNED_TARGETS=1`, eval blend off during import.

Reduced CP1 eval:

- Blend 0.1: `20260508_spy_zone_counts_branchq128_blend01_cp1_eval16`, 22/64.
  - Affinity: 4/16
  - Wildfire: 6/16
  - Rally: 0/16
  - Spy mirror: 12/16
- Blend 0.25: `20260508_spy_zone_counts_branchq128_blend025_cp1_eval16`, 22/64.
  - Affinity: 5/16
  - Wildfire: 7/16
  - Rally: 2/16
  - Spy mirror: 8/16
- Blend 0.5: `20260508_spy_zone_counts_branchq128_blend05_cp1_eval16`, 18/64.
  - Affinity: 3/16
  - Wildfire: 6/16
  - Rally: 0/16
  - Spy mirror: 9/16

Conclusion: Q128 is not promotable. Signed branch-return plumbing works, and the dataset now includes negative labels, but the 128-row fit did not clear the reduced gate. Next cycle should test whether scale changes the result before abandoning eval-time Q blending.

## Branch-Return Q512 Cycle

Collection:

- Run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_branchret_q_collect512`
- Parent: `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Mode: ONNX CPU, `Workers=16`, `AiThreads=16`, `-BranchReturnTargets`
- Regex: `Balustrade Spy|Dread Return`
- Completed: 512 examples after 1,816 scenarios.
- Wall clock: 7,383.7 sec.
- Target distribution:
  - Rows with any positive target: 457/512.
  - Rows with any negative target: 296/512.
  - Mixed positive/negative rows: 241/512.
  - All-loss negative rows: 55/512.
  - Mean observed branch targets per row: 2.83.

Training:

- Profile: `Pauper-Spy-Combo-ZoneCountsBranchQ512-20260508`
- Fresh clone from `Pauper-Spy-Combo-FastT5ZoneCountsRL-20260508`
- Import run: `local-training/local_pbt/action_counterfactual/20260508_spy_zone_counts_branchq512_import_e8p2_signed`
- 512 examples, 8,192 Q-only train passes.

Reduced CP1 eval:

- Blend 0.1: `20260508_spy_zone_counts_branchq512_blend01_cp1_eval16`, 21/64.
  - Affinity: 2/16
  - Wildfire: 10/16
  - Rally: 0/16
  - Spy mirror: 9/16
- Blend 0.25: `20260508_spy_zone_counts_branchq512_blend025_cp1_eval16`, 23/64.
  - Affinity: 2/16
  - Wildfire: 6/16
  - Rally: 2/16
  - Spy mirror: 13/16
- Blend 0.5: `20260508_spy_zone_counts_branchq512_blend05_cp1_eval16`, 20/64.
  - Affinity: 2/16
  - Wildfire: 7/16
  - Rally: 2/16
  - Spy mirror: 9/16

Conclusion: Scaling signed branch-return Q to 512 rows did not rescue eval-time Q blending. This path is not promotable as a reranker. The next experiment should move Q loss into terminal self-play as an auxiliary with policy/value learning active, not as a frozen Q-only fit plus eval-time blend.

## Hard Spy Timing Mask Diagnostic

Implementation:

- Added a default-off eval-only mask in `ComputerPlayerRL`.
- Env:
  - `RL_SPY_TIMING_MASK_ENABLE=1`
  - `RL_SPY_TIMING_MASK_BLOCK_DREAD=0|1`
  - `RL_SPY_TIMING_MASK_LOG=1`
- The Spy mask replaces a selected `Balustrade Spy` cast when the real Java library still contains lands.
- The optional Dread mask replaces a selected `Dread Return` cast/flashback unless `Lotleth Giant` is in the graveyard and the player controls at least three creatures.

Compilation:

- `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`

### Spy-only mask

Run:

- `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_maskbase_spytiming_mask_cp1_logged8`
- Profile: `Pauper-Spy-Combo-ZoneCountsMaskBase-20260508`
- CP1, 8 games per matchup, logged.

Result:

- Total: 11/32
- Affinity: 2/8
- Wildfire: 2/8
- Rally: 1/8
- Spy mirror: 6/8

Action health:

- Mask fired 92 times, all `spy_with_lands`.
- Selected Spy casts dropped to 0.
- There were still 24 no-hidden-land Spy opportunities by the log estimator, but the policy selected none of them.
- Dread Return remained unhealthy: 12 flashbacks, 11/12 not combo-ready, 9/12 before `Lotleth Giant` was in the graveyard.
- Mulligan remained unchanged: 32/32 keeps, 31/32 kept 0-1 true-land hands.

### Spy+Dread timing mask

Run:

- `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_maskbase_spydread_timing_mask_cp1_logged8`
- Profile: `Pauper-Spy-Combo-ZoneCountsMaskBase-20260508`
- CP1, 8 games per matchup, logged.

Result:

- Total: 7/32
- Affinity: 1/8
- Wildfire: 1/8
- Rally: 0/8
- Spy mirror: 5/8

Action health:

- Mask fired 291 times: 187 `spy_with_lands`, 104 `dread_not_ready`.
- Dread Return flashbacks dropped to 3, but all 3 were still not combo-ready.
- Selected Spy casts stayed effectively absent: 1 selected Spy cast, with hidden lands estimated by the log analyzer.
- Mulligan remained unchanged: 32/32 keeps, 30/32 kept 0-1 true-land hands.

Conclusion:

Hard timing masks do not produce a winning agent from the current policy. The mask removes the premature combo actions, but the policy does not reliably choose the later safe combo windows. This answers the diagnostic: the rest of the model is not good enough such that fixing only Spy/Dread timing yields a strong agent.

## Zone-Count Q-Auxiliary Self-Play

Purpose:

- Test candidate-Q as a joint auxiliary during terminal RL, instead of a frozen eval-time reranker.
- Start from the best recent zone-count maskbase checkpoint and leave Q blend off at eval time.

Implementation:

- Registry: `local-training/local_pbt/value_rl/20260508_spy_zone_counts_qaux_registry.json`
- Launcher: `local-training/local_pbt/value_rl/run_20260508_spy_zone_counts_qaux.ps1`
- Source: `Pauper-Spy-Combo-ZoneCountsMaskBase-20260508`
- Profile: `Pauper-Spy-Combo-ZoneCountsQAuxRL-20260508`
- Run: `local-training/local_pbt/value_rl/20260508_spy_zone_counts_qaux_r24_b4`
- 5,000 episodes, 24 runners, hybrid self-play/CP1, zone counts enabled.
- Q settings:
  - `CANDIDATE_Q_ONLY=0`
  - `CANDIDATE_Q_BLEND=0.0`
  - `CANDIDATE_Q_LOSS_COEF=1.0`
  - `CANDIDATE_Q_CRITICAL_ONLY=1`
  - `CANDIDATE_Q_FROM_MCTS_TARGETS=0`

Training:

- Completed cleanly at 5,000 episodes.
- Throughput started near 3.3 eps/s and slowed to about 2 eps/s after repeated ONNX exports.
- No runtime errors beyond standard PyTorch/ONNX warnings.

Reduced CP1 eval:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_qaux_cp1_eval16`
- Result: 26/64
- Matchups:
  - Affinity: 2/16
  - Wildfire: 6/16
  - Rally: 5/16
  - Spy mirror: 13/16

Full CP1 eval:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_qaux_cp1_eval32`
- Result: 42/128
- Matchups:
  - Affinity: 7/32
  - Wildfire: 11/32
  - Rally: 3/32
  - Spy mirror: 21/32

Logged CP1:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_qaux_cp1_logged8`
- Result: 9/32
- Action health: `local-training/analysis/action_health_20260508_qaux_logged8/action_health.csv`
- Key failures:
  - Mulligan: 32/32 keeps, 31/32 kept 0-1 true-land hands.
  - Spy: 18 casts, 18/18 with hidden lands estimated.
  - Dread Return: 14 flashbacks, 14/14 not combo-ready and 14/14 before `Lotleth Giant` was in the graveyard.
  - Lotleth target selection remained correct when actually available: 5/5.

Conclusion:

Q-auxiliary self-play is not promotable at 5k. It slightly beats the maskbase total by one game (42/128 vs 41/128) but remains below the 48/128 recent best and does not repair action health. The reduced 26/64 result was a noisy false positive.

## Hard Mulligan Override Diagnostic

Purpose:

- Test whether the visible all-keep mulligan pathology is masking a stronger post-mulligan policy.
- Keep this strictly as an eval-only diagnostic. The override is default-off and does not apply during training unless explicitly enabled.

Implementation:

- Patched `ComputerPlayerRL` to implement the existing `MULLIGAN_HARD_OVERRIDES_ENABLE` hook.
- The override counts resources as true lands plus configurable pseudo-lands.
- Default pseudo-lands: `Land Grant`, `Lotus Petal`, `Generous Ent`, `Troll of Khazad-dum`.
- Key envs:
  - `MULLIGAN_HARD_OVERRIDES_ENABLE=1`
  - `MULLIGAN_HARD_MIN_EFFECTIVE_LANDS=1|2`
  - `MULLIGAN_HARD_MAX_EFFECTIVE_LANDS=5`
  - `MULLIGAN_HARD_MAX_MULLIGANS=2`

### Minimum 1 effective resource

Run:

- `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_qaux_hardmull_eff1_cp1_logged8`
- Source profile: `Pauper-Spy-Combo-ZoneCountsQAuxRL-20260508`
- CP1, 8 games per matchup, logged.

Result:

- Total: 11/32
- Affinity: 1/8
- Wildfire: 3/8
- Rally: 0/8
- Spy mirror: 7/8

Action health:

- Override fired 2 times.
- Mulligans: 2/34 decisions.
- Spy timing remained bad: 28 Spy casts, 25/28 with hidden lands estimated.
- Dread timing remained bad: 16/16 flashbacks not combo-ready.

### Minimum 2 effective resources

Run:

- `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_qaux_hardmull_eff2_cp1_logged8`
- Source profile: `Pauper-Spy-Combo-ZoneCountsQAuxRL-20260508`
- CP1, 8 games per matchup, logged.

Result:

- Total: 9/32
- Affinity: 1/8
- Wildfire: 1/8
- Rally: 0/8
- Spy mirror: 7/8

Action health:

- Override fired 22 times.
- Mulligans: 18/50 decisions.
- Spy timing got worse by the main health metric: 26 Spy casts, 26/26 with hidden lands estimated.
- No-hidden-land Spy opportunities fell to 1/123.
- Dread timing remained bad: 12/12 flashbacks not combo-ready.

Conclusion:

The mulligan hard override does not produce a usable upper bound. A light guard barely changes the sample; a stronger guard hurts and still leaves the same premature Spy/Dread failures. This reinforces the timing-mask conclusion: the current post-mulligan policy does not know how to convert safe windows, so programmatically patching one early decision does not expose a hidden winning agent.

## Spy Action-Facts Candidate Features

Purpose:

- Test the remaining short representation hypothesis: expose the same Java facts used by diagnostics directly to action candidates.
- Avoid changing model shape by using ability-candidate feature slots that were unused for `ACTIVATE_ABILITY_OR_SPELL`.

Implementation:

- Patched `ComputerPlayerRL` with default-off `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE`.
- For `Balustrade Spy` spell candidates:
  - feature 43: is Spy spell candidate
  - feature 44: true lands still in own library, capped and scaled by 4
  - feature 45: no true lands still in own library
- For `Dread Return` spell/flashback candidates:
  - feature 46: is Dread Return candidate
  - feature 47: combo-ready graveyard (`Lotleth Giant` present and at least 3 own creatures)
- Registry: `local-training/local_pbt/value_rl/20260508_spy_zone_counts_actionfacts_registry.json`
- Eval registry: `local-training/local_pbt/value_rl/20260508_spy_zone_counts_actionfacts_eval_registry.json`
- Launcher: `local-training/local_pbt/value_rl/run_20260508_spy_zone_counts_actionfacts.ps1`

Training:

- Source: `Pauper-Spy-Combo-ZoneCountsMaskBase-20260508`
- Profile: `Pauper-Spy-Combo-ZoneCountsActionFactsRL-20260508`
- Run: `local-training/local_pbt/value_rl/20260508_spy_zone_counts_actionfacts_r24_b4`
- 5,000 episodes, 24 runners, hybrid self-play/CP1.
- Completed cleanly at 5,000 episodes.

Reduced CP1 eval:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_actionfacts_cp1_eval16`
- Result: 14/64
- Matchups:
  - Affinity: 3/16
  - Wildfire: 2/16
  - Rally: 0/16
  - Spy mirror: 9/16

Logged CP1:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_zone_counts_actionfacts_cp1_logged8`
- Result: 17/32
- Action health: `local-training/analysis/action_health_20260508_actionfacts_logged8/action_health.csv`
- Key failures:
  - Mulligan: 32/32 keeps.
  - Spy: 33 casts, 31/33 with hidden lands estimated.
  - Dread Return: 23 flashbacks, 23/23 not combo-ready.
  - Lotleth target choice still correct when available: 8/8.

Conclusion:

The direct candidate facts are not enough under the same 5k terminal-RL recipe. The reduced gate is far below promotion quality, and the logged sample shows the same premature Spy/Dread pathology. This weakens the "one short feature" hypothesis: representation alone is not the active blocker unless paired with a stronger credit/search signal.

## Action-Facts Selective Multi-Ply MCTS Sanity Check

Purpose:

- Check whether the new candidate facts make the current selective multi-ply MCTS stack useful enough to justify an AlphaZero-style training loop.
- Keep the sample small and logged because previous online MCTS probes were negative.

Run:

- `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_actionfacts_selective_mpmcts8_cp1_logged4`
- Profile: `Pauper-Spy-Combo-ZoneCountsActionFactsRL-20260508`
- CP1, 4 games per matchup, logged.
- Settings:
  - `--mcts`
  - `--multi-ply-mcts`
  - `--mcts-selective-enable`
  - `MCTS_SELECTIVE_KEYWORDS=Balustrade Spy,Dread Return,Lotleth Giant`
  - `MCTS_ITERATIONS=8`
  - `MCTS_DETERMINIZATIONS=2`
  - `MCTS_ROLLOUT_DEPTH=2`
  - `MCTS_MAX_OUR_ACTIONS=6`
  - `MCTS_TIMEOUT_MS=750`
  - `MCTS_ITER_TIMEOUT_MS=300`

Result:

- Total: 5/16
- Affinity: 0/4
- Wildfire: 0/4
- Rally: 1/4
- Spy mirror: 4/4
- MCTS activations: 74

Action health:

- Land play regressed relative to no-MCTS: 7 pass-over-land cases from 72 land-play opportunities.
- Spy timing was not repaired: 17 Spy casts, 17/17 with hidden lands estimated.
- Dread timing was not repaired: 4/4 Dread flashbacks not combo-ready.
- There were no no-hidden-land Spy opportunities in this sample.

Conclusion:

Selective multi-ply MCTS with the action-facts checkpoint is not a useful online policy improver. It activates, but it still chooses bad Spy/Dread timing and even hurts basic land-play health. This does not justify a local AlphaZero-style training loop using the current search/value stack.

## Recommended Next Session

Do not keep running small one-off BC/Q variants against the same checkpoint without changing the label quality or training objective.

Suggested next step:

1. Inspect CFQ300 logged action behavior only if needed, but it is probably not worth a full logged sweep unless deciding whether to salvage Q.
2. Collect a Spy critical-action branch-return dataset with `-BranchReturnTargets`, not the older normalized winning-target mass.
3. Train candidate-Q on all branched candidates with `CANDIDATE_Q_MCTS_SIGNED_TARGETS=1`, so losing actions are negative Q targets and winning actions are positive Q targets in the same row.
4. Evaluate low blends first (`0.1`, `0.25`, `0.5`) on 64-game sweeps, promote only if full 128-game CP1 beats 48/128 and action health improves.
5. If branch-return Q still fails, return to self-play RL from the zone-count or mask-baseline seed with branch-Q loss as an auxiliary, not eval-time blend only.

Operational notes:

- ONNX CPU branch collection with `Workers=16`, `AiThreads=16`, and `ONNX_FORCE_CPU=1` is the stable local throughput path.
- The regex quoting issue is fixed for the checked-in wrapper, but use the dedicated softbranch runner to avoid PowerShell pipe issues.
- HPC was not used in this session. This kind of branch collection is CPU-bound and would be a reasonable candidate for Zaratan CPU nodes if local wall clock becomes the bottleneck.
