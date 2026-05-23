# Thesis-Clean Signed Branch-Return Policy

Date: 2026-05-13

## Motivation

The accepted CP7 compact-log probe showed the same hard-pressure pattern as
prior health logs: the policy reaches plausible development states, then fails
to convert. In the Rally loss it passed through several mana-action windows,
cast `Balustrade Spy` only at 5 life, milled itself, and died before a follow-up.
In the Affinity loss it reached a large graveyard and still did not produce a
decisive post-mill line before the opponent closed.

The next thesis-clean mechanism was therefore not another continuation schedule.
It was the previously identified missing objective: train the policy itself from
generic terminal branch returns, instead of using branch returns only as a
candidate-Q reranker.

## Code Change

Added default-off Python learner envs:

```text
BRANCH_RETURN_POLICY_LOSS_COEF
BRANCH_RETURN_POLICY_TEMPERATURE
BRANCH_RETURN_POLICY_MIN_GAP
BRANCH_RETURN_POLICY_TARGET_MIX
```

When signed branch-return targets are present (`CANDIDATE_Q_MCTS_SIGNED_TARGETS=1`),
the loss interprets the MCTS target tensor as terminal returns in `[-1, 1]`,
with `-2` as unobserved. It builds a softmax preference target over only the
branch-evaluated legal candidates, then applies cross-entropy to the current
policy conditional on that observed set.

This is thesis-clean: the target source is terminal win/loss from generic branch
rollouts, with no action-text filters, card names, Spy terminal labels, heuristic
step rewards, or selective MCTS card gates.

Wrapper support was added to `scripts/run_action_counterfactual.ps1`.

Verification:

```text
python -m py_compile py4j_entry_point.py
PowerShell parser check for run_action_counterfactual.ps1
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

Important follow-up: `scripts/run_action_counterfactual.ps1` had drifted to
`MODEL_D_MODEL=256` and `MODEL_NUM_LAYERS=4` defaults, while the canonical
thesis-clean Spy checkpoint and eval registries use `128/2`. That made the first
two May 13 branch-policy imports architecture-mismatched and unsuitable as
policy-quality evidence. The wrapper default is now back to `128/2`.

## Head-Only Branch Policy Import

Profile:

```text
Pauper-Spy-Combo-Value-BranchRetPolicyHead-20260513
```

Source checkpoint: accepted `Pauper-Spy-Combo-Value`.

Input dataset:

```text
local-training/local_pbt/action_counterfactual/20260512_thesis_clean_generic_branchret_balanced_cp7
```

Training command shape:

```text
ImportTrainingDataPath=<balanced_cp7_branch_returns>
BranchReturnTargets=true
BranchReturnPolicyLossCoef=0.5
BranchReturnPolicyTemperature=0.5
BranchReturnPolicyMinGap=0.25
BranchReturnPolicyTargetMix=0.5
DistillHeadOnly=true
TrainEpochs=6
CandidatePermutations=2
LOAD_OPTIMIZER_STATE=0
RESET_TRAINING_STATE_ON_LOAD=1
```

Import result:

```text
importedTrainingExamples=197
trainPassSamples=2364
train_steps=78
```

Model hashes after import:

```text
model.pt        C063BF46A794
model_latest.pt C063BF46A794
```

## Reduced CP7 Gate

Run:

```text
20260513_branchret_policy_head_spy_cp7_unique_eval16
```

Result:

```text
Overall: 0/64 = 0.00%
```

Per matchup:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 0 | 16 | 0.00% |
| Jund Wildfire | 0 | 16 | 0.00% |
| Mono Red Rally | 0 | 16 | 0.00% |
| Grixis Affinity | 0 | 16 | 0.00% |

## Corrected Low-Dose Anchored Check

After fixing the wrapper defaults, reran a small guarded branch-policy import:

```text
Pauper-Spy-Combo-Value-BranchRetPolicyAnchoredLow128-20260513
```

Settings:

```text
BranchReturnPolicyLossCoef=0.05
BranchReturnPolicyTemperature=1.0
BranchReturnPolicyMinGap=0.25
BranchReturnPolicyTargetMix=0.10
REFERENCE_POLICY_KL_COEF=1.0
DistillHeadOnly=true
TrainEpochs=2
CandidatePermutations=1
MODEL_D_MODEL=128
MODEL_NUM_LAYERS=2
```

Import result:

```text
training_examples=197
train_pass_samples=394
train_steps=14
model_latest.pt hash=793F2046CA33 size=71087106
```

Reduced CP7 screen:

```text
20260513_branchret_policy_anchorlow128_spy_cp7_unique_eval4
Overall: 6/16 = 37.50%
```

Per matchup:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 4 | 4 | 100.00% |
| Jund Wildfire | 0 | 4 | 0.00% |
| Mono Red Rally | 1 | 4 | 25.00% |
| Grixis Affinity | 1 | 4 | 25.00% |

This corrected run is non-catastrophic but still below the accepted CP7 profile,
especially into Jund. Do not expand to 16 games or spend HPC on this branch.

## Corrected Original-Strength Check

Also reran the original branch-policy settings after the wrapper fix:

```text
Pauper-Spy-Combo-Value-BranchRetPolicyHead128-20260513
```

Settings:

```text
BranchReturnPolicyLossCoef=0.5
BranchReturnPolicyTemperature=0.5
BranchReturnPolicyMinGap=0.25
BranchReturnPolicyTargetMix=0.5
DistillHeadOnly=true
TrainEpochs=6
CandidatePermutations=2
MODEL_D_MODEL=128
MODEL_NUM_LAYERS=2
```

Import result:

```text
training_examples=197
train_pass_samples=2364
train_steps=78
model_latest.pt hash=402AE1756784 size=71087106
```

Reduced CP7 screen:

```text
20260513_branchret_policy_head128_spy_cp7_unique_eval4
Overall: 4/16 = 25.00%
```

Per matchup:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 3 | 4 | 75.00% |
| Jund Wildfire | 0 | 4 | 0.00% |
| Mono Red Rally | 1 | 4 | 25.00% |
| Grixis Affinity | 0 | 4 | 0.00% |

## Verdict

Rejected. Do not run CP1 and do not spend HPC.

The signed branch-return policy objective is mechanically wired, but both valid
128/2 reruns failed the reduced CP7 gate. The original 0/64 result should be
treated as an architecture-mismatch diagnostic, not as a clean policy result.
The corrected original-strength run was worse than the low-dose anchored run,
so this same-shape offline branch-return preference dataset is rejected.

Next work should not collect another same-shape branch-return dataset. The
remaining useful directions are:

1. Improve the generic teacher before training from it, especially by producing
   trajectory-level targets rather than isolated one-step preferences.
2. Resume card-level belief/determinization work only after a calibration gate
   shows the belief head can represent non-Spy hidden zones.
3. Use longer terminal continuation only if a local mechanism first clears a
   reduced CP1/CP7 gate.
