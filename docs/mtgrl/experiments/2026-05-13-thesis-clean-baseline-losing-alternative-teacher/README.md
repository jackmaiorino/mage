# Thesis-Clean Baseline-Losing Alternative Teacher

Date: 2026-05-13

## Question

Can a generic terminal counterfactual teacher focus on genuinely unsolved states by keeping only decisions where the accepted policy's baseline action loses and at least one sibling action wins?

This is thesis-clean:

- terminal `WIN` only;
- no heuristic step rewards;
- no Spy-specific terminal mode;
- no Spy combo action facts;
- no action-text/card-name filters;
- no selective MCTS keyword gates.

## Implementation

Added a default-off `ActionCounterfactualTrainer` option:

```text
--baseline-losing-alternative-only=true
```

When enabled, the trainer only emits a policy target if:

1. the policy's chosen baseline action was among the evaluated branches;
2. that baseline branch lost by terminal outcome;
3. at least one sibling branch won by terminal outcome.

The target distribution is a softmax over the winning sibling branches using the existing generic terminal branch score.

Wrappers:

```powershell
scripts/run_action_counterfactual.ps1 -BaselineLosingAlternativeOnly
scripts/run_spy_line_search.ps1 -BaselineLosingAlternativeOnly
```

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
PowerShell parser checks passed for both wrappers
```

## Collection

Run:

`20260513_baseline_losing_alt_affinity_cp7_collect64`

Export:

`local-training/local_pbt/action_counterfactual_data/20260513_baseline_losing_alt_affinity_cp7_collect64.ser`

Settings:

- accepted `Pauper-Spy-Combo-Value` checkpoint;
- Spy vs Grixis Affinity, CP7;
- 64 scenarios;
- `max_decision_depth=10`;
- `top_k=4`;
- `random_extra=2`;
- `max_game_turns=12`;
- `skip_pass_training=true`;
- `skip_blank_training=true`;
- `skip_mulligan_training=true`;
- collect-only.

Result:

```text
trainedScenarios=14
skippedScenarios=50
candidateExamples=22
selectedExamples=22
```

Accepted-checkpoint score:

```text
top1=0/22 = 0.00%
targetSetTop1=0/22 = 0.00%
avgTargetProb=0.215337
avgRank=2.9545
```

This is the useful result: unlike the searched-prefix corpora, this filter found states the accepted policy did not already rank correctly.

## Training Checks

### Low-Dose Head-Only

Profile:

`Pauper-Spy-Combo-Value-BaselineAltAffinity-20260513`

Import:

`20260513_baselinealt_affinity_headonly_e4p4_train22`

Settings:

- `MCTS_KL_LOSS_COEF=0.25`;
- `DISTILL_HEAD_ONLY=1`;
- 4 epochs;
- 4 candidate permutations.

Post-train score:

```text
top1=7/22 = 31.82%
targetSetTop1=8/22 = 36.36%
avgTargetProb=0.243143
avgRank=2.4091
```

Affinity CP7 gate:

```text
20260513_baselinealt_affinity_headonly_e4p4_affinity_cp7_eval16
2/16 = 12.50%
```

### Stronger Head-Only

Profile:

`Pauper-Spy-Combo-Value-BaselineAltAffinityStrong-20260513`

Import:

`20260513_baselinealt_affinity_headonly_e8p4_kl1_train22`

Settings:

- `MCTS_KL_LOSS_COEF=1.0`;
- `DISTILL_HEAD_ONLY=1`;
- 8 epochs;
- 4 candidate permutations.

Post-train score:

```text
top1=10/22 = 45.45%
targetSetTop1=13/22 = 59.09%
avgTargetProb=0.311936
avgRank=2.2727
```

Affinity CP7 gate:

```text
20260513_baselinealt_affinity_headonly_e8p4_kl1_affinity_cp7_eval16
1/16 = 6.25%
```

### Policy-Path-Only

Profile:

`Pauper-Spy-Combo-Value-BaselineAltAffinityPolicyPath-20260513`

Import:

`20260513_baselinealt_affinity_policypath_e8p4_kl05_train22`

Settings:

- `MCTS_KL_LOSS_COEF=0.5`;
- `DISTILL_POLICY_PATH_ONLY=1`;
- 8 epochs;
- 4 candidate permutations.

Post-train score:

```text
top1=10/22 = 45.45%
targetSetTop1=13/22 = 59.09%
avgTargetProb=0.406249
avgRank=2.0455
```

Affinity CP7 gate:

```text
20260513_baselinealt_affinity_policypath_e8p4_kl05_affinity_cp7_eval16
2/16 = 12.50%
```

## Verdict

Rejected for promotion and HPC.

The filter is the first generic offline teacher today that found genuinely missed accepted-policy states, so the mechanism is worth keeping. But the 22-example Affinity-only corpus did not transfer to fresh CP7 Affinity games under head-only or policy-path imports. Directly fitting sparse local mistake labels is not enough.

Next useful work should either broaden this teacher across more hard-matchup states or change the training objective so sparse corrected decisions are replayed in coherent trajectories, rather than importing isolated one-step labels.

## Broad CP7 Unique-Pool Follow-Up

Question:

Was the Affinity-only miss corpus too sparse and too narrow?

Collection:

`20260513_baseline_losing_alt_unique_cp7_collect128`

Export:

`local-training/local_pbt/action_counterfactual_data/20260513_baseline_losing_alt_unique_cp7_collect128.ser`

Settings:

- accepted `Pauper-Spy-Combo-Value` checkpoint;
- four-opponent unique CP7 pool;
- 128 scenarios;
- `max_decision_depth=10`;
- `top_k=4`;
- `random_extra=2`;
- collect-only;
- early stop threshold 96 examples, not reached.

Result:

```text
trainedScenarios=33
skippedScenarios=95
candidateExamples=66
selectedExamples=66
elapsed_sec=2704.3
```

Accepted-checkpoint score:

```text
top1=0/66 = 0.00%
targetSetTop1=0/66 = 0.00%
avgTargetProb=0.276201
avgRank=2.6818
```

Training:

`Pauper-Spy-Combo-Value-BaselineAltUniquePolicyPath-20260513`

Import:

`20260513_baselinealt_unique_policypath_e8p4_kl05_train66`

Settings:

- `MCTS_KL_LOSS_COEF=0.5`;
- `DISTILL_POLICY_PATH_ONLY=1`;
- 8 epochs;
- 4 candidate permutations.

Post-train score:

```text
top1=51/66 = 77.27%
targetSetTop1=59/66 = 89.39%
avgTargetProb=0.669014
avgRank=1.3788
model_latest.pt hash prefix=42B8FD7F65D9
```

Reduced CP7 gate:

`20260513_baselinealt_unique_policypath_e8p4_kl05_cp7_eval4`

Overall:

`5/16 = 31.25%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 3 | 4 | 75.00% |
| Jund Wildfire | 1 | 4 | 25.00% |
| Mono Red Rally | 0 | 4 | 0.00% |
| Grixis Affinity | 1 | 4 | 25.00% |

Read:

The broader corpus confirmed the baseline-losing-alternative filter can find many genuinely missed labels, and the policy-path import can fit them. It still failed the fresh-game CP7 gate, especially Rally. This rejects isolated one-step imports from the current filter. The next version needs coherent replay or an online loop that forces the corrected decisions in context and collects downstream states, rather than fitting independent state labels.

## Replay DAgger Follow-Up

Question:

Can the broad correction set work if replay turns it into in-context prefix anchors plus first-deviation repairs?

Replay:

`20260513_baseline_losing_alt_unique_cp7_replay_dagger`

Inputs:

- replay file: `20260513_baseline_losing_alt_unique_cp7_collect128/action_training_samples.csv`;
- accepted checkpoint;
- same four-opponent CP7 deck pool;
- `ReplayDeviationRepeat=8`.

Replay result:

```text
replayMatched=0/66
scenarioWins=20/33
deviationExamples=10
daggerExamples=115
```

Accepted score:

```text
DAgger:    top1=35/115 = 30.43%, avgTargetProb=0.284499
Deviation: top1=0/10   =  0.00%, avgTargetProb=0.249936
```

Training:

`Pauper-Spy-Combo-Value-BaselineAltUniqueDagger-20260513`

Import:

`20260513_baselinealt_unique_dagger_policypath_e4p2_kl05_train115`

Settings:

- DAgger replay export;
- `MCTS_KL_LOSS_COEF=0.5`;
- `DISTILL_POLICY_PATH_ONLY=1`;
- 4 epochs;
- 2 candidate permutations.

Post-train score:

```text
DAgger set:        top1=98/115 = 85.22%, avgTargetProb=0.771429
Original miss set: top1=29/66  = 43.94%, avgTargetProb=0.413636
model_latest.pt hash prefix=30195788A7E0
```

Reduced CP7 gate:

`20260513_baselinealt_unique_dagger_policypath_e4p2_kl05_cp7_eval4`

Overall:

`3/16 = 18.75%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 2 | 4 | 50.00% |
| Jund Wildfire | 1 | 4 | 25.00% |
| Mono Red Rally | 0 | 4 | 0.00% |
| Grixis Affinity | 0 | 4 | 0.00% |

Final read:

The replay/DAgger version fitted the coherent export but still degraded fresh CP7 play. The current offline counterfactual family has a useful diagnostic filter, not a working policy-improvement operator. Do not scale it on HPC.
