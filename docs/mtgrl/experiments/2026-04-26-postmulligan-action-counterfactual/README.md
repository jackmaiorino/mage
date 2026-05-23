# Post-Mulligan Action Counterfactual

Date: 2026-04-26

## Question

Can a terminal-only, post-mulligan action counterfactual pass teach Spy Combo useful deep action choices after the mulligan counterfactual repair?

This was intended as the first experiment after the London/mulligan work: keep training terminal-only, avoid CP7 in the training loop, and label early Spy decisions by replaying the same seed while forcing alternate action candidates.

## Implementation

Added:

- `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/ActionCounterfactualTrainer.java`
- `scripts/run_action_counterfactual.ps1`
- `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.spy_combo.txt`

The trainer:

- Runs a greedy baseline rollout and records early eligible action decisions.
- Replays the same seed with the previous greedy prefix forced.
- Forces top policy candidates at the target decision.
- Converts terminal branch outcomes into `mctsVisitTargets`.
- Uses terminal outcome only. When every usable branch wins, it can use faster terminal wins as a terminal-only tiebreaker, controlled by `--win-turn-bonus`.

Compile check:

```powershell
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

## Smoke

Command:

```powershell
.\scripts\run_action_counterfactual.ps1 -Profile Pauper-Generalist-Value-v2 -Scenarios 1 -BatchSize 4 -TimeoutSec 45 -PostTrainWaitMs 120000 -RunId smoke_action_cf_20260426_c -Workers 1 -MaxDecisionDepth 1 -TopK 2 -RandomExtra 0 -TrainEpochs 1 -MctsKlLossCoef 1.0 -SkipCompile
```

Result:

- `trainedScenarios=1`
- `candidateExamples=1`
- `selectedExamples=1`
- Branches: one loss, one win for the first Spy action decision versus Wildfire.

Output:

- `local-training/local_pbt/action_counterfactual/smoke_action_cf_20260426_c`

## Main Run

Command:

```powershell
.\scripts\run_action_counterfactual.ps1 -Profile Pauper-Generalist-Value-v2 -Scenarios 24 -BatchSize 32 -TimeoutSec 60 -PostTrainWaitMs 300000 -RunId 20260426_spy_action_cf_s24_d5_k4 -Workers 8 -MaxDecisionDepth 5 -TopK 4 -RandomExtra 1 -TrainEpochs 4 -MctsKlLossCoef 3.0 -SkipCompile
```

Output:

- `local-training/local_pbt/action_counterfactual/20260426_spy_action_cf_s24_d5_k4`

Results:

- Elapsed: `1883.6s`
- `trainedScenarios=24/24`
- `skippedScenarios=0`
- Branch games: `434`
- Branch wins: `329`
- Branch losses: `105`
- Timeouts: `0`
- Failed force applications: `0`
- Candidate examples: `112`
- Train pass samples: `448`

Training targets by action type:

| action type | selected examples |
|---|---:|
| `ACTIVATE_ABILITY_OR_SPELL` | 69 |
| `SELECT_TARGETS` | 35 |
| `SELECT_CARD` | 6 |
| `CHOOSE_USE` | 2 |

One runtime rules/action failure appeared during branch rollouts:

```text
RL PLAYER ACTIVATION FAILED
Failed ability: Flashback sacrifice three creatures
```

## Post-Run Mulligan Probe

Command:

```powershell
.\scripts\run_mulligan_probe.ps1 -Profile Pauper-Generalist-Value-v2 -SamplesPerDeck 200 -RunId 20260426_after_spy_action_cf_mull_probe -SkipCompile
```

Output:

- `local-training/local_pbt/mulligan_probes/20260426_after_spy_action_cf_mull_probe`

Result:

- The model regressed to near unconditional keep behavior.
- Spy, Wildfire, and Rally kept `100%` of first hands in the 200-sample probe.
- Affinity kept `199/200`.
- Example Spy bucket results:
  - effective lands 0: `keep=100.0%`, `meanPkeep=0.890`
  - effective lands 1: `keep=100.0%`, `meanPkeep=0.916`
  - effective lands 2: `keep=100.0%`, `meanPkeep=0.937`

This is the most important negative result: action-only distillation changed the shared representation/head behavior enough to damage mulligans.

## CP7 Eval

Initial reduced sweep:

```powershell
.\.mtgrl_venv\Scripts\python.exe .\scripts\run_cp7_eval_sweep.py --registry local-training/local_pbt/cp7_eval_sweeps/generalist_v2_registry.json --profiles Pauper-Generalist-Value-v2 --games-per-matchup 2 --games-per-job 1 --skill 7 --parallel 4 --ai-threads 12 --timeout-sec 600 --run-id 20260426_after_spy_action_cf_cp7_g2 --split-agent-decks --opponents "spy,wildfire,rally,affinity" --skip-compile
```

Four early Spy jobs failed with shared GPU service `getDeviceInfo` timeout, so those cells were rerun serially:

```powershell
.\.mtgrl_venv\Scripts\python.exe .\scripts\run_cp7_eval_sweep.py --registry local-training/local_pbt/cp7_eval_sweeps/generalist_v2_registry.json --profiles Pauper-Generalist-Value-v2 --games-per-matchup 2 --games-per-job 1 --skill 7 --parallel 1 --ai-threads 12 --timeout-sec 600 --run-id 20260426_after_spy_action_cf_cp7_spy_retry_g2 --split-agent-decks --opponents "spy,wildfire" --limit-matchups 4 --skip-compile
```

Combined valid result:

- `0/32` wins versus CP7 Skill 7.
- First sweep valid games: `0/28`.
- Retry for missing Spy cells: `0/4`.
- MCTS activations: `0`.

Output folders:

- `local-training/local_pbt/cp7_eval_sweeps/20260426_after_spy_action_cf_cp7_g2`
- `local-training/local_pbt/cp7_eval_sweeps/20260426_after_spy_action_cf_cp7_spy_retry_g2`

## Logged Failure

Command:

```powershell
.\.mtgrl_venv\Scripts\python.exe .\scripts\run_cp7_eval_sweep.py --registry local-training/local_pbt/cp7_eval_sweeps/generalist_v2_registry.json --profiles Pauper-Generalist-Value-v2 --games-per-matchup 1 --games-per-job 1 --skill 7 --parallel 1 --ai-threads 12 --timeout-sec 600 --run-id 20260426_after_spy_action_cf_spy_wildfire_log_g1 --split-agent-decks --opponents "wildfire" --limit-matchups 1 --eval-game-logging --skip-compile
```

Key log:

- `local-training/local_pbt/cp7_eval_sweeps/20260426_after_spy_action_cf_spy_wildfire_log_g1/game_logs/Pauper-Generalist-Value-v2__Deck_-_Spy_Combo__vs__Deck_-_Jund_Wildfire/game_20260426_125609_0001.txt`

Failure mode:

- Spy kept a zero-land seven:
  - `Lead the Stampede; Land Grant; Winding Way; Troll of Khazad-dum; Sagu Wildling; Quirion Ranger; Tinder Wall`
  - `P_keep=0.976`
- It cast `Land Grant`.
- It fetched `Forest`.
- It then passed instead of playing `Forest` on turn 1:
  - `Pass=0.512212`
  - `Play Forest=0.487788`
- It eventually lost on turn 13 with `Balustrade Spy`, `Lotleth Giant`, and other action in hand.

## Conclusion

The generic action counterfactual machinery works mechanically: it can replay branches, force actions, produce labels, train, and survive medium runs without branch timeouts.

It did not improve CP7 performance. The experiment is negative for the current training recipe:

1. Action distillation alone is not stable enough after mulligan repair.
2. The model still fails basic sequencing, such as playing the fetched `Forest`.
3. A post-run trainer review found that branch-forcing ordinals counted one-option pass decisions, while the recorded training examples filtered them out. Deeper labels in this run can therefore be noisy because a branch may have forced a later one-option-aligned ordinal rather than the intended filtered decision.
4. We need a training recipe that preserves learned mulligan behavior while improving action choices.
5. CP7 eval startup needs a service readiness fix or serial warmup; otherwise first parallel jobs can fail with shared GPU timeout.

## Follow-Up Fixes Applied

After the result review, the trainer was updated for the next run:

- Branch-forcing ordinals now ignore target-type decisions with fewer than two candidates, matching the filtered training-example list.
- `action_branch_samples.csv` now includes `forced_text`.
- `action_training_samples.csv` now includes `baseline_text` and `best_text`.
- `run_cp7_eval_sweep.py` now has `--serial-warmup-jobs` with default `1`, so the first eval job warms the shared GPU service before parallel jobs launch.
- `ActionCounterfactualTrainer` now has optional `--loss-turn-bonus`. Default is `0.0`; setting it above zero allows all-loss CP7 branches to produce terminal-only labels by preferring later terminal losses. This is a curriculum option, not part of the original binary win/loss run.

Post-fix smoke observations:

- RL-opponent smoke produced a visible bad label: `Pass` beat `Play Forest` by terminal outcome against the weak RL opponent.
- CP7-opponent smoke produced no binary label because both branches lost, but it exposed the need for an all-loss terminal tiebreaker if CP7 is used as the branch opponent.

## Next Experiment

Run action counterfactual with multi-task rehearsal instead of action-only distillation:

1. Interleave a small London/mulligan counterfactual rehearsal batch after each action CF pass.
2. Prefer CP7 or mixed CP7/RL branch opponents for action CF; pure RL-opponent labels can reinforce bad sequencing.
3. If using CP7 branch opponents, enable a small `--loss-turn-bonus` to get learning signal before the model can actually win branches.
4. Add a probe gate before CP7 eval:
   - reject if any non-Spy deck has first-hand keep rate above `90%`;
   - reject if Spy zero-effective-land keep is above a configured threshold unless the hand has `Land Grant` plus an executable line.
5. Rerun Spy action CF with the same 24-scenario size using the ordinal fix.
6. Inspect branch labels directly for `Pass` vs `Play Forest` and other mana sequencing choices.
7. Probe mulligans and only run CP7 if they remain sane.

This keeps the project terminal-only, but treats mulligan and action competence as a continual-learning problem rather than independent one-off fine-tunes.
