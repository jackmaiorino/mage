# Execution Log

Date: 2026-04-26

## Drain/Save Repair

Problem: London counterfactual runs could finish scenario generation, then log Py4J training-channel or `saveLatestModelAtomic` failures during shutdown.

Cause found: the learner drain counter reached zero before the learner thread finished post-batch sync/save bookkeeping. The caller could then save and shut down while the learner was still using the Python gateway.

Changes:

- Added `PythonModel.awaitTrainingDrained(timeoutMs)` and delegated it through lazy, ONNX, and profile-router models.
- Added pending/in-flight learner tracking in `PythonMLService`.
- Made `MulliganCounterfactualTrainer` wait for a real drain before saving.
- Changed drain timeout from warning-only to hard failure, so failed drains do not proceed into save/shutdown races.

Validation:

- `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`
- Scratch smoke: `smoke_london_drain_20260426_b`
- Result: no saveLatest shutdown failure after the in-flight learner fix.

## London Line Runs

Initial cleaned run:

```powershell
.\scripts\run_mulligan_counterfactual.ps1 `
  -Profile Pauper-Generalist-Value-v2 `
  -Pairs 128 `
  -BatchSize 32 `
  -PostTrainWaitMs 60000 `
  -RunId 20260426_london_balanced_sharp_drain_cf_128_w10 `
  -Workers 10 `
  -LineMode `
  -LineMaxMulls 2 `
  -LineBottomCombos 6 `
  -LineTargetTemperature 0.25 `
  -RandomDecisions `
  -SkipCompile
```

Result:

- `selectedSamples=359`
- No drain/save failure.
- Probe `20260426_after_london_balanced_sharp_drain_probe` still collapsed by deck:
  - Spy Combo: 100% mulligan
  - Mono Red Rally: 100% mulligan
  - Grixis Affinity: 100% keep
  - Jund Wildfire: mostly keep, high-land buckets mulligan

Focused binary-head controls added:

- `LineTrainBottoms`
- `LineTrainEpochs`
- `MctsKlLossCoef`

Focused run with bottom training disabled:

- Run: `20260426_london_binary_epochs8_kl3_cf_128_w12_b`
- `selectedSamples=167`
- `trainPassSamples=1336`
- Probe: `20260426_after_london_binary_epochs8_kl3_probe`

Result before hand-feature fix:

- The head moved more strongly, but still learned mostly deck/resource priors.
- Within deck/effective-land buckets, `p_keep` standard deviation was effectively zero for non-Spy decks.
- Diagnosis: the current input path was not giving the mulligan head an easy usable representation of actual hand composition.

## Representation Repair

Changes:

- Added hand fingerprint to the base-state cache key in `ComputerPlayerRL`.
  - This prevents London follow-up prompts from reusing stale pre-mulligan state.
- Added generic mulligan hand-composition features to candidate features:
  - type counts
  - mana-value buckets
  - average mana value
  - signed hashed bag of card names

This is representation only. It does not add heuristic keep/mull constraints or non-terminal rewards.

Pre-retrain probe:

- Run: `20260426_after_hand_features_pretrain_probe`
- Bucket-level `p_keep` variance became non-zero, confirming the input no longer collapsed completely by deck/resource.

Retrain with hand features:

```powershell
.\scripts\run_mulligan_counterfactual.ps1 `
  -Profile Pauper-Generalist-Value-v2 `
  -Pairs 128 `
  -BatchSize 64 `
  -PostTrainWaitMs 300000 `
  -RunId 20260426_london_binary_handfeat_epochs8_kl3_cf_128_w12 `
  -Workers 12 `
  -LineMode `
  -LineMaxMulls 2 `
  -LineBottomCombos 6 `
  -LineTargetTemperature 0.25 `
  -LineTrainBottoms:$false `
  -LineTrainEpochs 8 `
  -MctsKlLossCoef 3.0 `
  -RandomDecisions `
  -SkipCompile
```

Result:

- `selectedSamples=140`
- `trainPassSamples=1120`
- Probe: `20260426_after_london_binary_handfeat_epochs8_kl3_probe`

Probe summary:

| Deck | Keep rate | Mean `P_keep` |
| --- | ---: | ---: |
| Spy Combo | 80.0% | 0.665 |
| Jund Wildfire | 74.5% | 0.652 |
| Mono Red Rally | 58.5% | 0.522 |
| Grixis Affinity | 60.5% | 0.550 |

Important improvement:

- Within-bucket `p_keep` standard deviations rose to roughly `0.12-0.24` on many buckets.
- The model now distinguishes hands within the same deck/effective-land bucket.

Remaining issue:

- Some confident decisions are strategically questionable.
- The label source is still noisy because it uses one random-play terminal rollout per forced line.

## CP7 Smoke Eval

First CP7 sweep attempt used system Python and failed with `No module named 'torch'`; results were invalid `0/0`.

Correct rerun used `.mtgrl_venv\Scripts\python.exe`:

```powershell
.\.mtgrl_venv\Scripts\python.exe .\scripts\run_cp7_eval_sweep.py `
  --registry local-training/local_pbt/cp7_eval_sweeps/generalist_v2_registry.json `
  --profiles Pauper-Generalist-Value-v2 `
  --games-per-matchup 2 `
  --games-per-job 1 `
  --skill 7 `
  --parallel 4 `
  --ai-threads 12 `
  --timeout-sec 600 `
  --run-id 20260426_generalist_v2_handfeat_cp7_g2_venv `
  --split-agent-decks `
  --skip-compile
```

Result:

- `0/32` vs CP7 skill 7 across the split 4x4 deck matrix.
- No MCTS activations.
- This is a valid sweep, but too small for precise winrates.

Conclusion:

- The mulligan representation/training path is now healthier and no longer collapsed to all-keep/all-mull by bucket.
- This does not yet translate to CP7 wins because the full gameplay policy remains too weak.
- Next experiment should focus on post-mulligan gameplay/value quality, not more single-rollout random-play mulligan labels.
