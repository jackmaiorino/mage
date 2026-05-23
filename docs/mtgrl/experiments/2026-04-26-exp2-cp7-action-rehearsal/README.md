# Exp2 CP7 Action Counterfactual With Mulligan Rehearsal

Date: 2026-04-26

## Question

Can we fix the post-mulligan action failure from the previous run while preserving the London mulligan repair?

The specific failure we were targeting was Spy Combo casting `Land Grant`, fetching `Forest`, then passing instead of playing the `Forest`. The goal was still terminal-only training: no shaped game rewards, no hand-written "keep 2-3 lands" constraints, and no heuristic action reward.

## Profile

Created an isolated experiment profile:

- `Pauper-Generalist-Value-v2-Exp2`

Source checkpoint:

- `Pauper-Generalist-Value-v2/models/snapshots/snapshot_step_5000.pt`

Reason:

- The main `Pauper-Generalist-Value-v2` profile had already been damaged by the first action-only counterfactual run, so Exp2 started from a cleaner pre-action checkpoint.

## Implementation Changes Used

Action counterfactual fixes from the previous experiment were included:

- Branch-forcing ordinals now ignore target-type decisions with fewer than two candidates, matching the filtered training examples.
- `action_branch_samples.csv` records forced candidate text.
- `action_training_samples.csv` records baseline and selected candidate text.
- `--loss-turn-bonus` can use later terminal losses as a terminal-only all-loss tiebreaker.
- `--skip-pass-best` drops examples where the selected terminal-best action is `Pass`.
- CP7 eval has `--serial-warmup-jobs` to avoid first-job shared GPU startup failures.

## Runs

### Stopped Large Mulligan Repair

Run:

- `20260426_exp2_london_binary_handfeat_epochs8_kl3_cf_128_w12`

Status:

- Stopped at `64/128` scenarios.
- It was underutilizing the machine and had not saved a model checkpoint.
- This run was not used for conclusions.

### Smaller Mulligan Repair

Command shape:

```powershell
.\scripts\run_mulligan_counterfactual.ps1 -Profile Pauper-Generalist-Value-v2-Exp2 -Pairs 32 -BatchSize 32 -TimeoutSec 60 -PostTrainWaitMs 300000 -Opponent rl -RunId 20260426_exp2_london_repair_s32_w12_c -ReportEvery 4 -Workers 12 -LineMode -LineMaxMulls 2 -LineBottomCombos 0 -LineMarginMin 0.05 -LineTargetTemperature 0.50 -LineTrainEpochs 8 -MctsKlLossCoef 3.0 -SkipCompile
```

Output:

- `local-training/local_pbt/mulligan_counterfactual/20260426_exp2_london_repair_s32_w12_c`

Result:

- `trained_scenarios=32`
- `trained_pass_samples=816`
- elapsed `2511.5s`
- model checkpoint saved

Probe after this repair was directionally usable for Spy:

- Spy effective lands 0: keep `45.6%`, mean `P_keep=0.478`
- Spy effective lands 1: keep `72.2%`, mean `P_keep=0.531`
- Spy effective lands 2: keep `87.2%`, mean `P_keep=0.601`

Non-Spy buckets were still noisy, but this was good enough to audit action labels.

### CP7 Action Label Audit

Initial audit:

- `20260426_exp2_cp7_action_label_audit_s4_d2`

Result:

- 8 labels total.
- 4 selected `Pass`, so the labels were unsafe to train blindly.

Filtered audit:

- `20260426_exp2_cp7_action_label_audit_s2_d2_skippass`

Result:

- `--skip-pass-best` worked.
- Selected labels included `Cast Lotus Petal` and `Play Swamp`.

### Main CP7 Action Counterfactual

Command shape:

```powershell
.\scripts\run_action_counterfactual.ps1 -Profile Pauper-Generalist-Value-v2-Exp2 -Opponent cp7 -Scenarios 12 -BatchSize 32 -TimeoutSec 75 -PostTrainWaitMs 300000 -RunId 20260426_exp2_cp7_action_cf_s12_d4_k3_skippass -ReportEvery 2 -Workers 4 -MaxDecisionDepth 4 -TopK 3 -RandomExtra 0 -TrainEpochs 4 -LossTurnBonus 0.99 -MctsKlLossCoef 3.0 -SkipPassBest -SkipCompile
```

Output:

- `local-training/local_pbt/action_counterfactual/20260426_exp2_cp7_action_cf_s12_d4_k3_skippass`

Result:

- `trained_scenarios=11/12`
- `candidate_examples=24`
- `train_pass_samples=96`
- elapsed `826.1s`
- branch games `100`
- branch wins `5`
- failed forced applications `9`

Selected label mix:

| selected candidate | examples |
|---|---:|
| `Play Forest` | 12 |
| `Cast Lotus Petal` | 5 |
| `Play Swamp` | 3 |
| `Cast Land Grant` | 2 |
| blank target text | 2 |

The main concern is that only `5/100` CP7 branch games actually won. Most labels therefore came from all-loss terminal tiebreaks, not from actions that demonstrated a winning line.

### Mulligan Rehearsal

Command shape:

```powershell
.\scripts\run_mulligan_counterfactual.ps1 -Profile Pauper-Generalist-Value-v2-Exp2 -Pairs 16 -BatchSize 32 -TimeoutSec 60 -PostTrainWaitMs 300000 -Opponent rl -RunId 20260426_exp2_after_action_mull_rehearsal_s16_e2 -ReportEvery 4 -Workers 12 -LineMode -LineMaxMulls 2 -LineBottomCombos 0 -LineMarginMin 0.05 -LineTargetTemperature 0.50 -LineTrainEpochs 2 -MctsKlLossCoef 2.0 -SkipCompile
```

Output:

- `local-training/local_pbt/mulligan_counterfactual/20260426_exp2_after_action_mull_rehearsal_s16_e2`

Result:

- `trained_scenarios=16`
- `trained_pass_samples=98`
- elapsed `1268.5s`
- model checkpoint saved

## Post-Rehearsal Mulligan Probe

Command:

```powershell
.\scripts\run_mulligan_probe.ps1 -Profile Pauper-Generalist-Value-v2-Exp2 -SamplesPerDeck 200 -RunId 20260426_exp2_after_action_rehearsal_probe -SkipCompile
```

Output:

- `local-training/local_pbt/mulligan_probes/20260426_exp2_after_action_rehearsal_probe`

Overall first-hand keep rates:

| deck | keep rate | mean P_keep |
|---|---:|---:|
| Spy Combo | `67.0%` | `0.539` |
| Jund Wildfire | `56.0%` | `0.531` |
| Mono Red Rally | `41.0%` | `0.491` |
| Grixis Affinity | `54.0%` | `0.495` |

Important bucket failures:

- Wildfire 0 effective lands: keep `85.7%` on 7 samples.
- Wildfire 4 effective lands: keep `34.8%`.
- Rally 0 effective lands: keep `52.9%`.
- Rally 3 effective lands: keep `27.1%`.
- Affinity 1 effective land: keep `45.6%`.

Conclusion from the probe:

- Spy was not catastrophically regressed.
- The generalist mulligan policy was still not stable across decks.
- This failed the intended "mulligan remains sane" gate.

## CP7 Evaluation

Even though the probe did not pass cleanly, a reduced CP7 sweep was run as an objective damage check.

Command shape:

```powershell
.\.mtgrl_venv\Scripts\python.exe .\scripts\run_cp7_eval_sweep.py --registry local-training/local_pbt/cp7_eval_sweeps/exp2_registry.json --profiles Pauper-Generalist-Value-v2-Exp2 --games-per-matchup 2 --games-per-job 1 --skill 7 --parallel 8 --serial-warmup-jobs 1 --ai-threads 8 --timeout-sec 600 --run-id 20260426_exp2_after_action_rehearsal_cp7_g2_p8 --split-agent-decks --opponents "spy,wildfire,rally,affinity" --skip-compile
```

Output:

- `local-training/local_pbt/cp7_eval_sweeps/20260426_exp2_after_action_rehearsal_cp7_g2_p8`

Result:

- `0/32` wins
- `0.0%` winrate
- no MCTS activations
- no eval infrastructure failures
- elapsed about `7.4` minutes

Every 2-game matchup cell was `0/2`.

## Utilization Notes

The CP7 eval run used `parallel=8` with one-game jobs:

- During the parallel phase, CPU sampled around `65-80%`.
- GPU stayed low, usually single-digit utilization.
- The GPU service was mostly idle because games produced few model scoring requests.
- This path is mostly Java simulation plus CP7 decision work, not neural inference.
- Utilization dropped near the end because only a few long-tail jobs remained.

Future CP7 sweeps should start at `parallel=10-12` on this 24-logical-processor machine, with a serial warmup job retained. For counterfactual training, simply adding scenario workers is not enough; the generator needs branch-level parallelism and a deeper queue of branch games to keep CPU saturated.

## Conclusion

This experiment is negative.

What worked mechanically:

- The isolated Exp2 profile avoided damaging the main profile further.
- The ordinal fix and `--skip-pass-best` filter produced more plausible action labels than the first action-CF run.
- Serial CP7 eval warmup prevented shared GPU startup failures.
- `parallel=8` CP7 eval was a better utilization shape than earlier lower-parallel sweeps.

What failed:

- CP7 branch search found too few actual wins: only `5/100` branch games.
- Most action labels were all-loss tiebreak labels, not terminal-winning labels.
- Mulligan behavior remained unstable after rehearsal.
- Reduced CP7 eval was still `0/32`.

The key lesson is that one-ply action distillation from all-loss CP7 branches is not a good enough learning signal. It is still terminal-only, but it does not identify winning lines. For Spy Combo especially, the model needs multi-action line discovery, not isolated action preferences.

## Next Experiment

Run a terminal-only winning-line discovery experiment, starting with a Spy specialist rather than the four-deck generalist.

Hypothesis:

- If search can find actual terminal-winning Spy lines, training on the complete winning decision prefix will be much more useful than training on one-ply all-loss tiebreaks.
- A specialist is the fastest way to validate the method because the generalist currently shows cross-deck interference in mulligans and action heads.

Proposed profile:

- `Pauper-Spy-Combo-LineSearch-v1`
- initialize from the clean `snapshot_step_5000.pt` checkpoint or the cleanest pre-action Spy/generalist checkpoint available.

Generator:

- Sample Spy Combo as the agent deck.
- Opponent pool: CP7 skill 7 over the four-deck small meta, optionally with sampled hidden hand determinization.
- For each seed, search action lines up to depth `6-8`.
- Keep terminal reward only.
- Train only when a branch actually wins, or when a line reaches a clearly better terminal outcome by binary win/loss over a paired baseline.
- Do not use loss-turn all-loss labels for the first version.
- Store the full successful prefix: London mulligan choice, bottoms if present, and all post-mulligan decisions until the decisive line.

Training:

- Train the policy heads on the complete winning prefix using existing candidate-target machinery.
- Interleave a small Spy mulligan rehearsal batch after each line-search batch.
- Freeze or heavily downweight unrelated decks for this first proof.
- Add a probe that directly checks `Pass` versus `Play Forest`, `Cast Lotus Petal`, and `Cast Land Grant` on captured Spy states.

Success gates before a CP7 sweep:

- At least `20` terminal-winning prefixes found in the first search batch.
- Replay accuracy on the discovered winning prefixes above `80%`.
- Spy zero-effective-land mulligan bucket does not move toward unconditional keep.
- Targeted Spy action probe prefers the searched winning action over `Pass` on at least `70%` of captured states.

Eval:

- First: replay the fixed training seeds to verify the model can reproduce searched wins.
- Second: CP7 reduced sweep, Spy agent only, `games-per-matchup=4`, `parallel=10-12`.
- Third: if Spy shows nonzero CP7 wins, repeat with Wildfire line search.

Utilization plan:

- Implement or modify the search generator so branch games are the unit of parallel work.
- Keep at least `2x` logical cores worth of queued branch games.
- Sample CPU/GPU utilization after the first 60 seconds.
- If CPU is below `80%`, increase branch workers or reduce per-scenario serial depth.
- If GPU is below `15%` and CPU is saturated, accept that the bottleneck is simulation; if CPU is not saturated, fix orchestration before continuing.

This keeps the core project principle intact: terminal reward only. The change is that we stop asking all-loss one-ply branches to teach combo play and instead require the search procedure to find actual winning terminal trajectories before training.
