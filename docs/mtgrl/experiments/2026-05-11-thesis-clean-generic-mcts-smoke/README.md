# Thesis-Clean Generic MCTS Smoke

Date: 2026-05-11

## Question

Is non-selective train-time search currently stable and cheap enough to reconsider AlphaZero-style policy improvement?

This is not a promotion run. It is a bounded plumbing and throughput smoke from the accepted Affinity-pressure checkpoint.

## Setup

Registry: `local-training/local_pbt/thesis_clean/20260511_thesis_clean_generic_mcts_smoke_registry.json`

Settings:

- one active Spy profile
- terminal rewards only
- no action-facts features
- no zone-count toggle
- no Spy terminal mode
- no selective MCTS keywords
- `MCTS_TRAINING_ENABLE=1`
- `MULTI_PLY_MCTS=1`
- `MCTS_ITERATIONS=2`
- `MCTS_DETERMINIZATIONS=1`
- `MCTS_SKIP_TOP_PROB=0.95`
- max `200` episodes or `1800` seconds

Gate:

- complete without stalls or GPU service failures
- record nonzero MCTS activations
- produce enough episodes to estimate slowdown

## Result: local 200-episode smoke

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_generic_mcts_smoke_200ep_rerun1/phase_001_train`

Status: rejected as plumbing-only, not a candidate checkpoint.

Observed:

- Started from the accepted Affinity-pressure checkpoint at episode `292239`.
- Produced parsed training rows through episode `292438` before manual stop; head-usage logging also recorded episode `292439`.
- Parsed new training sample: `190` games, `32` wins (`16.84%`), average `24.41s/game`.
- Opponent mix in parsed rows: `127` self-play games (`21` wins) and `63` CP1 games (`11` wins).
- Throughput while active was about `0.1-0.5 eps/s` with `8` local runners, far below the non-search local training path.
- `run_local_pbt.py` printed `ONNX export not available, skipping TRT`; the GPU service never registered profile contexts in metrics during the run.
- Active Spy weights were unchanged after the run: `model.pt` and `model_latest.pt` SHA-256 hashes matched `pre_20260511_thesis_clean_generic_mcts_smoke_30m`.
- MCTS gate stats were not captured. The trainer exited with `rc=0` one episode below the stop target, and the orchestrator restarted it instead of treating the bounded smoke as complete. The run was then manually stopped to avoid a no-progress restart loop.

Decision:

- Do not spend HPC on generic train-time MCTS yet.
- Do not evaluate or promote this branch; no checkpoint changed.
- Next work should be instrumentation and bounded-run plumbing:
  - capture `MCTS_GATE` stats into run artifacts on every trainer exit,
  - make bounded `TOTAL_EPISODES` runs stop cleanly when the trainer exits at or near target,
  - diagnose why the direct smoke path could not import ONNX export support,
  - rerun a smaller smoke only after those observability fixes.

## Plumbing fixes

Implemented after the failed direct smoke:

- `scripts/run_local_pbt.py`
  - archives `local-training/local_pbt/trainer.log` on trainer exit before restart can overwrite it,
  - surfaces MCTS/eval diagnostics from the archived trainer log in orchestrator stdout,
  - treats clean trainer exits within `PBT_TARGET_EXIT_TOLERANCE` of `TOTAL_EPISODES` as complete,
  - kills only the trainer process tree instead of all local `java.exe` processes,
  - logs the `ImportError` reason when ONNX export support is unavailable.
- `scripts/run_mtgrl_autonomous_cycle.ps1`
  - falls back to `py -3.12` instead of plain `py` when `.mtgrl_venv` is absent.
- `RLTrainer.trainMultiProfile`
  - prints `MCTS_GATE` and MCTS timing stats at multi-profile shutdown.

Verification:

- `.mtgrl_venv\Scripts\python.exe -m py_compile scripts/run_local_pbt.py`
- `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`
- PowerShell parser check for `scripts/run_mtgrl_autonomous_cycle.ps1`

## Result: venv 16-episode plumbing smoke

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_generic_mcts_smoke_venv_16ep/phase_001_train`

Status: successful plumbing smoke, rejected as a model candidate.

Observed:

- Used `.mtgrl_venv\Scripts\python.exe`; ONNX export succeeded.
- Exported fresh ONNX bundle `v20260511T111336_658827`.
- Finished cleanly with `exit=0`.
- Produced 16 new parsed rows, `3` wins.
- MCTS fired heavily: archived trainer log reported `MCTS_STATS_FINAL calls=839`, `avg_wallMs=224`, `avg_iters=2`.
- Average search cost was dominated by leaf evaluation: `eval=108089us` per iteration in `MCTS_STATS_FINAL`.
- The run updated `model_latest.pt`; it was restored immediately from `pre_20260511_thesis_clean_generic_mcts_smoke_30m`.

Interpretation:

- Generic non-selective train-time MCTS is wired and active.
- Even the minimal `2`-iteration setting is expensive locally: about 16 bounded episodes in roughly 4 minutes with 4 runners.
- The current bottleneck is search evaluation cost, not determinization setup or clone cost.
- Still not HPC-worthy as a training run. It is credible enough for local search-budget/skip-gate calibration.

## Result: gate-capture 4-episode diagnostic

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_generic_mcts_gate_capture_4ep/phase_001_train`

Status: successful observability check, rejected as a model candidate.

Observed:

- ONNX was up to date and reused.
- Finished cleanly with `exit=0`.
- Produced 3 parsed new rows before the tolerance stop, `2` wins.
- Orchestrator stdout captured:
  - `MCTS_GATE: total=725 sampler_null=0 fewcand=0 wrongtype=34 confident=9 not_tactical=0 activations=689`
- The run updated `model_latest.pt`; it was restored immediately from `pre_20260511_thesis_clean_generic_mcts_smoke_30m`.

Interpretation:

- The MCTS gate is not blocked by determinization or selective filtering.
- With `MCTS_SELECTIVE_ENABLE=0`, almost every eligible decision activates search.
- Most skipped gate decisions were wrong action type or high-confidence policy decisions.
- The next experiment should reduce cost generically, not add Spy-specific gates.

## Result: generic skip-threshold calibration

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_generic_mcts_skip70_12ep/phase_001_train`

Status: successful cost-calibration smoke, rejected as a model candidate.

Change from prior smoke:

- `MCTS_SKIP_TOP_PROB=0.70` instead of `0.95`.
- This is thesis-clean: it gates on policy confidence, not deck/card identity.

Observed:

- Finished cleanly with `exit=0`.
- Produced 11 parsed new rows, `3` wins.
- `MCTS_GATE: total=1835 sampler_null=0 fewcand=0 wrongtype=58 confident=125 not_tactical=0 activations=1693`
- `MCTS_STATS_FINAL calls=1693 avg_wallMs=215 avg_iters=2`
- Per-iteration cost remained dominated by evaluation: `eval=105308us`.
- The run updated `model_latest.pt`; it was restored immediately from `pre_20260511_thesis_clean_generic_mcts_smoke_30m`.

Interpretation:

- Lowering the skip threshold to `0.70` was not enough. Search still activated on about `92%` of MCTS-gated decisions.
- The expensive part is model evaluation inside MCTS, not clone/setup/walk overhead.
- A train-time MCTS run should not go to HPC until a generic search-frequency or leaf-evaluation batching change cuts cost by at least an order of magnitude.

Next local directions:

- Try a much stricter generic gate such as `MCTS_SKIP_TOP_PROB=0.40` or an entropy/disagreement gate.
- Batch MCTS leaf evaluations across active searches, if feasible.
- Consider root-only visit-target generation on a small generic decision subset rather than every ambiguous action.

## Result: stricter generic skip-threshold calibration

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_generic_mcts_skip40_8ep/phase_001_train`

Status: successful cost-calibration smoke, rejected as a model candidate.

Change from prior smoke:

- `MCTS_SKIP_TOP_PROB=0.40`.

Observed:

- Finished cleanly with `exit=0`.
- Produced 7 parsed new rows, `5` wins. This sample is too small and too noisy for quality conclusions.
- `MCTS_GATE: total=732 sampler_null=0 fewcand=0 wrongtype=20 confident=330 not_tactical=0 activations=401`
- `MCTS_STATS_FINAL calls=401 avg_wallMs=219 avg_iters=2`
- Per-iteration cost remained dominated by evaluation: `eval=107598us`.
- The run updated `model_latest.pt`; it was restored immediately from `pre_20260511_thesis_clean_generic_mcts_smoke_30m`.

Interpretation:

- `0.40` is the first threshold that materially cuts generic MCTS frequency: activations fell to about `55%` of MCTS-gated decisions.
- It is still far too expensive: roughly `57` searches per parsed episode at about `219 ms/search`, with 4 local runners.
- Threshold-only gating is unlikely to make AlphaZero-style train-time MCTS affordable by itself.
- Next high-EV local work is a generic gate based on decision type plus uncertainty, or batched leaf evaluation. HPC remains parked.

## Generic Sparse-Search Gate

Implemented `MCTS_TRAINING_SAMPLE_PROB` in `ComputerPlayerRL`.

- Default is `1.0`, preserving prior behavior.
- Applies only to train-time MCTS, not eval-time override.
- Randomly samples otherwise-eligible MCTS decisions after generic confidence gating.
- Adds `sampled_out` to `MCTS_GATE` diagnostics.
- This is thesis-clean: it does not inspect deck identity, action text, card names, or Spy-specific state.

Verification:

- `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`

## Result: sparse generic MCTS calibration

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_generic_mcts_skip40_sample10_12ep/phase_001_train`

Status: successful cost-calibration smoke, rejected as a model candidate.

Settings:

- `MCTS_SKIP_TOP_PROB=0.40`
- `MCTS_TRAINING_SAMPLE_PROB=0.10`

Observed:

- Finished cleanly with `exit=0`.
- Produced 11 parsed new rows, `1` win.
- `MCTS_GATE: total=1088 sampler_null=0 fewcand=0 wrongtype=66 confident=648 sampled_out=968 not_tactical=0 activations=44`
- `MCTS_STATS_FINAL calls=44 avg_wallMs=221 avg_iters=2`
- Per-iteration cost was still dominated by evaluation: `eval=108374us`.
- The run updated `model_latest.pt`; it was restored immediately from `pre_20260511_thesis_clean_generic_mcts_smoke_30m`.

## Result: Existing Parallel Rollout Path

Run:

```text
20260512_thesis_clean_mcts_parallel_smoke_8ep
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260512_thesis_clean_mcts_parallel_smoke_registry.json
```

Profile:

```text
Pauper-Spy-Combo-Value-MCTSParallelSmoke-20260512
```

Settings:

```text
MCTS_TRAINING_ENABLE=1
MULTI_PLY_MCTS=1
MCTS_ITERATIONS=4
MCTS_DETERMINIZATIONS=1
MCTS_MAX_OUR_ACTIONS=1
MCTS_PARALLEL_ROLLOUTS=4
MCTS_TIMEOUT_MS=750
MCTS_ITER_TIMEOUT_MS=250
MCTS_SKIP_TOP_PROB=0.40
MCTS_TRAINING_SAMPLE_PROB=0.10
MCTS_SELECTIVE_ENABLE=0
OPPONENT_SAMPLER=self
TOTAL_EPISODES=8
NUM_GAME_RUNNERS=2
```

Observed:

- Finished cleanly with `exit=0`.
- Produced 8 clone episodes.
- `MCTS_GATE: total=895 sampler_null=0 fewcand=0 wrongtype=74 confident=583 sampled_out=789 not_tactical=0 activations=31`
- `MCTS_STATS_FINAL calls=31 avg_wallMs=111 avg_iters=4 total_iters=124`
- Per-iteration cost remained value-eval dominated: `eval=108485us`.
- The important change is wall-time per search: the existing parallel path ran 4 iterations in about `111 ms`, compared with prior serial sparse smoke at 2 iterations in about `221 ms`.

Interpretation:

The leaf evaluator is still expensive, but parallel root-child rollout already gives useful batching/overlap. This reopens a small local MCTS training experiment: sparse generic MCTS with `MCTS_PARALLEL_ROLLOUTS=4` can now produce roughly 4x the search iterations per wall-clock search versus the serial path.

This is still not an HPC green light by itself. The next local step should train a clone for a modest episode delta with the parallel sparse-search settings and then reduced-gate it. If that fails CP1/CP7, stop. If it clears both, then consider larger confirmation and only then Zaratan.

## Result: Sparse Parallel MCTS Train-64

Run:

```text
20260512_thesis_clean_mcts_parallel_sparse_train64
```

Continuation:

- same clone as the 8-episode parallel smoke;
- trained to `TOTAL_EPISODES=64`;
- `NUM_GAME_RUNNERS=4`;
- same sparse parallel MCTS settings.

Observed:

- Finished cleanly with `exit=0`.
- `MCTS_GATE: total=6529 sampler_null=0 fewcand=0 wrongtype=220 confident=4108 sampled_out=5849 not_tactical=0 activations=252`
- `MCTS_STATS_FINAL calls=252 avg_wallMs=108 avg_iters=4 total_iters=1008`
- `model_latest.pt` SHA-256 after training: `13970D5ABC2A`

Reduced CP1 screen:

```text
20260512_mcts_parallel_sparse_train64_spy_cp1_unique_eval16
24/64 = 37.50%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 10 | 16 | 62.50% |
| Jund Wildfire | 9 | 16 | 56.25% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 2 | 16 | 12.50% |

Verdict: rejected. Parallelization makes the search cheaper enough to revisit locally, but the current teacher is still harmful. The bad CP1 result suggests the problem is search-target quality and/or forcing the MCTS-selected action during training, not just wall-clock cost.

Next local change: add a thesis-clean flag to collect MCTS visit targets without forcing the played action. That separates "search as auxiliary target" from "search as behavior policy" and should reveal whether forced poor MCTS actions are the damaging part.

## Implementation: Optional No-Force MCTS Training

Added:

```text
MCTS_TRAINING_FORCE_ACTION=0
```

Default remains `1`, preserving previous behavior. When disabled, train-time MCTS still records visit targets for the Python KL auxiliary, but the actual rollout action remains the policy-sampled action. This is thesis-clean because it gates only on a generic search-training behavior, not deck identity or card text.

Verification:

```text
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

## Result: Sparse Parallel MCTS Aux No-Force Train-64

Run:

```text
20260512_thesis_clean_mcts_aux_noforce_train64
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260512_thesis_clean_mcts_aux_noforce_registry.json
```

Settings:

```text
MCTS_TRAINING_ENABLE=1
MCTS_TRAINING_FORCE_ACTION=0
MULTI_PLY_MCTS=1
MCTS_ITERATIONS=4
MCTS_PARALLEL_ROLLOUTS=4
MCTS_SKIP_TOP_PROB=0.40
MCTS_TRAINING_SAMPLE_PROB=0.10
MCTS_KL_LOSS_COEF=1.0
TOTAL_EPISODES=64
```

Observed:

- Finished cleanly with `exit=0`.
- `MCTS_GATE: total=7137 sampler_null=0 fewcand=0 wrongtype=238 confident=4249 sampled_out=6373 not_tactical=0 activations=321`
- `MCTS_STATS_FINAL calls=321 avg_wallMs=107 avg_iters=4 total_iters=1284`
- `model_latest.pt` SHA-256 after training: `6BCD7E77EFF6`

Reduced CP1 screen:

```text
20260512_mcts_aux_noforce_train64_spy_cp1_unique_eval16
32/64 = 50.00%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 12 | 16 | 75.00% |
| Jund Wildfire | 8 | 16 | 50.00% |
| Mono Red Rally | 6 | 16 | 37.50% |
| Grixis Affinity | 6 | 16 | 37.50% |

Reduced CP7 screen:

```text
20260512_mcts_aux_noforce_train64_spy_cp7_unique_eval16
28/64 = 43.75%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 14 | 16 | 87.50% |
| Jund Wildfire | 8 | 16 | 50.00% |
| Mono Red Rally | 3 | 16 | 18.75% |
| Grixis Affinity | 3 | 16 | 18.75% |

Verdict: rejected, but informative. Disabling force-action fixed the catastrophic CP1 collapse seen in forced sparse MCTS (`24/64 -> 32/64`), so forcing low-quality MCTS choices into behavior was harmful. CP7 still fails, so the visit-target KL is not yet a useful hard-skill policy improver at coefficient `1.0`.

Next local branch: keep no-force and lower `MCTS_KL_LOSS_COEF` so search visits act as a weak auxiliary instead of a strong target.

## Result: Sparse Parallel MCTS Aux No-Force Coef25 Train-64

Run:

```text
20260512_thesis_clean_mcts_aux_noforce_coef25_train64
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260512_thesis_clean_mcts_aux_noforce_coef25_registry.json
```

Settings changed from the no-force branch:

```text
MCTS_KL_LOSS_COEF=0.25
```

Observed:

- Finished cleanly with `exit=0`.
- `MCTS_GATE: total=7600 sampler_null=0 fewcand=0 wrongtype=252 confident=4213 sampled_out=6860 not_tactical=0 activations=344`
- `MCTS_STATS_FINAL calls=344 avg_wallMs=108 avg_iters=4 total_iters=1376`

Reduced CP1 screen:

```text
20260512_mcts_aux_noforce_coef25_train64_spy_cp1_unique_eval16
31/64 = 48.44%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 12 | 16 | 75.00% |
| Jund Wildfire | 7 | 16 | 43.75% |
| Mono Red Rally | 8 | 16 | 50.00% |
| Grixis Affinity | 4 | 16 | 25.00% |

Verdict: rejected at CP1. Lowering the KL coefficient did not improve the local gate relative to coefficient `1.0`; it mainly traded Wildfire/Affinity down for Rally up. Stop coefficient cycling on this MCTS target.

## Result: Sparse Parallel MCTS Aux No-Force Train-256

Run:

```text
20260512_thesis_clean_mcts_aux_noforce_train256
```

This continued the coefficient-`1.0` no-force clone from episode `64` to episode `256`.

Observed:

- Finished cleanly with `exit=0`.
- `MCTS_GATE: total=22795 sampler_null=0 fewcand=0 wrongtype=780 confident=13316 sampled_out=20559 not_tactical=0 activations=938`
- `MCTS_STATS_FINAL calls=938 avg_wallMs=108 avg_iters=4 total_iters=3752`
- `model_latest.pt` SHA-256 after training: `C137505455BA`

Reduced CP1 screen:

```text
20260512_mcts_aux_noforce_train256_spy_cp1_unique_eval16
37/64 = 57.81%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 15 | 16 | 93.75% |
| Jund Wildfire | 8 | 16 | 50.00% |
| Mono Red Rally | 9 | 16 | 56.25% |
| Grixis Affinity | 5 | 16 | 31.25% |

Reduced CP7 screen:

```text
20260512_mcts_aux_noforce_train256_spy_cp7_unique_eval16
28/64 = 43.75%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 7 | 16 | 43.75% |
| Mono Red Rally | 5 | 16 | 31.25% |
| Grixis Affinity | 3 | 16 | 18.75% |

Verdict: rejected. Longer no-force sparse MCTS training improves CP1 but remains capped at the CP7 gate. This branch is not HPC-ready.

Interpretation: pure self-play is not training the hard pressure matchups that define CP7 failure. If MCTS auxiliary training is revisited, the next local test should use the accepted pressure/opponent curriculum instead of self-play only. Otherwise, move on to stronger target generation rather than more MCTS coefficient cycling.

## Result: Sparse Parallel MCTS Aux No-Force Hardmix Train-128

Run:

```text
20260512_thesis_clean_mcts_aux_noforce_hardmix_train128
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260512_thesis_clean_mcts_aux_noforce_hardmix_registry.json
```

Settings changed from the no-force branch:

```text
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.50
SKILL_MIX=1:0.25,3:0.25,7:0.50
MCTS_TRAINING_FORCE_ACTION=0
MCTS_ITERATIONS=4
MCTS_PARALLEL_ROLLOUTS=4
MCTS_SKIP_TOP_PROB=0.40
MCTS_TRAINING_SAMPLE_PROB=0.10
```

Observed:

- Finished cleanly with `exit=0`.
- `MCTS_GATE: total=9833 sampler_null=0 fewcand=0 wrongtype=480 confident=5884 sampled_out=8909 not_tactical=0 activations=377`
- `MCTS_STATS_FINAL calls=377 avg_wallMs=109 avg_iters=4 total_iters=1508`

Reduced CP7 screen:

```text
20260512_mcts_aux_noforce_hardmix_train128_spy_cp7_unique_eval16
25/64 = 39.06%
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 13 | 16 | 81.25% |
| Jund Wildfire | 5 | 16 | 31.25% |
| Mono Red Rally | 4 | 16 | 25.00% |
| Grixis Affinity | 3 | 16 | 18.75% |

Verdict: rejected at CP7. Adding hard-skill pressure to the no-force MCTS auxiliary branch did not fix the hard-matchup failure and remains below the accepted checkpoint. Do not spend HPC on this MCTS target family.

Interpretation:

- Sparse search reduces MCTS calls by the needed order of magnitude: from hundreds/thousands per tiny smoke to `44` calls over 11 parsed games.
- This makes a longer local branch reasonable.
- It is still not enough evidence for HPC. Next gate is a 30-60 minute local sparse-MCTS continuation from the accepted checkpoint, followed by a reduced CP1 sanity eval.

## Eval-Time Multi-Ply Quality Check

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260511_accepted_generic_multipy_mcts_cp7_eval4`

Purpose: before spending more training or HPC budget on search-as-policy-improvement, test whether the current generic multi-ply MCTS can improve the accepted checkpoint as an eval-time action selector.

Settings:

- accepted Affinity-pressure checkpoint
- CP7, 4 games per matchup
- `ISMCTS_ENABLE=1`
- `MULTI_PLY_MCTS=1`
- `MCTS_ITERATIONS=8`
- `MCTS_DETERMINIZATIONS=1`
- `MCTS_MAX_OUR_ACTIONS=0`
- `MCTS_SKIP_TOP_PROB=0.40`
- `MCTS_TIMEOUT_MS=500`
- no selective keywords

Result: `4/16 = 25.00%`

| Opponent | Wins | Games | MCTS Activations |
| --- | ---: | ---: | ---: |
| Spy Combo | 2 | 4 | 276 |
| Jund Wildfire | 1 | 4 | 206 |
| Mono Red Rally | 1 | 4 | 61 |
| Grixis Affinity | 0 | 4 | 143 |

Decision: do not scale this MCTS configuration.

It is both slow and weaker than the accepted policy. The current generic multi-ply MCTS target is not yet a useful policy-improvement oracle for Spy, so HPC should remain parked for this path until search quality improves.

## Value-Based Final Selection Probe

Patch:

- added `MCTSNode.bestActionByMeanValue()`;
- added opt-in `MCTS_FINAL_SELECTION=value` for `MultiPlyMCTS`;
- default remains `MCTS_FINAL_SELECTION=visits`, preserving existing behavior.

Verification:

- `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`

Run: `local-training/local_pbt/cp7_eval_sweeps_cdrive/20260511_accepted_generic_multipy_mcts_valuepick_cp7_eval4`

Settings: same as the eval-time multi-ply quality check, plus `MCTS_FINAL_SELECTION=value`.

Result: `5/16 = 31.25%`

| Opponent | Wins | Games | MCTS Activations |
| --- | ---: | ---: | ---: |
| Spy Combo | 3 | 4 | 159 |
| Jund Wildfire | 1 | 4 | 136 |
| Mono Red Rally | 1 | 4 | 61 |
| Grixis Affinity | 0 | 4 | 111 |

Decision: keep the opt-in plumbing, but do not train with it yet.

Choosing by mean value is a reasonable generic option and slightly improved the tiny diagnostic (`4/16` to `5/16`), but it is still far below the accepted policy and still cannot beat Affinity. The limiting issue is not only final action selection; the current value-guided search is not producing a reliable policy-improvement signal.

## Flat Policy-Rollout MCTS Probe

Run:

```text
20260512_accepted_flat_policyrollout_mcts_cp7_eval4
```

Purpose: test the non-multipy `PolicyValueMCTS` path with policy-driven truncated rollouts before doing any more train-time search work.

Settings:

```text
accepted Pauper-Spy-Combo-Value checkpoint
CP7, 4 games per matchup
ISMCTS_ENABLE=1
MULTI_PLY_MCTS=0
MCTS_ITERATIONS=4
MCTS_DETERMINIZATIONS=1
MCTS_ROLLOUT_DEPTH=3
MCTS_PARALLEL_ROLLOUTS=4
MCTS_SKIP_TOP_PROB=0.40
MCTS_SELECTIVE_ENABLE=0
```

Result:

```text
5/16 = 31.25%
MCTS activations: 549
```

| Opponent | Wins | Games | MCTS Activations |
| --- | ---: | ---: | ---: |
| Spy Combo | 2 | 4 | 142 |
| Jund Wildfire | 2 | 4 | 167 |
| Mono Red Rally | 1 | 4 | 130 |
| Grixis Affinity | 0 | 4 | 110 |

Decision: rejected.

Policy-driven truncated rollouts are not a better eval-time oracle at this budget. The search is expensive, fires often, and still cannot beat Affinity. Do not train or scale this flat policy-rollout MCTS target.
