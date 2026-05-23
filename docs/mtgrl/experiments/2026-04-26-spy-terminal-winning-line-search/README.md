# Spy Terminal-Winning Line Search

Date: 2026-04-26

## Question

Can terminal-only search find actual Spy Combo winning prefixes against CP7 and train useful action choices without using all-loss tiebreak labels?

This was the follow-up to the negative all-loss action counterfactual work. The change was to train only from full-game terminal wins discovered by search.

## Implementation

Extended `ActionCounterfactualTrainer` with `--winning-prefix-mode`.

Behavior:

- Run breadth-first forced-prefix search from sampled Spy seeds.
- Replay full games from fixed deck order.
- Expand top policy candidates at each eligible decision.
- Train only when the forced prefix leads to an actual terminal win.
- Convert the winning prefix into one-hot `mctsVisitTargets`.
- Keep reward terminal-only; no step reward shaping.

Added wrapper:

- `scripts/run_spy_line_search.ps1`

Important filters added after smoke:

- `--skip-pass-training`: do not train `Pass` labels from winning prefixes.
- `--skip-blank-training`: let search use target decisions, but do not train blank candidate-text labels.

## Profile

Created:

- `Pauper-Spy-Combo-LineSearch-v1`

Initialized from:

- `Pauper-Generalist-Value-v2/models/snapshots/snapshot_step_5000.pt`

The profile was reset to that checkpoint before each major variant.

## Smoke

Run:

- `20260426_spy_line_search_smoke_cp7_s8_d3_n7`

Shape:

- CP7 skill 7
- 8 scenarios
- 8 workers
- max prefix depth 3
- max search nodes 7

Result:

- `1/8` scenarios produced a terminal-winning prefix.
- 3 training examples were generated.
- This proved the mechanism works, but one label was `Pass`, so subsequent runs enabled pass filtering.

## Variant 1: All Types, Pass Filter

Run:

- `20260426_spy_line_search_cp7_s24_d4_n15_p16_skippass`

Shape:

- 24 scenarios
- 16 workers
- max prefix depth 4
- max search nodes 15
- CP7 skill 7
- `--skip-pass-training`

Result:

- `8/24` trained scenarios.
- 14 examples.
- Label mix included useful actions, but also 6 blank target-selection labels.
- Reduced CP7 eval after this pass: `0/16`.

Mulligan probe remained acceptable:

- Spy effective lands 0: keep `43.5%`
- effective lands 1: keep `66.4%`
- effective lands 2: keep `92.0%`

## Throughput Fix

The first scaled searches underutilized the machine even with many scenario workers.

Thread dump showed:

- `ACTION-CF-*` workers mostly waiting on game threads.
- Game threads waiting on the AI simulation pool.
- Only about five `AI-SIM-MAD` threads doing real work.

Fix:

- Added `-AiThreads` to `scripts/run_spy_line_search.ps1`.
- Set `AI_MAX_THREADS_FOR_SIMULATIONS=24`.

Result:

- Search CPU utilization rose to about `100%` during the main phase.
- Tail utilization still falls when only a few long scenarios remain.
- Next structural throughput improvement is branch-node parallelism instead of scenario-level parallelism.

## Variant 2: Action Only

Run:

- `20260426_spy_line_search_actiononly_cp7_s48_d6_n31_p24_ai24_e8`

Shape:

- 48 scenarios
- 24 scenario workers
- AI sim threads 24
- max prefix depth 6
- max search nodes 31
- action type restricted to `ACTIVATE_ABILITY_OR_SPELL`
- pass labels filtered
- 8 train epochs

Result:

- `7/48` trained scenarios.
- 8 clean action examples.
- 64 train-pass samples.
- Label mix:
  - `Cast Land Grant`: 5
  - `Play Forest`: 2
  - `Swampcycling`: 1
- Reduced CP7 eval: `0/16`.

Mulligan probe:

- Spy effective lands 0: keep `35.9%`
- effective lands 1: keep `70.0%`
- effective lands 2: keep `83.0%`

Conclusion:

- Labels were clean, but search hit rate was too low.

## Variant 3: All Types, Clean Training Labels

Run:

- `20260426_spy_line_search_alltypes_cp7_s48_d6_n31_ai24_e8_cleanlabels`

Shape:

- 48 scenarios
- 24 scenario workers
- AI sim threads 24
- max prefix depth 6
- max search nodes 31
- all action/target/card-use types available for search
- `--skip-pass-training`
- `--skip-blank-training`
- 8 train epochs

Result:

- `13/48` trained scenarios.
- 13 clean examples.
- 104 train-pass samples.
- Prefix records: 21 terminal-winning searched nodes, 333 non-winning nodes.
- Label mix:
  - `Play Forest`: 6
  - `Cast Land Grant`: 6
  - `Cast Lotus Petal`: 1
- Reduced CP7 eval: `0/16`.

Mulligan probe:

- Spy effective lands 0: keep `45.8%`
- effective lands 1: keep `63.8%`
- effective lands 2: keep `92.7%`

## Replay Gate And Memorization Stress

Added:

- `--replay-file` / `--replay-max-scenarios` to `ActionCounterfactualTrainer`.
- `scripts/run_spy_line_replay_probe.ps1`.
- Replay CSV now includes expected and actual action type so target-head drift is visible.
- `scripts/run_spy_line_search.ps1` now persists `action_training_samples.csv` before enqueueing training, disables train-queue drops, and sizes `TRAIN_QUEUE_MAX_EPISODES` for the planned distillation volume.

Initial replay probe on Variant 3:

- Run: `20260426_spy_line_search_alltypes_clean_replay_probe`
- Result: `0/13` exact action matches.
- Scenario wins: `10/13`.

High-strength memorization attempt:

- Run: `20260426_spy_line_search_alltypes_cp7_s38_d6_n31_ai24_w38_e80_kl10_nodrop`
- Reset from `snapshot_step_5000.pt`.
- 38 scenarios, 38 scenario workers, 24 AI sim threads.
- Max prefix depth 6, max search nodes 31.
- 80 distillation epochs, `MCTS_KL_LOSS_COEF=10.0`.
- Pass and blank labels filtered.
- No train-queue dropping.

Result:

- `17/38` trained scenarios.
- 94 selected labels.
- 7,520 train-pass samples.
- Elapsed `1989.0s`.
- Model saved cleanly at `2026-04-26 23:21`.

Throughput notes:

- 48 workers starts all scenarios immediately and reaches training faster than 24 workers, but a few long-tail scenarios can still waste wall-clock after label yield stops.
- Reducing this run to 38 scenarios avoided the no-label tail.
- Search phase reached full CPU.
- Training phase was correct but inefficient: Java/Py4J serialization into `trainMulti` dominated, and GPU utilization came in bursts rather than staying saturated.
- The first high-epoch attempt was invalid because the Java train queue dropped samples when full. This is now fixed in the line-search wrapper with `TRAIN_QUEUE_DROP_ON_FULL=0`.

Replay after high-strength distillation:

- Run: `20260426_spy_line_search_s38_e80_nodrop_replay_probe`
- Exact row matches: `6/94` (`6.38%`).
- Index matches: `39/94`.
- Text matches: `7/94`.
- Action-type mismatches: `15/94`.
- Scenario wins: `13/17`.

Reduced CP7 eval after high-strength distillation:

- Run: `20260426_spy_line_search_s38_e80_nodrop_cp7_g4_p8`
- Overall: `0/16`.
- Spy vs Spy: `0/4`.
- Spy vs Jund Wildfire: `0/4`.
- Spy vs Mono Red Rally: `0/4`.
- Spy vs Grixis Affinity: `0/4`.

## Conclusion

The line-search direction is mechanically valid and better aligned with the project goal than all-loss action counterfactual labels:

- It finds actual terminal wins against CP7.
- It produces plausible Spy labels like `Cast Land Grant` and `Play Forest`.
- It can run at full CPU utilization once `AI_MAX_THREADS_FOR_SIMULATIONS` is set correctly.
- It does not obviously corrupt Spy mulligans in these small passes.

It did not yet improve reduced CP7 eval:

- All evaluated variants remained `0/16`.
- The replay gate disproves the earlier "just scale data volume" hypothesis for this implementation.
- Even 94 labels repeated for 7,520 train-pass samples only reached `6/94` exact replay and stayed `0/16` in CP7 eval.
- Many replay rows have the same candidate index but different candidate text after the first divergent action, so game replay is a noisy diagnostic. Still, exact first-line reproduction is far too weak.

## Next Step

Do not go back to all-loss one-ply labels.

Do not scale to 256-512 scenarios yet. The next experiment should isolate whether the action-head target plumbing can overfit the captured labels at all.

Recommended next experiment:

1. Add an in-memory tensor replay gate inside `ActionCounterfactualTrainer`:
   - after distillation drains, score the exact retained `TrainingData` tensors;
   - report top-1 index accuracy and target probability on the exact tensors used for training;
   - this removes game-state drift and tests only candidate-head learning.
2. Add forced-prefix replay:
   - for each expected row, force all earlier expected prefix actions and then score the current decision;
   - this distinguishes target-head failure from cascading game divergence.
3. If tensor accuracy is low:
   - debug `mctsVisitTargets` loss application, action-type head routing, and masking.
4. If tensor accuracy is high but forced-prefix replay is low:
   - debug candidate identity/candidate ordering and action-type capture.
5. Only after tensor and forced-prefix replay pass should we scale scenario count.

The immediate engineering fixes from this experiment are complete:

- set `AI_MAX_THREADS_FOR_SIMULATIONS` explicitly for search runs;
- persist selected labels before training;
- disable train-queue dropping for line-search distillation.

The remaining blocking question is whether the model can overfit and replay its own searched labels under controlled conditions.
