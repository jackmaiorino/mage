# Spy Separate-Critic Value RL

Date: 2026-05-08

Profile: `Pauper-Spy-Combo-FastT5SeparateCriticRL-20260508`

Source checkpoint: `Pauper-Spy-Combo-FastT5Contrast-20260506`

## Question

The accepted FastT5 Spy checkpoint had a useful action policy but an unhealthy value head. The model also had a separate critic encoder, but the active value scoring path was still using the shared policy CLS representation.

This experiment asks whether using the separate critic encoder, resetting the critic head, and running terminal-only RL can create a value signal strong enough to improve Spy Combo action timing.

## Code Changes

- `mtg_transformer.py`
  - Added `VALUE_USE_SEPARATE_CRITIC_ENCODER=1`.
  - `score_candidates`, `_process_value`, and the ONNX `SingleHeadScorer` now use `encode_state_critic(...)` for value scoring when enabled.
- `py4j_entry_point.py`
  - Added `RESET_TRAINING_STATE_ON_LOAD=1` so a seeded run can intentionally reset counters and actually enter warmup.
- `scripts/run_local_pbt.py`
  - Added a Windows retry loop around ONNX staging directory rename to avoid transient export failures.

## Training

Run: `local-training/local_pbt/value_rl/20260508_spy_fastt5_separate_critic_resume2_r24_b4`

Registry: `local-training/local_pbt/value_rl/20260508_spy_fastt5_separate_critic_registry.json`

Training settings:

| Setting | Value |
| --- | --- |
| Episodes | 10,000 |
| Reward | Terminal only |
| Opponent mix | 70% self-play, 30% CP1 |
| Search | None |
| Returns | Monte Carlo first, GAE auto-gated |
| Critic warmup | 1,000 steps |
| Warmup policy loss | 0 |
| Warmup value loss | 20 |
| Main policy loss | 0.5 |
| Main value loss | 5 |
| Actor LR | `5e-5` |
| Critic LR | `5e-4` |
| Local throughput | About 1.7-3.0 eps/s late in run |

Training completed cleanly at 10,000 generated episodes. Final ONNX export: `v20260508T033732_910725`.

The in-training `value_accuracy.csv` still reported `0.0000` near the end, but the raw average values separated slightly: final average win value `0.0555`, final average loss value `0.0399`.

## CP1 Eval

Full eval run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_fastt5_separate_critic_10k_cp1_eval32`

One Wildfire game crashed inside CP7 (`ComputerPlayer7.calculateActions` on an empty action list), so the denominator is 127 instead of 128.

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 21 | 32 | 65.62% |
| Jund Wildfire | 7 | 31 | 22.58% |
| Mono Red Rally | 4 | 32 | 12.50% |
| Grixis Affinity | 6 | 32 | 18.75% |
| Total | 38 | 127 | 29.92% |

Comparison:

| Checkpoint | CP1 Total |
| --- | ---: |
| Accepted FastT5Contrast | 33/128 |
| Value-RL hybrid 10k | 37/128 |
| Separate-critic value RL 10k | 38/127 |

This is not a meaningful promotion. It is a small total improvement, but most of it comes from Spy mirror strength while fast external matchups remain poor.

## Logged Diagnostic

Logged eval run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_fastt5_separate_critic_10k_cp1_logged8`

Analysis root: `local-training/analysis/value_head_20260508_separate_critic_logged8`

Small logged result:

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 4 | 8 | 50.00% |
| Jund Wildfire | 1 | 8 | 12.50% |
| Mono Red Rally | 0 | 8 | 0.00% |
| Grixis Affinity | 1 | 8 | 12.50% |
| Total | 6 | 32 | 18.75% |

### Value Head

The separate critic did improve logged value ranking:

| Subset | Rows | Base Winrate | AUC | Mean Win Score | Mean Loss Score | Score Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All decisions | 4,270 | 18.69% | 0.586 | 0.109 | 0.092 | +0.018 |
| Nontrivial options | 2,389 | 20.68% | 0.646 | 0.177 | 0.168 | +0.009 |
| Action-head-like | 4,044 | 18.52% | 0.582 | 0.105 | 0.087 | +0.017 |
| Critical action option | 75 | 30.67% | 0.727 | 0.183 | 0.172 | +0.011 |
| Critical combo | 111 | 36.94% | 0.761 | 0.188 | 0.176 | +0.012 |
| Per-game mean value | 32 | 18.75% | 0.731 | 0.108 | 0.081 | +0.027 |

This is a real improvement over the prior value-RL checkpoint, whose logged all-decision AUC was `0.479` and per-game mean AUC was `0.411`.

### Action Health

The policy still fails at the core Spy Combo decisions:

| Metric | Result |
| --- | ---: |
| Mulligan keep rate | 32/32 |
| Kept hands with 0-1 land | 28/32 |
| Land play selected | 173/178 |
| Pass over land | 0/178 |
| Spy cast opportunities | 48 |
| Spy casts | 18/48 |
| No-hidden-land Spy opportunities | 4/48 |
| Spy casts with no hidden lands | 1/18 |
| Spy casts with hidden lands | 17/18 |
| Dread Return flashbacks | 9 |
| Premature non-combo Dread Returns | 9/9 |

Interpretation:

- The old pass-over-land issue is no longer the blocker.
- Mulligans remain structurally broken. The model keeps every hand, including many 0-1 land hands.
- Spy timing remains broken. Almost every Spy cast still happens while real lands are hidden in the library.
- Dread Return timing remains broken. Every logged flashback was non-combo and premature.
- The improved critic is visible in logged values, but the policy does not use that value signal to avoid bad committed actions.

## Conclusion

The separate critic encoder helps the value head. It does not, by itself, make Spy Combo play correctly.

The current architecture can learn a value ranking over whole games and critical combo states, but the action selector still behaves like a high-confidence policy head that ignores action-conditioned value when the tactical choice is "wait, develop, or do not fire the combo yet."

This result narrows the next step. More one-off BC variants are low value. More terminal-only RL without changing how value affects action choice is also unlikely to fix the exact failure mode quickly.

## Next Experiment

Build an explicit value-gated action selector for Spy-critical action families, then evaluate it without another long training run.

Scope:

- Keep terminal-only training intact.
- Do not add heuristic step rewards.
- Add an eval-time flag that reranks only a narrow set of critical choices using the model's own value head:
  - `Balustrade Spy` cast decisions.
  - `Dread Return` flashback decisions.
  - `Dread Return` target decisions.
  - Mulligan keep/mulligan decisions if the value head emits meaningful scores for them.
- Compare baseline policy choice versus value-gated choice in logged CP1 eval.

Success criteria:

- CP1 total improves materially over `38/127`, or
- Logged action health improves even if total is noisy:
  - Fewer hidden-land Spy casts.
  - Fewer premature Dread Returns.
  - Better no-hidden-land Spy conversion.
  - Lower keep rate for clearly dead Spy hands.

If value gating improves action health, the long-term path is train-time MCTS or policy improvement targets generated from the value-gated selector. If value gating cannot improve action health despite better critic AUC, the likely blocker is state/action representation rather than RL algorithm choice.
