# Thesis-Clean Card-Belief Infrastructure

Date: 2026-05-12

## Thesis Boundary

This experiment keeps the morphed thesis line: generic RL/belief machinery may learn Spy, but training must not name Spy's combo pieces or add Spy-specific rewards.

Allowed here:

- Terminal outcome remains the only gameplay reward.
- Card-belief targets are generic hidden-zone labels: normalized opponent hand+library counts over the active deck-pool vocabulary.
- The same label construction applies to every deck in the pool.

Still disabled:

- `RL_SPY_COMBO_CANDIDATE_FEATURES_ENABLE`
- heuristic step rewards
- Spy-specific terminal modes
- action-text/card-name regex labels
- selective MCTS keyword gates

## Implementation

Java:

- `StateSequenceBuilder`
  - Added opt-in `RL_CARD_BELIEF_LABELS_ENABLE`.
  - Builds a deterministic deck-pool vocabulary from `RL_CARD_BELIEF_DECK_LIST` or `DECK_LIST_FILE`.
  - Exposes `cardBeliefDim()`, `cardBeliefVocab()`, and `cardBeliefMaxCounts()`.
  - Computes hidden-card labels from opponent hand+library only.
- `ComputerPlayerRL`
  - Records card-belief labels centrally alongside archetype labels.
  - Added `RL_CARD_BELIEF_DETERMINIZATION_ENABLE`.
  - Installs a per-decision thread-local card-belief context for MCTS determinization.
  - Added `card_belief_dets` to MCTS gate stats.
- `SharedGpuTensorSerde`, `PythonMLBatchManager`, `PythonEntryPoint`, `SharedGpuPythonModel`, and shared GPU protocol
  - Added optional card-belief label transport and card-belief inference.
- `DeterminizationSampler`
  - Added thread-local card-belief support.
  - When enabled, samples the usual public-card archetype posterior, then uses card-belief predictions to bias the hidden hand/library ordering from the remaining canonical deck pool.

Python:

- `mtg_transformer.py`
  - Added opt-in `card_belief_head` when `CARD_BELIEF_DIM > 0`.
- `py4j_entry_point.py`
  - Added `CARD_BELIEF_LOSS_COEF`.
  - Added `CARD_BELIEF_HEAD_ONLY`.
  - Uses MSE on sigmoid card-belief logits vs normalized hidden-count labels.
- `gpu_service_core.py` / `gpu_service_host.py`
  - Added shared-GPU card-belief prediction opcode.

## Verification

Commands:

```powershell
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
.\\.mtgrl_venv\\Scripts\\python.exe -m py_compile Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/mtg_transformer.py Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_core.py Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_host.py
```

Both passed after the final wiring.

## Smoke Training

Profile:

- `Pauper-Spy-Combo-Value-CardBeliefHeadOnly-20260512`

Registry:

- `local-training/local_pbt/thesis_clean/20260512_thesis_clean_card_belief_headonly_registry.json`

Settings:

- `CARD_BELIEF_DIM=63`
- `RL_CARD_BELIEF_LABELS_ENABLE=1`
- `CARD_BELIEF_HEAD_ONLY=1`
- `CARD_BELIEF_LOSS_COEF=1.0`
- policy/value/entropy/MCTS losses disabled
- 16 local episodes

Result:

- Training completed with `rc=0`.
- `model_latest.pt` changed to hash prefix `0AC946054D7D`.
- `model.pt` remained the accepted Spy base hash prefix `14E1A1DA4F3E`.
- Training loss total was card-belief loss only under this configuration.

## MCTS Determinization Smoke

Registry:

- `local-training/local_pbt/thesis_clean/20260512_thesis_clean_card_belief_mcts_smoke_eval_registry.json`

Run:

- `local-training/local_pbt/cp7_eval_sweeps/20260512_card_belief_mcts_smoke_eval1_counter`
- One CP1 game, Spy vs Jund Wildfire.
- `--mcts --mcts-iterations 1 --mcts-determinizations 1 --mcts-skip-top-prob 1.01`

Result:

- `0/1`, not a performance result.
- `mcts_activations=24`
- `card_belief_dets=24`

This proves the card-belief determinization path is active when enabled.

## Calibration Continuation

Continuation:

- Same profile: `Pauper-Spy-Combo-Value-CardBeliefHeadOnly-20260512`
- Two local 256-episode continuations, reaching episode 528.
- Still head-only: policy/value/entropy/search losses disabled.

Loss trend from `training_losses.csv` (`total_loss` is card-belief loss in this setup):

- rows 000-025: avg 0.3476
- rows 025-075: avg 0.3573
- rows 075-125: avg 0.2732
- rows 125-175: avg 0.2282
- rows 175-225: avg 0.2176
- rows 225-230: avg 0.1685

Read: the generic card-belief head is learning on local data. This clears the local plumbing/calibration gate, but it does not by itself improve the policy because the policy/value path was intentionally frozen.

Latest clone hashes after calibration:

- `model.pt`: `CF62C47728E3`
- `model_latest.pt`: `1575105F0831`

Accepted baseline profiles were hash-checked after the run and remained unchanged.

## Bounded Eval

No-MCTS sanity eval:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260513_cardbelief528_nomcts_eval4`
- Result: 10/16 = 62.5%
- Matchups:
  - Spy mirror: 4/4
  - Jund Wildfire: 2/4
  - Mono Red Rally: 2/4
  - Grixis Affinity: 2/4

This confirms head-only card-belief training did not damage the policy.

Paired tiny-MCTS eval:

- Standard determinization: `20260513_cardbelief528_standard_mcts_eval2`
  - 0/8, `card_belief_dets=0`
- Card-belief determinization: `20260513_cardbelief528_cardbelief_mcts_eval2`
  - 0/8, `card_belief_dets` matched MCTS activations in each job

Read: card-belief determinization is wired and active, but the one-iteration forced MCTS setting is destructive regardless of determinization source. This is a negative result for tiny forced eval-time MCTS, not for the belief head.

## Read

The infrastructure is now viable. This is not yet evidence that card-belief improves play; the trained head has only 16 smoke episodes and the MCTS smoke was one game. The result is a plumbing gate: generic hidden-card labels train, save, reload, infer, and drive determinization without Spy-specific teaching.

## Next Experiment

Since the head learned and forced tiny-MCTS failed, the next thesis-clean experiment should use card belief as an auxiliary representation objective during ordinary terminal-only RL:

- Start from accepted Spy policy plus the calibrated card-belief head.
- Unfreeze policy/value/encoder.
- Keep `RL_CARD_BELIEF_LABELS_ENABLE=1` and `CARD_BELIEF_LOSS_COEF` small, e.g. 0.05-0.10.
- Keep MCTS disabled.
- Run a local 128-256 episode hardmix continuation.
- Eval no-MCTS first. Do not spend HPC unless this local auxiliary-RL branch beats the accepted baseline or at least preserves policy while improving downstream belief metrics.

## Auxiliary RL Continuation

Profile:

- `Pauper-Spy-Combo-Value-CardBeliefAuxRL-20260513`

Registry:

- `local-training/local_pbt/thesis_clean/20260513_thesis_clean_card_belief_auxrl_registry.json`

Settings:

- Started from calibrated `Pauper-Spy-Combo-Value-CardBeliefHeadOnly-20260512` `model_latest.pt` hash `1575105F0831`.
- `CARD_BELIEF_HEAD_ONLY=0`
- `CARD_BELIEF_LOSS_COEF=0.05`
- `MCTS_TRAINING_ENABLE=0`
- `ISMCTS_ENABLE=0`
- `RL_CARD_BELIEF_DETERMINIZATION_ENABLE=0`
- No heuristic rewards or Spy-specific action features.

Local +128 episodes:

- Reached episode 128.
- `model_latest.pt` hash `D4AD6ECE89E2`.
- Eval `20260513_cardbelief_auxrl128_nomcts_eval4`: 10/16 = 62.5%
  - Grixis Affinity: 1/4
  - Jund Wildfire: 4/4
  - Mono Red Rally: 2/4
  - Spy mirror: 3/4

Local +128 more episodes:

- Reached episode 256.
- `model_latest.pt` hash `83AD54F91E2C`.
- Eval `20260513_cardbelief_auxrl256_nomcts_eval4`: 4/16 = 25.0%
  - Grixis Affinity: 1/4
  - Jund Wildfire: 1/4
  - Mono Red Rally: 1/4
  - Spy mirror: 1/4

Read: the first small continuation preserved gameplay, but the second tranche regressed sharply. Treat this branch as unstable and do not spend HPC on it. The next branch should restart from the calibrated head-only checkpoint with a more conservative RL update: lower actor/critic/encoder learning rates, smaller PPO epsilon, lower auxiliary coefficient, and the same no-MCTS/no-Spy-feature thesis boundary.

## Conservative Auxiliary RL

Profile:

- `Pauper-Spy-Combo-Value-CardBeliefAuxRLConservative-20260513`

Settings differed from the unstable branch by lowering the card-belief coefficient and PPO step pressure:

- `CARD_BELIEF_LOSS_COEF=0.01`
- `PPO_EPSILON=0.05`
- `ACTOR_LR=5e-5`
- `CRITIC_LR=1e-4`
- `OTHER_LR=2e-5`
- `VALUE_LOSS_COEF=2.0`
- `ENTROPY_LOSS_MULT=0.02`

Result:

- Reached episode 128.
- `model_latest.pt` hash `F5B7FD325CE1`.
- Eval `20260513_cardbelief_auxrlconservative128_nomcts_eval4`: 7/16 = 43.75%
  - Grixis Affinity: 0/4
  - Jund Wildfire: 2/4
  - Mono Red Rally: 1/4
  - Spy mirror: 4/4

Read: the conservative settings avoided the 25% collapse but did not improve the accepted line and still failed the Affinity/Rally pressure gate.

## Reference-Anchored Auxiliary RL

Patch:

- Added default-off `REFERENCE_POLICY_KL_COEF`.
- Reuses the existing frozen reference model loader via `MCTS_REFERENCE_MODEL_PATH`.
- When enabled, the learner adds a generic cross-entropy/KL anchor from the frozen reference policy distribution to the current policy distribution over the same legal candidates.
- This is thesis-clean: no action text, card names, Spy terminal mode, Spy hand pool, or selective search gate.

Profile:

- `Pauper-Spy-Combo-Value-CardBeliefAuxRLAnchored-20260513`

Settings:

- Started from calibrated `Pauper-Spy-Combo-Value-CardBeliefHeadOnly-20260512` `model_latest.pt` hash `1575105F0831`.
- Same conservative RL settings as above.
- `REFERENCE_POLICY_KL_COEF=0.50`
- `MCTS_REFERENCE_MODEL_PATH=.../Pauper-Spy-Combo-Value-CardBeliefHeadOnly-20260512/models/model_latest.pt`

Local +128 episodes:

- Reached episode 128.
- `model_latest.pt` hash `559C4A8D5E4A`.
- Saved safety copy `model_latest_ep128.pt` with the same hash before extending.
- Eval `20260513_cardbelief_auxrlanchored128_nomcts_eval4`: 11/16 = 68.75%
  - Grixis Affinity: 3/4
  - Jund Wildfire: 3/4
  - Mono Red Rally: 1/4
  - Spy mirror: 4/4

Read: this is the best local card-belief auxiliary result so far, but the sample is tiny and Rally remains weak. Continue locally before any HPC. The next tranche should confirm the reference-policy anchor is actually loading in stdout and test whether the branch survives another +128 without repeating the unanchored regression.

Local +128 more episodes:

- Confirmed the frozen reference policy loaded in the shared GPU log with `coef=0.5000`.
- Reached episode 256.
- `model_latest.pt` hash `EE0A8612F2B6`.
- Saved safety copy `model_latest_ep256.pt` with the same hash.
- Eval `20260513_cardbelief_auxrlanchored256_nomcts_eval4`: 5/16 = 31.25%
  - Grixis Affinity: 1/4
  - Jund Wildfire: 0/4
  - Mono Red Rally: 1/4
  - Spy mirror: 3/4

After this regression, restored `model_latest.pt` to the episode-128 safety copy:

- `model_latest.pt` hash `559C4A8D5E4A`
- `model_latest_ep128.pt` hash `559C4A8D5E4A`
- `model_latest_ep256.pt` hash `EE0A8612F2B6`

Larger CP1 screen on the restored episode-128 checkpoint:

- Eval `20260513_cardbelief_auxrlanchored128_nomcts_eval16`
- Skill: CP1
- Result: 40/64 = 62.5%
  - Grixis Affinity: 4/16 = 25.0%
  - Jund Wildfire: 11/16 = 68.75%
  - Mono Red Rally: 9/16 = 56.25%
  - Spy mirror: 16/16 = 100.0%

Partial CP7 screen on the restored episode-128 checkpoint:

- Eval `20260513_cardbelief_auxrlanchored128_cp7_eval16`
- Skill: CP7
- Completed before abort/interruption: 34/64 jobs, 14/33 recorded wins = 42.42%
  - Spy mirror: 9/16 = 56.25%
  - Jund Wildfire: 3/14 = 21.43%
  - Mono Red Rally: 2/3 = 66.67%
  - Grixis Affinity: no completed games in the partial result

Read: the restored episode-128 branch is useful diagnostically but not promotable. The CP1 result was solid, but CP7 failed the hard matchups early: Jund could reach only 5/16 even if the remaining two games were wins, well below the accepted CP7 Jund reference of 36/60 = 60.0%, and the mirror result also regressed from the accepted 43/56 = 76.79%. Do not spend HPC on this branch.

## Non-Forced Belief-MCTS Probes

Question:

Would card-belief determinization help if MCTS is no longer forced on every decision and only fires at uncertain policy states?

Setup:

- Profile: `Pauper-Spy-Combo-Value-CardBeliefHeadOnly-20260512`
- Starting checkpoint: calibrated card-belief head-only model
- Skill: CP7
- Eval size: 2 games per matchup, 8 games total
- `--mcts-iterations 4`
- `--mcts-determinizations 4`
- no Spy-specific terminal mode, card facts, keyword gates, or heuristic rewards

Run:

`20260513_cardbelief528_nonforced_mcts_i4d4_skip80_eval2`

Settings:

- `--mcts-skip-top-prob 0.80`

Result:

`1/8 = 12.5%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 1 | 2 | 50.0% |
| Jund Wildfire | 0 | 2 | 0.0% |
| Mono Red Rally | 0 | 2 | 0.0% |
| Grixis Affinity | 0 | 2 | 0.0% |

Run:

`20260513_cardbelief528_nonforced_mcts_i4d4_skip50_eval2`

Settings:

- `--mcts-skip-top-prob 0.50`

Result:

`0/8 = 0.0%`

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy Combo | 0 | 2 | 0.0% |
| Jund Wildfire | 0 | 2 | 0.0% |
| Mono Red Rally | 0 | 2 | 0.0% |
| Grixis Affinity | 0 | 2 | 0.0% |

Read:

Rejected for promotion and HPC. The failure is not only forced-MCTS overuse; even uncertainty-gated card-belief MCTS is destructive with the current evaluator/rollout stack. Do not scale this line until the MCTS backup/rollout policy is improved and passes a tiny local gate against the accepted no-search checkpoint.
