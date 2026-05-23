# Spy/Wildfire Hybrid CP1 Self-Play Curriculum

Date: 2026-04-28

## Purpose

The previous valid evals showed that the active Spy Combo and Wildfire value profiles are still 0% against CP7 skill 1 and skill 7. Continuing directly against skill 7 is therefore likely to provide sparse terminal signal and poor wall-clock value.

This experiment tests whether the same terminal-reward system learns faster when the two target decks train against a lower-cost, lower-strength mix:

- 75% profile-routed meta self-play
- 25% CP7 skill 1
- opponent-side RL training enabled for profile-routed games
- no online MCTS
- terminal reward only
- active 128-dim, 2-layer value checkpoints

## Profiles

- `Pauper-Spy-Combo-Value`
- `Pauper-Wildfire-Value`

## Launch Settings

- `TRAIN_PROFILES=2`
- `TOTAL_EPISODES=25000`
- `NUM_GAME_RUNNERS=96`
- `OPPONENT_SAMPLER=meta_hybrid`
- `META_HYBRID_META_P=0.75`
- `SKILL_MIX=1:1.0`
- `SELFPLAY_OPPONENT_TRAINING=1`
- `LEARNER_BATCH_MAX_EPISODES=4`
- `LEARNER_BATCH_MAX_STEPS=1600`
- `TRAIN_MULTI_MAX_STEPS=1600`
- `TRAIN_CHUNK_SIZE=192`
- `PENDING_TRAIN_MAX=96`

## Pre-Run State

- Spy active episodes: 17,464
- Wildfire active episodes: 17,466
- Valid CP1 probe after architecture fix: Spy 0/8, Wildfire 0/8
- Valid CP7 anchor sweep after architecture fix: Spy 0/8, Wildfire 0/8

## Execution Notes

- The first launch was stopped early around episode 19.1k because periodic ONNX export attempted to overwrite active files held by ONNX Runtime on Windows.
- The export/reload path was changed to versioned ONNX directories plus a `.active_dir` pointer before resuming the experiment.

## 25k Results

Run: `20260428_spy_wildfire_meta_hybrid_25k_cp1_eval_g10_p8`

CP1 held-out eval, 10 games per matchup across the four-deck pool:

- `Pauper-Spy-Combo-Value`: 10/40 = 25.0%
- `Pauper-Wildfire-Value`: 11/40 = 27.5%

Per-matchup:

- Spy vs Spy Combo: 7/10
- Spy vs Jund Wildfire: 1/10
- Spy vs Mono Red Rally: 0/10
- Spy vs Grixis Affinity: 2/10
- Wildfire vs Spy Combo: 3/10
- Wildfire vs Jund Wildfire: 4/10
- Wildfire vs Mono Red Rally: 1/10
- Wildfire vs Grixis Affinity: 3/10

Interpretation: this is not strong play yet, but it is a real transfer improvement over the pre-run CP1 eval of 0/8 for both profiles. The next phase should increase CP1 exposure while keeping enough profile-routed meta play to train both sides under their correct deck profiles.

## Follow-Up Phase

- Continue Spy/Wildfire to 35,000 episodes per profile.
- Change `META_HYBRID_META_P` from `0.75` to `0.50`.
- Keep `SKILL_MIX=1:1.0`.
- Re-run the same CP1 eval sweep at 35k.

## Later Results

### 35k p50

Run: `20260429_spy_wildfire_meta_hybrid_35k_cp1_eval_g10_p16`

- Spy: 15/40 = 37.5%
- Wildfire: 3/40 = 7.5%

The p50 mix helped Spy but damaged Wildfire. Logged Wildfire probe still showed a major basic-action issue:

- pass-over-land: 22/42 = 52.4%
- early pass-over-land: 10/42 = 23.8%
- turn-1 pass-over-land: 3/4 = 75.0%

### 45k p75 Recovery

Run: `20260429_spy_wildfire_meta_hybrid_45k_p75_cp1_eval_g10_p16`

- Spy: 10/40 = 25.0%
- Wildfire: 4/40 = 10.0%

Wildfire action-health improved but eval did not:

- pass-over-land: 26/69 = 37.7%
- early pass-over-land: 6/69 = 8.7%
- turn-1 pass-over-land: 1/4 = 25.0%

### 55k p75 Low Entropy

Run: `20260429_spy_wildfire_meta_hybrid_55k_lowentropy_cp1_eval_g10_p16`

- Spy: 7/40 = 17.5%
- Wildfire: 13/40 = 32.5%

Logged Wildfire probe:

- pass-over-land: 0/25 = 0.0%
- turn-1 pass-over-land: 0/3 = 0.0%
- mulligan keeps: 4/4, including 1 keep with 0-1 lands
- Cleansing Wildfire target decisions: 2
- Cleansing target self-bridge: 0/2
- Cleansing target self-other: 1/2
- Cleansing target opponent: 1/2

Interpretation: low entropy repaired the obvious land-drop pathology in the probe and recovered Wildfire CP1 performance, but it exposed a deeper target-selection failure for deck-engine cards like `Cleansing Wildfire`.

### Wildfire Prefix Counterfactual Attempt

Run: `20260429_wildfire_prefix_cp1_s48_w24_ai24_d6_n64`

- Purpose: terminal-only winning-prefix search for Wildfire, CP1 opponent pool.
- Important throughput fix: `scripts/run_action_counterfactual.ps1` now sets `AI_MAX_THREADS_FOR_SIMULATIONS`; without this, search sat around 30-36% CPU. With `-AiThreads 24`, search reached roughly 80-95% CPU in the main phase.
- Search result: 3/48 scenarios produced winning prefixes, 16 selected labels, 512 train samples, tensor replay 16/16.
- Label mix was not useful for the observed target problem: all labels were `ACTIVATE_ABILITY_OR_SPELL`, mostly early mana actions; no `SELECT_TARGETS` labels for `Cleansing Wildfire`.
- Post-train CP1 eval: `20260429_wildfire_after_prefix_acf_cp1_eval_g10_p16`, Wildfire 0/40.
- Decision: restore the pre-counterfactual 55k low-entropy checkpoint and continue terminal selfplay/CP1 training rather than compounding the regression.

### 65k p75 Low Entropy Continuation

Run: `20260429_spy_wildfire_meta_hybrid_65k_p75_lowentropy_cp1_eval_g10_p16`

- Spy: 3/28 completed = 10.7%
- Wildfire: 7/40 = 17.5%
- Throughput note: the first launch used the eval script's default `--serial-warmup-jobs=1`, which left only one Java worker active. Restarting with `--serial-warmup-jobs 0` immediately brought CPU to roughly 80-100% with 16 eval workers.

Logged Wildfire probe:

- pass-over-land: 0/63 = 0.0%
- turn-1 pass-over-land: 0/0
- mulligan keeps: 4/4, including 1 keep with 0-1 lands
- Cleansing Wildfire target decisions: 6
- Cleansing target self-bridge: 0/6
- Cleansing target self-other: 3/6
- Cleansing target opponent: 3/6

Logged Spy probe:

- mulligan keeps: 4/4, all with 0-1 lands
- `Balustrade Spy` cast: 2 times across 4 logged games
- `Dread Return` cast: 0 times
- Common selected plays were normal board-development actions (`Gatecreeper Vine`, `Overgrown Battlement`, `Masked Vandal`, `Lead the Stampede`, `Winding Way`).

Interpretation: continuing the same p75 low-entropy curriculum past 55k regressed both profiles. Wildfire retained basic land-drop behavior but did not learn the deck-engine target choice. Spy showed the expected midrange-minima failure: it plays the deck as a slow creature deck instead of reliably converting Spy into `Dread Return`/`Lotleth Giant`.

### Checkpoint Selection

Because 65k regressed, active latest is no longer treated as best. Before the next run:

- Backed up the 65k active models to `local-training/local_pbt/model_backups/20260429_pre_checkpoint_selection_65k`.
- Quick-swept surviving Spy snapshots with 2 CP1 games per matchup.
- Best surviving Spy candidate: `snapshot_step_19000.pt`, 3/8 = 37.5%.
- Restored Spy active model from `snapshot_step_19000.pt`.
- Restored Wildfire active model from the exact 55k pre-counterfactual backup.

Next run direction: split training by profile rather than forcing one shared curriculum. Spy should continue from the 35k-adjacent checkpoint under a CP1-heavier/meta-selfplay mix to fight the midrange minimum. Wildfire should not simply continue generic selfplay until we have a terminal-only way to make `Cleansing Wildfire` target selection appear in the training signal.

### Spy 75k CP1-Heavy Continuation

Run: `20260429_spy_checkpoint19000_cp1heavy_p25_r144_to75k`

- Start point: restored Spy from `snapshot_step_19000.pt`, the best surviving quick-check checkpoint after 65k regression.
- Training mix: single-profile Spy, `META_HYBRID_META_P=0.25`, `SKILL_MIX=1:1.0`, `NUM_GAME_RUNNERS=144`.
- Throughput: CPU generally saturated near 100%, with expected dips during ONNX export, learner/backpressure phases, and final straggler episodes. The dips were transient and recovered without intervention.
- Reached `75,000` episodes.

CP1 eval:

Run: `20260429_spy_checkpoint19000_cp1heavy_75k_cp1_eval_g10_p16`

- Spy total: 16/40 = 40.0%
- vs Grixis Affinity: 2/10
- vs Jund Wildfire: 3/10
- vs Mono Red Rally: 3/10
- vs Spy Combo: 8/10

Logged Spy probe:

- pass-over-land: 0/43 = 0.0%
- mulligan keeps: 6/6, all with 0-1 lands
- `Balustrade Spy` cast: 2 across 6 logged games
- `Dread Return` cast: 0
- `Lotleth Giant` cast: 0

CP7 anchor:

Run: `20260429_spy_checkpoint19000_cp1heavy_75k_cp7_anchor_g5_p16`

- Spy total: 3/15 completed = 20.0%
- vs Grixis Affinity: 1/1 completed
- vs Jund Wildfire: 2/4 completed
- vs Mono Red Rally: 0/5
- vs Spy Combo: 0/5

Interpretation: CP1-heavy terminal training recovered a playable Spy policy and gave a nonzero CP7 anchor, so the 65k collapse was not permanent. However, the logged probe still shows weak combo conversion; the policy is still too often playing slow creature/value games rather than finishing through `Dread Return`. Next local step is a graduation curriculum with mostly CP1/CP3 and a small CP7 fraction. The 75k checkpoint was backed up to `local-training/local_pbt/model_backups/20260429_spy_75k_cp1heavy_checkpoint`.

## Decision Rule

After the run reaches 25,000 episodes per target profile or earlier if it is clearly unhealthy:

1. Check throughput and GPU/CPU saturation.
2. Check rolling self-play/CP1 training winrates by opponent type.
3. Run a valid CP1 eval sweep for Spy/Wildfire.
4. If CP1 remains 0% with no value-head improvement, pivot away from generic game-only curriculum toward a narrower terminal-only deck-execution curriculum.
5. If CP1 improves, continue the curriculum with either a CP1/CP2 skill mix or a lower self-play ratio.
