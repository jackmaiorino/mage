# Thesis-Clean Reference-Anchored Hard-Opponent Continuation

Date: 2026-05-13

## Question

Can accepted improve through a small amount of plain terminal-only continuation if policy drift is constrained by a frozen-reference KL anchor and the opponent sampler is weighted toward hard matchups?

Thesis boundary: clean. No heuristic step rewards, Spy-specific candidate features, Spy terminal mode, action-text filters, selective MCTS keywords, card-belief auxiliary, or MCTS training.

## Setup

- Start profile: `Pauper-Spy-Combo-Value`
- Clone: `Pauper-Spy-Combo-Value-RefAnchorHard-20260513`
- Opponent deck list: `decklist.active_profile_pool_thesis_affinity_pressure_20260510.txt`
- Reference anchor: `REFERENCE_POLICY_KL_COEF=0.50`
- Frozen reference: accepted `Pauper-Spy-Combo-Value/models/model_latest.pt`
- Conservative PPO:
  - `PPO_EPSILON=0.05`
  - `VALUE_LOSS_COEF=2.0`
  - `ENTROPY_LOSS_MULT=0.02`
  - `ACTOR_LR=5e-5`
  - `CRITIC_LR=1e-4`
  - `OTHER_LR=2e-5`
- Opponent sampler:
  - `OPPONENT_SAMPLER=hybrid`
  - `HYBRID_SELFPLAY_P=0.25`
  - `SKILL_MIX=1:0.20,3:0.30,7:0.50`

Note: an initial background launch had quoting errors and started the default accepted profile briefly. It was stopped immediately. Accepted `model_latest.pt` remained unchanged at SHA prefix `72857AA2975A`.

## Run

- Train run: `20260513_ref_anchor_hard_train128b`
- Episodes: 128
- Checkpoint hash: `0254564B2577`
- Safety copy: `model_latest_ep128.pt`
- Trainer exit: rc `0`

## Reduced CP7 Gate

Run: `20260513_ref_anchor_hard128_cp7_g4`

| Opponent | Wins | Total |
| --- | ---: | ---: |
| Spy mirror | 2 | 4 |
| Jund Wildfire | 1 | 4 |
| Mono Red Rally | 2 | 4 |
| Grixis Affinity | 0 | 4 |
| Overall | 5 | 16 |

## Decision

Rejected. The reference anchor did not prevent hard-opponent terminal-only continuation from degrading the accepted policy. Do not spend HPC on this continuation profile.

The result reinforces the current pattern: small local terminal-only continuation from the accepted checkpoint is unstable unless a mechanism first clears reduced CP7 gates locally.
