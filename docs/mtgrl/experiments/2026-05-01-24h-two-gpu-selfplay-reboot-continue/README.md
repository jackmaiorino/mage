# 24h Two-GPU Pure Self-Play Continuation

Date: 2026-05-01 to 2026-05-02

## Objective

Continue the two-GPU pure self-play curriculum after a PC restart and measure whether more terminal-only self-play improves Pauper Spy Combo and Jund Wildfire against CP evaluators.

This run intentionally did not use MCTS or step rewards. The goal was to keep the learning signal terminal-only while maximizing local throughput.

## Run

- Autonomous run: `20260501_24h_twogpu_selfplay_reboot_continue`
- Started: 2026-05-01 7:40 AM Eastern
- Completed: 2026-05-02 7:40 AM Eastern
- Training phases: seven 3-hour phases plus one final 5869-second partial phase
- Profiles trained:
  - `Pauper-Spy-Combo-Value`
  - `Pauper-Wildfire-Value`
  - `Pauper-Rally-Anchor-Value`
  - `Pauper-Affinity-Anchor-Value`
- Eval gates: CP1 and CP3, Spy/Wildfire only, 4 games per matchup

## Configuration

- `TRAIN_PROFILES=4`
- `NUM_GAME_RUNNERS=64`
- `OPPONENT_SAMPLER=self`
- `SELFPLAY_OPPONENT_TRAINING=1`
- `RL_HEURISTIC_STEP_REWARDS=0`
- `MCTS_TRAINING_ENABLE=0`
- `ISMCTS_ENABLE=0`
- `TRAIN_CUDA_DEVICE=cuda:0` RTX 4070 Super
- `INFER_CUDA_DEVICE=cuda:1` RTX 3050
- `USE_TRT_INFERENCE=1`
- `ONNX_GPU_MEM_LIMIT_MB=4096`
- `CUDA_MEM_FRACTION=0.70`
- `TRAIN_GPU_MAX_CONCURRENT=2`
- `LEARNER_BATCH_MAX_EPISODES=8`
- `LEARNER_BATCH_MAX_STEPS=4096`
- `TRAIN_CHUNK_SIZE=256`

## Episode Counts

| Profile | Start | End | Delta |
| --- | ---: | ---: | ---: |
| Pauper-Spy-Combo-Value | 128431 | 200919 | 72488 |
| Pauper-Wildfire-Value | 108519 | 183921 | 75402 |
| Pauper-Rally-Anchor-Value | 107104 | 272056 | 164952 |
| Pauper-Affinity-Anchor-Value | 84049 | 184344 | 100295 |

Total delta: 413137 episodes in about 24 hours, roughly 4.78 aggregate eps/s including eval/export overhead. During steady training, sampled throughput was usually about 4.5-5.4 aggregate eps/s.

## Phase Gates

Each phase gate is 16 total games per evaluated profile, so the point estimates are noisy.

| Phase | CP1 Spy | CP1 Wildfire | CP3 Spy | CP3 Wildfire |
| --- | ---: | ---: | ---: | ---: |
| 1 | 5/16 | 1/16 | 4/16 | 5/16 |
| 2 | 5/16 | 5/16 | 5/16 | 4/16 |
| 3 | 6/16 | 5/16 | 7/16 | 3/16 |
| 4 | 4/16 | 4/16 | 3/16 | 4/16 |
| 5 | 6/16 | 6/16 | 4/16 | 2/16 |
| 6 | 7/16 | 4/16 | 7/16 | 2/16 |
| 7 | 5/16 | 6/16 | 6/16 | 4/16 |

Phase 8 was the final partial phase and did not run an automatic gate because the 24-hour deadline expired immediately after training. A manual post-run eval was run instead.

## Final Post-Run Eval

Larger CP1/CP3 checks used 8 games per matchup. CP7 used 4 games per matchup.

| Eval | Spy | Wildfire |
| --- | ---: | ---: |
| CP1 | 14/32 = 43.75% | 8/32 = 25.00% |
| CP3 | 12/32 = 37.50% | 10/32 = 31.25% |
| CP7 | 5/16 = 31.25% | 3/16 = 18.75% |

Final per-matchup highlights:

- Spy is strong in mirrors but still poor into Rally and Affinity.
- Spy CP7: mirror 4/4, Wildfire 1/4, Rally 0/4, Affinity 0/4.
- Wildfire is mostly winning only against Spy.
- Wildfire CP7: Spy 2/4, mirror 1/4, Rally 0/4, Affinity 0/4.

## Interpretation

The run is not enough to claim convergence, but it is better than the earlier 0% evaluations. Pure self-play at the two-GPU split can generate useful signal quickly, especially for Spy, without adding intermittent rewards.

Spy is no longer purely stuck at zero, but it still looks matchup-fragile. The current policy appears to learn some Spy-vs-Spy and Spy-vs-Wildfire play while failing to handle Rally and Affinity pressure.

Wildfire is not improving enough under this curriculum. Its best final read is 31.25% vs CP3, but CP7 is 18.75%, and the per-matchup table suggests it is leaning on beating Spy rather than learning robust Wildfire game plans.

## Operational Notes

The final post-run eval initially failed because `local-training/local_pbt/cp7_eval_sweeps` is a junction to `D:\mtgrl-local-training\local_pbt\cp7_eval_sweeps`, and D: was nearly full. Generated eval `db` and `snapshot` folders from older sweeps were removed, freeing about 217 GB while preserving summary files.

## Next Experiment

Continue terminal-only learning, but stop treating all four profile episodes as equally valuable. The next experiment should be an adaptive self-play curriculum:

1. Keep two-GPU pure self-play as the base because it is the best local throughput path.
2. Oversample target profiles, especially Wildfire, so Rally/Affinity do not consume most of the generated games.
3. Add opponent/deck sampling weights from eval weakness:
   - Spy should see more Rally and Affinity opponents.
   - Wildfire should see more Rally, Affinity, and mirrors.
4. Keep terminal-only reward and no online MCTS for the main run.
5. Add larger post-run eval gates, not tiny 16-game phase gates, before drawing conclusions.

Recommended concrete run:

- `TRAIN_PROFILES=4`
- target weighted sampling: Spy 35%, Wildfire 35%, Rally 15%, Affinity 15%
- opponent weighted sampling from final losses
- 64 runners, same 4070/3050 split
- 12 hours first, with 8 games per matchup CP1/CP3 and 4 games per matchup CP7 at the end
