# Thesis-Clean Sparse MCTS Branch

Date: 2026-05-11

## Question

Can a thesis-clean, sparse train-time MCTS signal improve the accepted Affinity-pressure checkpoint without spending HPC budget?

This tests generic search-as-policy-improvement only:

- no Spy action-facts features
- no Spy terminal mode
- no action-text/card-name gates
- no selective MCTS keywords
- terminal rewards only
- generic confidence gate plus random sparse train-time MCTS sampling

## Training Setup

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_sparse_mcts_skip40_sample10_200ep/phase_001_train`

Backup: `local-training/local_pbt/model_backups/pre_20260511_thesis_clean_sparse_mcts_skip40_sample10_200ep`

Settings:

- start checkpoint: accepted Affinity-pressure checkpoint
- one active Spy profile
- `NUM_GAME_RUNNERS=8`
- `MCTS_TRAINING_ENABLE=1`
- `MULTI_PLY_MCTS=1`
- `MCTS_ITERATIONS=2`
- `MCTS_SKIP_TOP_PROB=0.40`
- `MCTS_TRAINING_SAMPLE_PROB=0.10`
- `MCTS_SELECTIVE_ENABLE=0`
- `SKILL_MIX=1:1.0`
- `HYBRID_SELFPLAY_P=0.70`

Result:

- parsed training rows: `199`
- training wins: `61/199`
- self-play: `39/139`
- CP1: `22/60`
- average parsed game seconds: `12.66`
- MCTS: `775` activations, `avg_wallMs=157`, `avg_iters=2`
- gate: `total=18358 sampler_null=0 fewcand=0 wrongtype=996 confident=10819 sampled_out=16511 not_tactical=0 activations=775`

## Reduced Eval

Eval registry: `local-training/local_pbt/thesis_clean/20260510_thesis_clean_affinity_pressure_spy_eval_registry.json`

Games: 16 per matchup, unique deck pool.

CP1: `35/64 = 54.69%`

- Spy mirror: `16/16`
- Wildfire: `10/16`
- Rally: `4/16`
- Affinity: `5/16`

CP3: `38/64 = 59.38%`

- Spy mirror: `16/16`
- Wildfire: `9/16`
- Rally: `8/16`
- Affinity: `5/16`

CP7: `21/63 = 33.33%`

- Spy mirror: `11/16`
- Wildfire: `6/15`
- Rally: `4/16`
- Affinity: `0/16`

## Decision

Rejected.

The branch looked promising on reduced CP1/CP3, but CP7 failed the gate and Affinity collapsed to `0/16`. This is below the accepted checkpoint's corrected unique CP7 result (`108/242 = 44.63%`) and not worth HPC.

The accepted checkpoint was restored from `pre_20260511_thesis_clean_sparse_mcts_skip40_sample10_200ep`.

## Next

The failure is likely opponent-distribution related rather than MCTS plumbing:

- training used `SKILL_MIX=1:1.0`, so the branch learned from self-play plus CP1 pressure only;
- reduced CP1/CP3 improved, while CP7 and Affinity regressed.

Next local experiment should keep sparse MCTS but train against a generic hard-skill mix, for example `SKILL_MIX=1:0.34,3:0.33,7:0.33` or `SKILL_MIX=7:1.0`, before any HPC submission.

## Hard-Skill Mix Follow-Up

Run: `local-training/local_pbt/autonomous_runs/20260511_thesis_clean_sparse_mcts_hardskill_mix_200ep/phase_001_train`

Backup: `local-training/local_pbt/model_backups/pre_20260511_thesis_clean_sparse_mcts_hardskill_mix_200ep`

Settings:

- start checkpoint: accepted Affinity-pressure checkpoint
- one active Spy profile
- `NUM_GAME_RUNNERS=8`
- `MCTS_TRAINING_ENABLE=1`
- `MULTI_PLY_MCTS=1`
- `MCTS_ITERATIONS=2`
- `MCTS_SKIP_TOP_PROB=0.40`
- `MCTS_TRAINING_SAMPLE_PROB=0.10`
- `MCTS_SELECTIVE_ENABLE=0`
- `SKILL_MIX=1:0.25,3:0.25,7:0.50`
- `HYBRID_SELFPLAY_P=0.50`

Training result:

- parsed training rows: `199`
- training wins: `59/199`
- self-play: `25/90`
- CP1: `14/27`
- CP3: `5/22`
- CP7: `15/60`
- average parsed game seconds: `27.89`
- MCTS: `701` activations, `avg_wallMs=211`, `avg_iters=2`
- gate: `total=16504 sampler_null=0 fewcand=0 wrongtype=771 confident=9592 sampled_out=14829 not_tactical=0 activations=701`

Reduced CP7 eval: `23/64 = 35.94%`

- Spy mirror: `12/16`
- Wildfire: `5/16`
- Rally: `3/16`
- Affinity: `3/16`

Decision: rejected.

The hard-skill mix did not fix the CP7 regression. It recovered Affinity slightly from the first sparse-MCTS branch, but still landed far below the accepted checkpoint's corrected unique CP7 result (`108/242 = 44.63%`). The accepted checkpoint was restored from `pre_20260511_thesis_clean_sparse_mcts_hardskill_mix_200ep`.

Next control: run the same generic hard-skill opponent mix with train-time MCTS disabled. That isolates whether the CP7 regression is caused by sparse MCTS targets or by the hard-skill curriculum itself.
