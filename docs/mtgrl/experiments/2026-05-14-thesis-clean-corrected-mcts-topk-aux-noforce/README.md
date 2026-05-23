# Thesis-clean corrected MCTS top-k auxiliary no-force - 2026-05-14

## Question

The earlier sparse train-time MCTS auxiliary runs were executed before the
root-action mapping bug was fixed. Does the corrected, generic root top-k MCTS
path produce a useful no-force policy auxiliary when trained briefly from the
accepted checkpoint?

## Thesis Boundary

Clean:

- terminal rewards only;
- no heuristic step rewards;
- no `SPY_LANDLESS_COMBO_WIN`;
- no card-name or action-text regex labels;
- no Spy action-facts candidate features;
- no selective search keywords;
- MCTS behavior is generic policy-confidence sampling plus generic root top-k.

## Setup

Profile:

```text
Pauper-Spy-Combo-Value-MCTSRootMapAuxTopKNoForce-20260514
```

Registry:

```text
local-training/local_pbt/thesis_clean/20260514_corrected_mcts_topk_aux_noforce_registry.json
```

Starting checkpoint:

```text
accepted model_latest.pt
sha256=72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E
```

Training:

```text
TOTAL_EPISODES=128
NUM_GAME_RUNNERS=4
OPPONENT_SAMPLER=hybrid
HYBRID_SELFPLAY_P=0.50
SKILL_MIX=1:0.25,3:0.25,7:0.50
MCTS_TRAINING_ENABLE=1
MCTS_TRAINING_FORCE_ACTION=0
MULTI_PLY_MCTS=1
MCTS_ROOT_TOP_K=4
MCTS_ROOT_INCLUDE_PASS=1
MCTS_ITERATIONS=4
MCTS_PARALLEL_ROLLOUTS=4
MCTS_SKIP_TOP_PROB=0.40
MCTS_TRAINING_SAMPLE_PROB=0.10
MCTS_KL_LOSS_COEF=1.0
```

## Training Result

The run completed cleanly.

```text
model_latest.pt sha256=7F1E17CC1D3EA4C5DD2648AD195521181A3D60373710516197BA9CAB74DE697B
MCTS_GATE: total=10548 sampler_null=0 fewcand=0 wrongtype=414 confident=6042 sampled_out=9494 activations=456
MCTS_STATS_FINAL: calls=456 avg_wallMs=158 avg_iters=4 total_iters=1824 root_map=1824/1824
```

The corrected mapper stayed mechanically healthy. Cost was still high and leaf
evaluation dominated per-iteration time.

## Pressure Eval

Run:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_corrected_mcts_topk_aux_noforce_pressure_g8
skill=7
opponents=Mono Red Rally,Grixis Affinity
```

The registry uses the weighted Affinity-pressure deck pool, so the opponent
filter expanded to repeated Rally and Affinity entries:

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 4 | 24 | 16.67% |
| Grixis Affinity | 8 | 44 | 18.18% |
| Combined | 12 | 68 | 17.65% |

## Verdict

Rejected. The root-mapping fix makes train-time MCTS mechanically valid, but
the corrected no-force top-k auxiliary is still a destructive policy-improvement
teacher. Do not scale this on HPC.

This closes the post-fix sparse train-time MCTS branch as a current scaling
candidate. Future search work needs a better evaluator or target generator, not
more coefficient or top-k tuning on this value-guided MCTS target.
