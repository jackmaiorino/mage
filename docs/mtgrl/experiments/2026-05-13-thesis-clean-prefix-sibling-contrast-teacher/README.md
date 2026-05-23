# Thesis-Clean Prefix Sibling-Contrast Teacher

Date: 2026-05-13

## Question

Can the generic prefix-search teacher produce more useful hard-matchup labels if it adds sibling outcome contrast at searched prefix nodes, instead of only positive winning-prefix targets?

This stays inside the morphed thesis boundary:

- terminal `WIN` only;
- generic branch order;
- no tactic autopilot;
- no action-text/card-name filters;
- no Spy terminal mode;
- no heuristic rewards;
- no selective MCTS keyword gates.

## Collection

Run:

`20260513_prefix_sibling_contrast_affinity_cp7_collect32`

Export:

`local-training/local_pbt/generic_prefix_data/20260513_prefix_sibling_contrast_affinity_cp7_collect32.ser`

Settings:

- accepted `Pauper-Spy-Combo-Value` checkpoint;
- Spy vs Grixis Affinity, CP7;
- 32 scenarios;
- `max_prefix_depth=8`;
- `train_prefix_depth=8`;
- `max_search_nodes=31`;
- `top_k=3`;
- `random_extra=1`;
- `prefix_sibling_contrast=true`;
- `prefix_sibling_contrast_search_nodes=20`;
- `skip_pass_training=true`;
- `skip_blank_training=true`;
- `skip_mulligan_training=true`;
- collect-only.

Result:

```text
trainedScenarios=15
skippedScenarios=17
candidateExamples=48
winningTrajectories=7
selectedExamples=47
```

## Score Probe

Run:

`20260513_prefix_sibling_contrast_affinity_cp7_score_accepted`

Accepted-checkpoint score:

```text
top1=47/47 = 100.00%
targetSetTop1=47/47 = 100.00%
avgTargetProb=0.484455
avgRank=1.0000
```

## Verdict

Rejected before training.

Sibling contrast produced a small thesis-clean Affinity CP7 corpus, but the accepted checkpoint already ranks every selected target top-1. Training from this data would repeat the earlier failure mode: fitting states the policy already scores correctly while not changing fresh-start hard-matchup play.

Do not import/train this dataset. The next teacher needs to sample genuinely unsolved live states or use online policy improvement, not more searched-prefix labels from states accepted already resolves locally.
