# Thesis-Clean Compact Affinity Log Review

Date: 2026-05-13

## Question

Can the compact game logs identify the next high-value thesis-clean failure mode in the accepted checkpoint's weakest CP7 matchup?

This is diagnostic-only. No training used card-name labels, Spy terminal modes, heuristic rewards, hand pools, or selective search gates.

## Run

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_compact_affinity_review_g4
profile: Pauper-Spy-Combo-Value
opponent: Deck - Grixis Affinity
skill: CP7
games: 4
format: compact
```

Result:

```text
1/4 = 25.00%
chunk_001 loss
chunk_002 loss
chunk_003 win
chunk_004 loss
```

## Tooling Fix

`scripts/export_game_log_trajectories.py` now parses compact `TOP:` summaries into option rows. This lets `scripts/analyze_logged_value_head.py` identify nontrivial decisions and visible alternatives from compact logs instead of requiring full verbose logs.

Validation:

```text
python -m py_compile scripts/export_game_log_trajectories.py scripts/analyze_logged_value_head.py scripts/view_game_log_trace.py
```

## Value-Head Diagnostic

Export:

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_compact_affinity_review_g4/diagnostics/compact_affinity_games.jsonl
games=4
decisions=519
```

Value analysis:

```text
local-training/local_pbt/cp7_eval_sweeps/20260513_compact_affinity_review_g4/diagnostics/value_head_analysis
decision_rows=475
all AUC=0.549842
nontrivial_options AUC=0.575085
turn_le_3 AUC=0.491033
turn_ge_4 AUC=0.586731
has_cast_spy_option AUC=0.000000 on n=3
```

This is too small for statistical claims, but it reinforces the branch-value probe: the value head is noisy at the specific decisions where Affinity games turn. In chunk 002 it assigned `0.738721` while the agent was at 11 life facing a lethal Affinity board and later died at 1 life. In chunk 001 it reached `0.717122` late in a game that still ended in a loss.

## Manual Findings

### chunk_001 loss

The agent mulliganed to four: `Forest; Dread Return; Gatecreeper Vine; Overgrown Battlement`. It eventually cast `Balustrade Spy` on turn 10, but the mill did not produce the lethal state. It later flashbacked `Dread Return` and targeted `Balustrade Spy` rather than `Lotleth Giant`; the graveyard did not contain the payoff at that point. The game drifted to turn 17 with the value head repeatedly positive while Affinity stabilized.

Interpretation: this is a post-mill/post-Spy execution and payoff-availability failure, not a simple "never cast Spy" failure.

### chunk_002 loss

The agent kept six: `Swamp; Land Grant; Winding Way; Lotus Petal; Masked Vandal; Elves of Deep Shadow`. It developed normally but never assembled Spy before Affinity's board became lethal. Around turn 6 the value head oscillated sharply and hit `0.738721` despite the agent being under lethal pressure from a very wide board.

Interpretation: the model underestimates public-board pressure and overvalues slow development lines after a non-combo opener.

### chunk_003 win

The winning game also mulliganed to four, but kept `Land Grant; Lotus Petal; Saruli Caretaker; Saruli Caretaker`. It produced a working mana base, cast Spy on turn 8, targeted self, flashed back `Dread Return`, selected `Lotleth Giant`, and won.

Interpretation: the policy can still execute the full line when the post-mill state contains the correct payoff and the game has enough time.

### chunk_004 loss

The agent mulliganed to two and kept `Overgrown Battlement; Saruli Caretaker`. It had no real development path and died before a meaningful combo state.

Interpretation: root mulligan is part of the pressure matchup problem, but prior root-mulligan prefix training already failed CP7. The issue is likely root selection plus downstream conversion under pressure, not root labels alone.

## Decision

Do not launch HPC from this diagnostic. The useful result is direction:

- avoid repeating root-mulligan-only training;
- avoid scaling MCTS until value/ranking improves;
- target a generic pressure-aware trajectory objective that can learn from public board state, mulligan depth, and downstream conversion without naming Spy cards.

Next experiment should be local and thesis-clean: collect compact/log-derived pressure failures or forced-branch trajectories, then train a decision-local objective that contrasts downstream terminal outcomes in context rather than blunt value-only importing every forced trajectory state.
