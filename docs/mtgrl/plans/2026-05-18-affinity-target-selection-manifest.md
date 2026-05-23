# Affinity Target-Selection Manifest - 2026-05-18

## Scope

Analysis-only manifest built from existing artifacts. No Maven, replay gate, league_bench, training, HPC, terminal search, or fresh source collection was run.

The thesis filter is: choose targets that test whether the accepted policy has generic, pressure-state failures that a bounded sibling or short-prefix search can turn into terminal-winning corrections.

## Inputs

- Generated at UTC: `2026-05-19T12:41:08+00:00`
- 32-game Affinity corpus: `local-training\local_pbt\corpora\20260518_affinity_failure_context`
- Current-family v77 decision summary: `local-training\local_pbt\spy_line_replay\20260518_v77_current_family_provenance_gate\v77_decision_summary.csv`
- v83 replay diagnostic: `local-training\local_pbt\spy_line_replay\20260518_v83_current_family_d013_roost_randomutil_restore_cp7\v83_current_family_d013_roost_randomutil_restore_cp7_diagnostic.md`

## Corpus Read

- 32-game corpus loss decisions ranked: `0`
- Current-family replay-ready candidates ranked: `37`
- High-pressure candidates with opponent permanents >=10: `16`
- Replay-ready current-family candidates: `37`

The older 32-game corpus contains the strongest pressure and source-zone signals, but most of those logs are not immediate replay targets because they lack the replay metadata that v77/v83 now prove is needed. The current-family artifacts are smaller, but they are actionable.

## Recommended Next Target

- Candidate: `v77_s4_D089_D092_flashback_target_window`
- Source: `v77_current_family`
- Scenario/seed: `4` / `907587944`
- Decision window: `D89` to `D92`
- Selected sequence: `D89 Flashback sacrifice three creatures -> D90 Masked Vandal -> D91 Masked Vandal (you) -> D92 Drossforge Bridge (EvalBot-Skill7)`
- Pressure: opponent permanents `11`, own life `2`
- Assessment: Very high: build one forced-prefix bridge to D089, then enumerate the D90-D92 target/sacrifice short window with terminal-win-only admission.

Why this advances the thesis: it is a live accepted-policy Affinity loss under public-board pressure, in a replay-ready current family, with a short generic target/sacrifice window around Dread Return rather than a card-scripted label import. A bounded sibling/short-prefix probe can admit only terminal-winning corrected suffixes, which directly tests whether the failure mode is policy-relevant rather than just replay-local.

Next exact unit: build one forced-prefix bridge for current-family scenario 4 through D089, carrying forward the same v83 source-profile CP7 shape and any needed per-search RandomUtil provenance. Then run a bounded sibling/short-prefix probe over the D90-D92 target/sacrifice window, admitting a case only if the baseline loses and the corrected suffix reaches terminal win. Do not train from this manifest alone.

## Top 10 Ranked Targets

| rank | candidate_id | source_family | rank_score | replay_ready | decision_number | window_end_decision | selected | selected_prob | opp_permanents | own_life | tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | v77_s4_D089_D092_flashback_target_window | v77_current_family | 117.424 | True | 89 | 92 | D89 Flashback sacrifice three creatures -> D90 Masked Vandal -> D91 Masked Vandal (you) -> D92 Drossforge Bridge (EvalBot-Skill7) | 0.5063 | 11 | 2 | current_family_window;terminal_combo_sequence;target_sequence;pressure_opp_permanents_ge_10;replay_ready_current_family |
| 2 | v77_s4_D088 | v77_current_family | 110.023 | True | 88 |  | Saruli Caretaker | 0.2481 | 12 | 2 | pressure_opp_permanents_ge_10;accepted_low_probability;target_sequence;replay_ready_current_family |
| 3 | v77_s4_D089 | v77_current_family | 109.424 | True | 89 |  | Flashback sacrifice three creatures | 0.5063 | 11 | 2 | pressure_opp_permanents_ge_10;terminal_combo_sequence;replay_ready_current_family |
| 4 | v77_s4_D095 | v77_current_family | 107.905 | True | 95 |  | Saruli Caretaker | 0.2996 | 11 | 2 | pressure_opp_permanents_ge_10;accepted_low_probability;target_sequence;replay_ready_current_family |
| 5 | v77_s4_D092 | v77_current_family | 107.838 | True | 92 |  | Drossforge Bridge (EvalBot-Skill7) | 0.3052 | 11 | 2 | pressure_opp_permanents_ge_10;accepted_low_probability;target_sequence;replay_ready_current_family |
| 6 | v77_s1_D116 | v77_current_family | 99.372 | True | 116 |  | Cast Saruli Caretaker | 0.219 | 16 | 4 | pressure_opp_permanents_ge_10;accepted_low_probability;replay_ready_current_family |
| 7 | v77_s4_D091 | v77_current_family | 98.411 | True | 91 |  | Masked Vandal (you) | 0.4241 | 11 | 2 | pressure_opp_permanents_ge_10;target_sequence;replay_ready_current_family |
| 8 | v77_s4_D085 | v77_current_family | 97.358 | True | 85 |  | Blood Fountain (EvalBot-Skill7) | 0.6368 | 12 | 2 | pressure_opp_permanents_ge_10;target_sequence;replay_ready_current_family |
| 9 | v77_s4_D090 | v77_current_family | 97.355 | True | 90 |  | Masked Vandal | 0.5121 | 11 | 2 | pressure_opp_permanents_ge_10;target_sequence;replay_ready_current_family |
| 10 | v77_s1_D089 | v77_current_family | 95.966 | True | 89 |  | Cast Masked Vandal | 0.2112 | 13 | 12 | pressure_opp_permanents_ge_10;accepted_low_probability;replay_ready_current_family |

## Replay-Ready Current-Family Shortlist

| rank | candidate_id | rank_score | decision_number | window_end_decision | selected | selected_prob | opp_permanents | bounded_search_assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | v77_s4_D089_D092_flashback_target_window | 117.424 | 89 | 92 | D89 Flashback sacrifice three creatures -> D90 Masked Vandal -> D91 Masked Vandal (you) -> D92 Drossforge Bridge (EvalBot-Skill7) | 0.5063 | 11 | Very high: build one forced-prefix bridge to D089, then enumerate the D90-D92 target/sacrifice short window with terminal-win-o... |
| 2 | v77_s4_D088 | 110.023 | 88 |  | Saruli Caretaker | 0.2481 | 12 | High: current-family metadata and v83 D013 replay parity make a bounded sibling/short-prefix probe actionable. |
| 3 | v77_s4_D089 | 109.424 | 89 |  | Flashback sacrifice three creatures | 0.5063 | 11 | High: current-family metadata and v83 D013 replay parity make a bounded sibling/short-prefix probe actionable. |
| 4 | v77_s4_D095 | 107.905 | 95 |  | Saruli Caretaker | 0.2996 | 11 | High: current-family metadata and v83 D013 replay parity make a bounded sibling/short-prefix probe actionable. |
| 5 | v77_s4_D092 | 107.838 | 92 |  | Drossforge Bridge (EvalBot-Skill7) | 0.3052 | 11 | High: current-family metadata and v83 D013 replay parity make a bounded sibling/short-prefix probe actionable. |

## High-Thesis Older Corpus Shortlist

| rank | candidate_id | rank_score | decision_number | selected | selected_prob | opp_permanents | replay_ready | replay_blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

These older rows should guide future collection/recollection, not the immediate replay gate, unless a replay-ready source family is rebuilt for them.

## Explicit Non-Targets

- `v83 D013`: Already passed as a setup/parity gate; repeating it does not test pressure recovery.
- `v77 D025`: Documented as mana/pass-adjacent bridge work, not a policy-relevant pressure decision.
- `singleton pass rows under pressure`: They dominate the corpus but often have no meaningful sibling branch to test.
- `old compact D084/D103/D134 anchors as immediate replay gates`: High thesis value but not immediate gates because the old logs lack replay seed/snapshot metadata.
- `automatic D025 or D089 ordinal walking`: Replay work should be target-selected by pressure and terminal-recovery value, not by next ordinal.

## Machine Artifacts

- CSV: `local-training\local_pbt\corpora\20260518_affinity_target_selection\target_selection_manifest.csv`
- JSON: `local-training\local_pbt\corpora\20260518_affinity_target_selection\target_selection_manifest.json`
