# Affinity Refresh Target Selection - 2026-05-19

## Scope

Artifact-only v115 analysis of the completed accepted-policy Grixis Affinity
replay-metadata refresh. No Maven, replay gate, terminal search, training, HPC,
or commit was run in this step.

Thesis filter: use replay/counterfactual work only when it can produce honest
terminal-winning correction evidence for accepted-policy failures. Do not repeat
the clean-negative current-family surfaces unless the mechanism materially
changes.

## Inputs

- v114 run root: `local-training/local_pbt/cp7_eval_sweeps/20260519_v114_affinity_replay_metadata_g8`
- v114 result: `3/8 = 0.375`; losses on chunks `1`, `4`, `5`, `6`, `8`
- v114 replay metadata: all 8 compact logs include `REPLAY:` metadata with `action_counterfactual_compatible=true`
- v115 failure corpus: `local-training/local_pbt/corpora/20260519_v115_affinity_refresh_failure_context`
- v115 raw target manifest: `local-training/local_pbt/corpora/20260519_v115_affinity_refresh_target_selection`
- Research priors: accepted CP7 Grixis Affinity gate remains `5/16` then `10/32`

## Corpus Read

- Loss games included: `5`
- Loss decision rows: `577`
- Raw manifest candidates: `145`
- Raw high-pressure candidates with `opp_permanents >= 10`: `44`
- Raw replay-ready current-family candidates: `37`

The raw manifest still ranks the old v77 current-family rows first because the
manifest builder does not know about the v103-v113 clean-negative exclusions.
Those raw rankings are retained in CSV/JSON for inspection, but they are not the
v115 recommendation.

## Excluded Current-Family Surfaces

Reject unless the mechanism materially changes:

- scenario 4 `D089-D092`, `D088`, `D095`, `D085`, `D084`, `D068`
- scenario 1 `D089`, `D090`, `D095`, `D096`, `D116`
- scenario 4 `D093`, because it is a singleton `Yes` row inside the rejected
  D089-D092/D095 family

After this filter, the highest strict-pressure fresh v114 row is not a v77 row.

## Recommended Fresh Target

- Candidate: `game_20260519_063714_0001_D061`
- Source log: `local-training/local_pbt/cp7_eval_sweeps/20260519_v114_affinity_replay_metadata_g8/game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity__chunk_001/game_20260519_063714_0001.txt`
- Replay metadata: scenario `1`, seed `763880686`, `action_counterfactual_compatible=true`
- Decision: displayed `D061`; ACF bridge target ordinal `20`
- Surface: `TARGET_PICK`, selected `Dread Return`
- Top alternatives: two `Lead the Stampede` copies
- Pressure: opponent permanents `11`, own life `16`
- State signal: own graveyard `20`, own hand `3`, target row tagged as source-zone-sensitive

Why this is the next low-risk action: it is a fresh accepted-policy loss from
the replay-metadata refresh, it is strict-pressure, it has bounded alternatives,
and it can only become training evidence if a forced sibling reaches terminal
win. This avoids another human-looking or ordinal-walking replay probe.

Caveat: the raw manifest marks this row `replay_ready=false` because the builder
only treats the old current-family summary as replay-ready. The compact log
itself has the new v114 replay metadata, and the script-only bridge conversion
completed cleanly.

## Script-Only Prep Completed

Built:

`local-training/local_pbt/corpora/20260519_v115_affinity_refresh_d061_bridge/forced_prefix_replay_game_20260519_063714_0001_through_D061_target_D061_v115_bridge.csv`

Bridge summary:

- rows: `21`
- skipped singleton/no-op compact rows: `40`
- ordinal space: `acf`
- target decisions: `[61]`
- first priority hand rows: `1`
- source library rows: `1`
- source candidate metadata rows: `21`

## Next Exact Unit

Run one bounded source-profile CP7 forced-prefix reachability gate using the
v115 D061 bridge CSV. If it reaches `prefix_failure_ordinal=-1`, run at most two
terminal-only sibling checks that force the two `Lead the Stampede` target
alternatives at D061. Admit the target only if a forced sibling reaches terminal
win. If reachability fails, record the first mismatch and do not broaden into a
search run.

## Machine Artifacts

- Corpus JSONL: `local-training/local_pbt/corpora/20260519_v115_affinity_refresh_failure_context/loss_decisions.jsonl`
- Corpus CSV: `local-training/local_pbt/corpora/20260519_v115_affinity_refresh_failure_context/loss_decisions.csv`
- Corpus summary: `local-training/local_pbt/corpora/20260519_v115_affinity_refresh_failure_context/summary.json`
- Raw manifest CSV: `local-training/local_pbt/corpora/20260519_v115_affinity_refresh_target_selection/target_selection_manifest.csv`
- Raw manifest JSON: `local-training/local_pbt/corpora/20260519_v115_affinity_refresh_target_selection/target_selection_manifest.json`
