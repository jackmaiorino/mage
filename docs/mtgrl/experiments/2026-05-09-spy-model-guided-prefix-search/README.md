# 2026-05-09 Spy Model-Guided Prefix Search

Goal: test whether terminal prefix search gets a better transfer surface when branch fallback uses the current model instead of the previous `NoSearchModelScoring` setting.

## Offline Teacher Probe

Run:

- `20260509_spy_modelguided_top3r0_cp1_n64_nodes7`
- Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- CP1, 64 natural starts
- `top_k=3`, `random_extra=0`
- 7 nodes, depth 6
- `TacticAutopilot`
- Model scoring enabled during search
- Collect-only, no heuristic rewards

Result:

- Searched wins: `51/64`
- Exported prefix tensors: `303`
- Elapsed: about `270s`

Accepted score on exported tensors:

- Run: `20260509_score_modelguided_top3r0_n64_nodes7_accepted`
- Top-1: `300/303 = 99.01%`
- Average target probability: `0.951149`

Accepted unforced replay/export:

- Run: `20260509_replay_modelguided_top3r0_n64_nodes7_accepted`
- Replay wins: `35/51`
- Matched decisions: `251/306 = 82.03%`
- Fresh first-deviation examples: `1`
- Anchored DAgger examples: `300`

Comparison:

- The earlier unguided top-3 n64/nodes31 corpus had `53/64` searched wins and accepted replay `31/53`.
- Model-guided n64/nodes7 keeps nearly the same teacher win count with fewer nodes and better accepted replay rate.
- The accepted policy already scores the model-guided tensors almost perfectly, so fitting these examples is unlikely to solve fresh-start play.

## Online Model-Guided Prefix Autopilot

Implementation:

- `TerminalPrefixSearch.Config` now has `modelGuidedFallback`.
- `ComputerPlayerRL` wires it through `RL_ONLINE_PREFIX_MODEL_GUIDED_FALLBACK`.
- When enabled, simulated suffixes use `ComputerPlayerRL.genericChoose` for non-tactic decisions instead of the prior uniform fallback.

Validation:

- RL module compile passed.

Run:

- `20260509_online_prefix_modelguided_trace_n7_cp1_eval4`
- Accepted profile, CP1, 16 games
- `RL_ONLINE_PREFIX_SEARCH_ENABLE=1`
- `RL_ONLINE_PREFIX_AUTOPILOT_ENABLE=1`
- `RL_ONLINE_PREFIX_MODEL_GUIDED_FALLBACK=1`
- `RL_ONLINE_PREFIX_SEARCH_MAX_NODES=7`
- `RL_ONLINE_PREFIX_SEARCH_MAX_DEPTH=6`
- `RL_ONLINE_PREFIX_SEARCH_TOP_K=3`
- `RL_ONLINE_PREFIX_SEARCH_MAX_ACTIVATIONS=6`
- `RL_ONLINE_PREFIX_SEARCH_TOTAL_TIMEOUT_MS=2000`
- `RL_ONLINE_PREFIX_SEARCH_BRANCH_TIMEOUT_MS=500`

Result:

- Total: `3/16`
- Spy mirror: `2/4`
- Jund Wildfire: `1/4`
- Mono Red Rally: `0/4`
- Grixis Affinity: `0/4`

Action health:

- `spy_cast_opportunities`: 31
- `spy_cast_hidden_land_opportunities`: 26/31
- `spy_casts`: 12
- `spy_casts_with_hidden_lands`: 9/12
- `dread_return_flashback_selected`: 18
- `premature_dread_flashback_not_combo_ready`: 18/18

Online prefix diagnostics:

- Logged prefix/autopilot events: 258
- Many searches timed out after 2-4 seconds despite a 2 second configured budget because model-guided suffix scoring is expensive.
- Found traces often started from the already selected action (`old == new`), so they queued long suffixes but rarely created a root override.
- Suffix forcing still missed live candidates frequently.

Conclusion:

Model-guided offline search is a better diagnostic teacher than unguided search at the same node budget, but it does not create a useful supervised target because the accepted policy already ranks those tensors correctly.

Online model-guided prefix search is not promotable: it is slower, still does not fix Spy/Dread timing, and does not beat the earlier n7 autopilot result.

## Longer Prefix Probe

Reason:

- The n64 model-guided corpus exported only 6 decisions per found win.
- Online model-guided traces often had 14-20 suffix actions, so replay losses might have been caused by untrained post-prefix drift.

Run:

- `20260509_spy_modelguided_top3r0_cp1_n32_nodes7_depth12`
- Accepted profile, CP1, 32 natural starts
- 7 nodes, `top_k=3`, model scoring enabled
- `MaxPrefixDepth=12`
- `TrainPrefixDepth=12`
- Collect-only, no heuristic rewards

Search result:

- Searched wins: `20/32`
- Exported decisions: `223`
- Elapsed: about `235s`

Accepted replay/export:

- `20260509_replay_modelguided_top3r0_n32_nodes7_depth12_accepted`
- Replay wins: `11/20`
- Matched decisions: `111/240 = 46.25%`
- First-deviation examples: `1`
- Anchored DAgger examples: `154`

Conclusion:

Depth-12 model-guided search is not a better training source at this budget. It exposes more drift, but teacher coverage drops and exact replay match collapses. The result supports the same diagnosis: terminal search can find lines, but converting those lines into a robust policy needs a different mechanism than prefix BC or anchored DAgger.

## Depth-12 Node-Budget Control

Reason:

- The depth-12/nodes7 result could have been a simple under-search artifact, so run one control with a larger node budget before abandoning this branch.

Run:

- `20260509_spy_modelguided_top3r0_cp1_n32_nodes15_depth12`
- Accepted profile, CP1, 32 natural starts
- 15 nodes, `top_k=3`, model scoring enabled
- `MaxPrefixDepth=12`
- `TrainPrefixDepth=12`
- Collect-only, no heuristic rewards

Search result:

- Searched wins: `24/32`
- Exported decisions: `267`
- Elapsed: about `170s`

Accepted replay/export:

- `20260509_replay_modelguided_top3r0_n32_nodes15_depth12_accepted`
- Replay wins: `16/24`
- Matched decisions: `146/288 = 50.69%`
- First-deviation examples: `3`
- Anchored DAgger examples: `246`

Conclusion:

The larger budget partially fixes teacher coverage, but the replay surface is still poor. Depth-12 traces are not a clean supervised target: the accepted policy can follow some of them to terminal wins, yet exact action agreement remains near coin-flip over the full prefix. Stop scaling model-guided prefix-search data unless the next implementation changes the policy-improvement mechanism.
