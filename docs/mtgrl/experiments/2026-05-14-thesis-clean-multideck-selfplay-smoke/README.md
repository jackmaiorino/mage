# Thesis-clean multi-deck self-play smoke - 2026-05-14

## Question

After the single-profile quick-variant surface closed, does returning to the
older thesis-clean multi-deck self-play recipe show any local signal on Spy's
hard CP7 pressure matchups?

This is a setup/signal experiment, not an HPC candidate. It uses isolated
D-backed profile clones so the canonical profile names are not mutated.

## Thesis Boundary

Clean:

- terminal win/loss rewards only;
- `RL_HEURISTIC_STEP_REWARDS=0`;
- no Spy terminal mode;
- no Spy candidate facts;
- no action-text/card-name filters;
- no MCTS/ISMCTS;
- four deck profiles trained together with `OPPONENT_SAMPLER=self` and
  `SELFPLAY_OPPONENT_TRAINING=1`.

## Setup

Registries:

```text
local-training/local_pbt/thesis_clean/20260514_multideck_selfplay_smoke_train_registry.json
local-training/local_pbt/thesis_clean/20260514_multideck_selfplay_smoke_eval_registry.json
```

Profiles are junctions under the normal profile root, backed by:

```text
D:\mtgrl-local-training\profiles\20260514_multideck_selfplay_smoke
```

Initial hashes:

| Profile | Start hash |
| --- | --- |
| Pauper-Spy-Combo-Value-MultiSelfplaySmoke-20260514 | `72857AA2975AB7434D7877B5CF49464D9CBEC927719BED8A2BD282E01879A66E` |
| Pauper-Wildfire-Value-MultiSelfplaySmoke-20260514 | `1CD7010F3A9E00B37F0AAEE29E684E5A7AE006DA3D3181EA7DAF1247ABBF70D6` |
| Pauper-Rally-Anchor-Value-MultiSelfplaySmoke-20260514 | `430D8A52F1D3C3D260507BC6E11E51585C0BD87E42AC1F21CD7F8C49D4A1B08F` |
| Pauper-Affinity-Anchor-Value-MultiSelfplaySmoke-20260514 | `9EAB7330ADB760FEB75AC2C13E75CED6EB12649E310F34CCC01FE22C83DE7F11` |

Spy was seeded from the immutable accepted eval snapshot:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_accepted_affinity_logged_cp7_g16/snapshot/rl/profiles/Pauper-Spy-Combo-Value/models/model_latest.pt
```

## Tranche 1

Training:

```text
local-training/local_pbt/20260514_multideck_selfplay_smoke_train128.out.log
TRAIN_PROFILES=4
NUM_GAME_RUNNERS=16
TOTAL_EPISODES_DELTA=128
MAX_WALL_SECONDS=3600
```

The run hit the wall-clock stop before all profiles reached target. Episode
counts:

| Profile | Episodes |
| --- | ---: |
| Spy | 92 |
| Wildfire | 66 |
| Rally | 128 |
| Affinity | 115 |

Resulting hashes:

| Profile | Hash |
| --- | --- |
| Spy | `A29F608E54EDCE1B7F5D6580A6A65499FF28735235FE471E8A3FB4E8A554F7E8` |
| Wildfire | `8C4D03AA270AAE12DAF9C4FC7BB3DD9A515DBADC4426E268DD4B7DD926EA5663` |
| Rally | `4F14DD1A9EA4CF779AD9120CA39F3883E85E0998FC0CC7E48515926EA7EDC529` |
| Affinity | `7ACFFF01D62A82E2F646AEECAE73819DB3DCB609D55E9B941510A533B9B221E4` |

Pressure gate:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_multideck_selfplay_smoke_pressure_g8
skill=7
opponents=Mono Red Rally,Grixis Affinity
games_per_matchup=8
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 2 | 8 | 25.00% |
| Grixis Affinity | 3 | 8 | 37.50% |
| Combined | 5 | 16 | 31.25% |

Reference fresh accepted pressure sample:

```text
Mono Red Rally: 5/8
Grixis Affinity: 2/8
combined: 7/16
```

## Decision

Continue locally, do not use HPC.

This does not clear the combined pressure gate, and the sample is small. But it
is the first thesis-clean local continuation in this tranche to improve the
Affinity point estimate instead of collapsing to `0/8` or `1/8`. The next step
is one more isolated local continuation tranche from these same junction
profiles, then the same Rally/Affinity pressure gate. If Affinity holds while
Rally recovers, expand locally; if not, reject this smoke without HPC.

## Tranche 2

The first tranche exposed an orchestrator mismatch: in multi-profile mode the
Java trainer can exit when one profile reaches the absolute episode target,
while other profiles remain below target. For the continuation, the stop
condition was changed to wall-clock time.

Training:

```text
local-training/local_pbt/20260514_multideck_selfplay_smoke_train_wall30m_tranche2.out.log
TRAIN_PROFILES=4
NUM_GAME_RUNNERS=16
TOTAL_EPISODES_DELTA=100000
MAX_WALL_SECONDS=1800
```

Episode counts after tranche 2:

| Profile | Episodes |
| --- | ---: |
| Spy | 532 |
| Wildfire | 365 |
| Rally | 773 |
| Affinity | 682 |

Resulting hashes:

| Profile | Hash |
| --- | --- |
| Spy | `D8042E8D72F8F26F6FBC52ED22E4B4C45F0D296B20D96D001D721069481B6ABC` |
| Wildfire | `5A69B9E0D9FF2BFDDA137529D7E1B258962AEC2F69BF0CBD72F6DA03959B645C` |
| Rally | `AD7B910B842EBF5CADD7DB3131C4511E273A955536237CA009326D42CA238FFA` |
| Affinity | `74B9A40C86F9B94755B88C331D8568C4EFB110705390BF4202E023C737C68F3C` |

Pressure gate:

```text
local-training/local_pbt/cp7_eval_sweeps/20260514_multideck_selfplay_smoke_tranche2_pressure_g8
```

| Opponent | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Mono Red Rally | 1 | 8 | 12.50% |
| Grixis Affinity | 2 | 8 | 25.00% |
| Combined | 3 | 16 | 18.75% |

## Final Decision

Rejected for promotion and HPC.

The initial Affinity bump did not survive another local tranche, and Rally
regressed. The experiment does validate D-backed isolated profile junctions and
wall-clock stopping for future multi-profile local work, but this small
multi-deck self-play continuation is not a scaling candidate under the 50 kSU
budget.
