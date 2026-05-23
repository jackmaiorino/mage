# 24h Experiment Summary

Window: 2026-05-12 19:35 ET to 2026-05-13 19:35 ET

## Executive Summary

The last 24 hours were mostly a local thesis-clean triage pass over Spy policy-improvement surfaces after the accepted 2026-05-10 Affinity-pressure checkpoint. No Zaratan compute was used: two login-node checks reached Duo and timed out before approval, no Slurm job was submitted, and no kSU allocation was spent.

No branch should be promoted to HPC right now. Several useful infrastructure fixes landed, but every candidate either failed a reduced local gate or improved one slice while regressing another. The strongest positive signal was `Pauper-Spy-Combo-Value-BranchTrajPolicy-20260513`, which reached `10/16` on the first reduced CP7 gate and `29/64 = 45.31%` on the expanded CP7 gate, but it regressed Grixis Affinity to `2/16 = 12.50%` versus the accepted CP7 Affinity reference of `13/62 = 20.97%`. That makes it diagnostic, not promotable.

Accepted reference for comparison:

```text
CP1: 133/256 = 51.95%
CP3: 132/256 = 51.56%
CP7: 108/242 = 44.63%
CP7 Grixis Affinity: 13/62 = 20.97%
```

## Main Conclusions

1. The short local variant surface remains exhausted. AWR selected-action targets, reference-anchored terminal continuation, representation-only features, value-only branch calibration, branch-pair ranking, deep mulligan distillation, and repaired MCTS all failed local gates.

2. Search can find thesis-clean winning lines, but the current training operators do not transfer those lines into fresh games. Corrected prefix trajectory collection produced CP7 winning trajectories, yet distillation and trajectory-RL expanded screens stayed below the accepted checkpoint. Many searched-prefix corpora were already solved by accepted at `93-100%` top-1.

3. The first genuinely unsolved offline teacher signal came from the baseline-losing-alternative filter. It found accepted-policy misses at `0/22` and `0/66` top-1, but direct imports and replay/DAgger still failed fresh CP7 gates. The filter is valuable diagnostic infrastructure, not yet a working policy-improvement operator.

4. MCTS had a real root action-index bug, now fixed, but repaired MCTS is still not useful locally. Root mapping validated cleanly, yet multi-ply/top-K MCTS went `0/4` into CP7 Affinity, flat top-K MCTS went `1/12`, and Rally top-K MCTS went `1/4` at high wall-clock cost.

5. The value head is a central blocker. On comparable baseline-losing-alternative branch pairs, the accepted value head preferred the winning branch only `1/4` times, with winning rows averaging `0.187849` and losing rows averaging `0.222264`. Value-only calibration and pair-rank losses did not repair this without hurting policy behavior.

6. The Affinity and Rally failures are not solved by one local patch. Affinity needs better interaction/pressure handling; Rally is not just a deep-mulligan issue. Larger Rally prefix search found winning trajectories, but accepted already scored the exported labels `60/60` top-1, and live receding search still lost.

## Candidate Results

| Surface | Best signal | Local gate result | Decision |
| --- | --- | --- | --- |
| AWR selected-action targets | Mechanically wired terminal-advantage weighting | Partial CP7 `22/54 = 40.74%`, Rally `1/16` | Reject |
| Card-belief infrastructure | Head-only loss learned and determinization path activated | Auxiliary RL unstable: `11/16` at 128ep, then `5/16` after extension; belief-MCTS destructive | Keep infrastructure, reject promotion |
| Reference-policy anchor | Generic frozen-reference KL works mechanically | Hard/curriculum anchor probes failed reduced CP7; hard continuation `5/16` | Reject |
| Corrected prefix trajectory teacher | Harness fix restored `128/2` model defaults and produced CP7 winning trajectories | Expanded CP7 screens `27/64`, `21/64`, `21/64`; Affinity-only KL `1/16` | Reject current transfer operator |
| Online prefix search | Deadline swallowing fixed; generic branch-order mode added | Generic Affinity `0/2`, Rally `0/2`; larger Rally budget found a prefix but live autopilot still lost | Reject for scaling |
| Baseline-losing-alternative teacher | Found unsolved labels: accepted `0/22` and `0/66` top-1 | Direct import `5/16`; replay/DAgger `3/16` | Keep filter, reject direct imports |
| MCTS root mapping and top-K | Root action mapper validated with all hits | Multi-ply Affinity `0/4`, flat Affinity `1/12`, Rally `1/4` at high cost | Keep fix, reject MCTS scaling |
| Branch value calibration | Proved value misranking on branch states | Value-only imports `6/16` or `7/16`, Affinity `0/4` | Reject current value-only family |
| Branch-pair value ranking | Pair export and rank loss work mechanically | Shared import `5/16`; critic-only `7/16`; MCTS after critic calibration `0/2` | Reject |
| Semantic effect flags | Exile/damage effect flags improved generic public-card extraction | Unanchored Affinity `5/15` but CP1 `30/64`; anchored CP1 `34/64` but CP7 Affinity `3/16` | Keep default-off, reject |
| Public board features | Generic battlefield pressure stats added | Eval-only stable `8/16`; trained clone `7/16`, Affinity `1/4` | Keep default-off, reject |
| Deep mulligan line distillation | Confirmed deep-mull failures are real | Smoke CP7 `7/15`, Jund `0/3`; larger corpus showed Rally `0/48` line wins | Reject quick fix |
| Rally big-prefix teacher | Larger generic search found 8 winning Rally trajectories | Accepted scored exported labels `60/60` top-1; fixed policy-miss export selected `0` examples | Reject training |
| Baseline-alt branch trajectory policy | First reduced CP7 gate `10/16` with Affinity `3/4` | Expanded CP7 `29/64`, but Affinity `2/16` | Reject promotion; best follow-up lead |

## Useful Infrastructure Added

- Corrected stale `MODEL_D_MODEL=256` / `MODEL_NUM_LAYERS=4` wrapper defaults back to the canonical thesis-clean `128/2` shape.
- Added generic frozen-reference policy KL anchoring.
- Added card-belief labels, head training, transport, inference, and determinization plumbing.
- Added corrected trajectory export/import paths and branch trajectory diagnostics.
- Added baseline-losing-alternative filtering to focus on accepted-policy losing decisions with winning siblings.
- Fixed online prefix search deadline control flow and added thesis-clean generic branch ordering.
- Fixed multi-ply MCTS root action mapping and added default-off root top-K gating.
- Added public semantic effect flag coverage and public board aggregate features.
- Added branch value probes, branch-pair export, pair-rank loss, and critic-only containment.
- Fixed `-PolicyMissOnly` so collect/export selection honors the policy-miss filter, not just import/training.

## Recommended Next Step

Repeat the baseline-alt branch trajectory policy mechanism, but collect under Affinity pressure from the accepted checkpoint and enforce opponent-balanced or Affinity-heavy example selection before training. Gate first on CP7 Affinity, then run the expanded four-opponent CP7 screen only if Affinity holds above the accepted reference.

Avoid HPC until a local branch clears both requirements:

```text
1. aggregate CP7 at or above accepted,
2. no regression in Grixis Affinity versus accepted.
```
