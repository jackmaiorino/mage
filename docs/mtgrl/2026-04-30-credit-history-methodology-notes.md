# MTGRL Credit Assignment / History / Methodology Notes

Date: 2026-04-30

Current run context: the active local experiment is still the two-GPU terminal-only self-play/eval cycle. These notes are for the next method change; they are not being applied mid-run.

## Synthetic Returns

Paper reviewed: Raposo et al., "Synthetic Returns for Long-Term Credit Assignment", arXiv 2102.12425, https://arxiv.org/pdf/2102.12425.

Relevant claim: the paper adds state-associative learning to an actor-critic agent. A separate model learns associations between earlier states and later rewards, then uses those learned associations as synthetic returns for TD-style policy learning. The motivation is exactly our pain point: TD/GAE has to propagate credit one step at a time and gets noisy when important actions are far from terminal reward.

Fit for MTGRL:

- This is compatible with the terminal-only philosophy if synthetic returns are treated as learned credit targets derived from terminal outcomes, not as hand-written game rewards.
- Spy Combo is a strong target case. "Keep hand with Spy + mana plan", "resolve mana conversion", "cast Spy", and "choose/resolve kill line" are sparse states that should explain a terminal win several actions later.
- Wildfire is less binary but still has long credit chains: early mana development, bounce/land destruction setup, and turning tempo into terminal wins.
- Our implementation may be simpler than the paper's distributed IMPALA setup because we already enqueue full per-episode trajectories and terminal outcomes.

Risks:

- The paper calls out additive regression credit ambiguity: if multiple earlier states predict the same terminal win, the model may assign credit to arbitrary correlated states.
- Gated synthetic returns were sensitive in the Atari Skiing case. We should expect tuning sensitivity and avoid making it a large code change before a small offline probe.
- If we train synthetic returns from a weak policy's failures only, it can reinforce the same midrange minima unless we include wins, counterfactual branches, or line-search generated successes.

Recommended v0:

1. Add an offline credit probe before touching online training.
2. Export trajectories with state/action metadata, chosen action, terminal outcome, and profile/deck labels.
3. Train a small auxiliary credit model over frozen state embeddings to predict terminal outcome and produce per-decision contribution scores.
4. Validate on held-out episodes: contribution spikes should align with plausible decisions such as mulligan keep/bottom choice, first land/mana setup, key cast/activation, and lethal line selection.
5. Only then wire contribution scores into the trainer as an optional advantage/value target blend behind an env flag.

Success gate:

- Better held-out terminal prediction than the current value head.
- Contribution attribution is not dominated by trivial late-game/pass artifacts.
- A short Spy/Wildfire training run improves CP1 or self-play eval faster than the current baseline at similar throughput.

## Previous-State Context

Current model context:

- `StateSequenceBuilder` builds a single current snapshot: phase, player/opponent stats, hand, battlefield, graveyard, own library, opponent battlefield/graveyard, exile, and stack.
- The transformer sees up to 256 state tokens plus candidate action features.
- There is no explicit stack of prior decision states in the main policy input.

Assessment:

- Full raw state stacking like AlphaGo is not automatically right here. MTG's public/current state is much closer to Markov than a single Go board frame, and raw stacking would quickly hit the 256-token cap, causing truncation.
- Some history is still valuable: hidden-information inference, repeated-action context, "what line am I on", and credit for choices whose consequences are not obvious from the current board.

Recommended v0:

- Add compact history tokens, not full prior snapshots.
- Keep the last 2-4 decision summaries: action type, chosen candidate id, turn/phase, mana/life/hand/library deltas, and maybe a hashed chosen-card/action id.
- Make it opt-in with an env flag and benchmark throughput before any long run.

Success gate:

- No material increase in state truncation.
- Less "pass over play land" / non-progress action collapse in action-health logs.
- Measurable eval lift on Spy/Wildfire before the same episode count.

## New Methodology Direction

We should not invent from a blank slate yet. The credible path is a terminal-only MTG-specific training methodology built from:

- PPO/actor-critic on terminal outcomes.
- Full-episode self-play on both sides.
- Counterfactual/line-search data for rare successful combo lines.
- Learned credit assignment from terminal outcomes, starting with a synthetic-returns-style offline probe.
- Compact temporal context if attribution/history diagnostics show the current snapshot loses useful information.

This can still become a novel methodology for MTG deck learning, but the next step should be a controlled ablation, not an unbounded architecture rewrite.

Next candidate experiment after the current two-GPU cycle:

1. Run the current checkpoint through CP1/CP3 gates and action-health diagnostics.
2. Export 5k-20k Spy/Wildfire trajectories, preserving full decision order.
3. Train the offline credit probe and inspect contribution rankings.
4. If attribution is sane, run `terminal + synthetic-credit blend` against the same baseline.
5. If attribution is noisy, try compact history tokens first, then repeat the probe.

## First Offline Probe Results

Date: 2026-04-30

Data:

- Source: recent logged Spy/Wildfire self-play text logs.
- Export: `local-training/analysis/spy_wildfire_recent_trajectories.jsonl`
- Size: 40 games, 18,843 total decision records.
- Actor-filtered probe data: 9,261 `PlayerRL1` decision records.

Simple event lift:

- Baseline PlayerRL1 winrate in this sample: `12/40 = 30%`.
- `has_dread_return_option`: `4/10 = 40%`, lift `+10%`.
- `selected_cast_dread_return`: `3/8 = 37.5%`, lift `+7.5%`.
- `has_cast_spy_option`: `4/13 = 30.8%`, about baseline.
- `selected_cast_spy`: `1/10 = 10%`, lift `-20%`.

Linear terminal-outcome probe:

- Script: `scripts/train_synthetic_return_probe.py`
- No-history probe:
  - Features: 38
  - Held-out AUC: `0.611`
  - Held-out BCE: `0.688` vs base-rate BCE `0.673`
- Compact-history probe with last 4 decision summaries:
  - Features: 50
  - Held-out AUC: `0.628`
  - Held-out BCE: `0.686` vs base-rate BCE `0.673`

Interpretation:

- This is not strong enough to wire into training yet.
- It is enough to justify a larger trajectory export and a better credit model, because the probe is learning non-random signal from terminal outcomes.
- Compact previous-decision context helped. That supports testing history tokens, but only in compact form, not full raw state stacking.
- Spy's current problem is not merely "learn to cast Balustrade Spy." In the sample, cast-Spy alone correlates negatively with wins, while Dread Return availability/casting correlates positively. The credit target must learn complete conversion lines.

## Larger Accepted-Spy Logged Probe

Date: 2026-05-07

Data:

- Source: `20260507_spy_fastt5_accepted_cp1_logged128`
- Profile: `Pauper-Spy-Combo-FastT5Contrast-20260506`
- CP1 logged games: 128
- Result: 24 wins, 104 losses
- Export: `local-training/analysis/synthetic_return_20260507_logged128/accepted_logged128_trajectories.jsonl`
- Decisions exported: 16,483

Parser fix:

- Eval game logs use `RESULT: WIN/LOSS`, not `Winner: PlayerRL1`.
- `scripts/export_game_log_trajectories.py` now reads that result line. Before this fix, the exporter mislabeled every eval log as a loss.

Probe update:

- `scripts/train_synthetic_return_probe.py` now adds visible state-count and key-card presence features from `state_summary`.
- This is still an offline diagnostic only.

Results:

| Feature Set | History Window | Test AUC | Test Loss | Base Loss |
| --- | ---: | ---: | ---: | ---: |
| action-only | 0 | 0.5642 | 0.6838 | 0.5191 |
| action-only | 4 | 0.5503 | 0.6920 | 0.5191 |
| action-only | 8 | 0.5364 | 0.6978 | 0.5191 |
| action-only | 16 | 0.5179 | 0.7015 | 0.5191 |
| state + action | 0 | 0.6258 | 0.8445 | 0.5191 |
| state + action | 4 | 0.6266 | 0.8523 | 0.5191 |
| state + action | 8 | 0.6262 | 0.8607 | 0.5191 |
| state + action | 16 | 0.6324 | 0.8606 | 0.5191 |

Interpretation:

- Visible state features add meaningful ranking signal, but calibration is poor and held-out BCE is worse than the base-rate predictor.
- Lower learning rate and fewer epochs reduced overconfidence but also reduced AUC; best lower-LR AUC was only `0.5761`.
- Compact history did not improve this sample enough to justify wiring history into online training yet.
- Do not train synthetic returns from the current linear probe. The next terminal-only step should collect branch counterfactual labels at critical states reached from natural CP1 starts, because logged failures are dominated by premature Spy and Dread Return timing.
