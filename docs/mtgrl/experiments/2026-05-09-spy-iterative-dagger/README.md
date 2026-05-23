# 2026-05-09 Spy Iterative DAgger

Goal: test whether a real second DAgger cycle helps where one-shot DAgger failed. The hypothesis was that the first DAgger fit created new deviations, so a second replay/export/fine-tune pass might repair them better than another one-off supervised patch.

## Cycle 2 Export

Teacher corpus:

- `20260509_spy_shallow_search_top3r0_cp1_n128_nodes31`
- 111 terminal-winning top-3 teacher starts
- Replay file: `winning_trajectories.csv`

Policy replayed:

- `Pauper-Spy-Combo-Top3R0DaggerR16-20260509`

Run:

- `20260509_replay_top3r0_dagger_r16_iter2_export_r16`
- `ReplayDeviationRepeat=16`
- `MaxDecisionDepth=8`
- 12 workers, 24 AI threads

Result:

- Replay wins: `62/111`
- Matched decisions: `525/666 = 78.83%`
- Fresh first-deviation examples: `22`
- Anchored DAgger examples: `930`

## Iter2 Fit

Profile:

- `Pauper-Spy-Combo-Top3R0DaggerR16Iter2-20260509`
- Seeded by copying `Pauper-Spy-Combo-Top3R0DaggerR16-20260509`

Dataset:

- `20260509_top3r0_dagger_iter2_agg_data`
- Hardlinked cycle 1 DAgger data plus cycle 2 DAgger data.

Fit:

- Run: `20260509_top3r0_dagger_iter2_agg_fit_e2p1`
- Imported examples after skip filters: `1750`
- Train passes: `3500`
- Terminal reward only, no heuristic rewards.

Score probe:

- Run: `20260509_score_top3r0_dagger_iter2_agg_e2p1`
- Examples scored: `1822`
- Top-1: `1429/1822 = 78.43%`
- Average target probability: `0.656425`

Clean same-start replay:

- Run: `20260509_replay_top3r0_dagger_iter2_agg_e2p1_unforced_clean`
- Replay wins: `74/111`
- Matched decisions: `443/666 = 66.52%`

This improves same-start wins versus the one-shot DAgger profile (`71/111`) and accepted checkpoint replay on the same corpus (`70/111`), but it does so by drifting farther from the exact teacher line.

## Reduced CP1 Eval

The normal `cp7_eval_sweeps` path is a junction to a nearly full `D:` volume, so `scripts/run_cp7_eval_sweep.py` now supports `--output-root` / `CP7_EVAL_SWEEP_ROOT`. The eval below was written to `local-training/local_pbt/cp7_eval_sweeps_cdrive`.

Run:

- `20260509_top3r0_dagger_iter2_cp1_eval4`
- CP1, 4 games per matchup, 16 games total
- No MCTS, no heuristic rewards.

Result:

- Total: `1/16`
- Spy mirror: `1/4`
- Jund Wildfire: `0/4`
- Mono Red Rally: `0/4`
- Grixis Affinity: `0/4`

Conclusion:

Iterative DAgger is not promotable in this form. It can improve replay wins on the saved teacher starts, but the unforced live distribution still collapses. The exact-line match drop is a warning sign: the model is finding alternate behavior on the saved starts, not learning a robust policy for fresh CP1 games.

Do not continue simple aggregate DAgger cycles without changing the data source or the policy-improvement mechanism.
