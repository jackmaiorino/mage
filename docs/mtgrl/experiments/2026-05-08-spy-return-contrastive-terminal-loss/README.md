# Spy Return-Contrastive Terminal Loss

Date: 2026-05-08

## Goal

Test whether a terminal-only contrastive policy loss can teach Spy Combo timing from normal RL trajectories:

- push up selected critical spell/mulligan actions from won trajectories;
- push down high-probability selected critical spell/mulligan actions from lost trajectories;
- keep the existing terminal reward contract intact.

This was motivated by the separate-critic result: the value head had useful ranking signal, but the policy still cast `Balustrade Spy` into hidden lands and flashbacked `Dread Return` too early.

## Setup

Source checkpoint:

- `Pauper-Spy-Combo-FastT5SeparateCriticRL-20260508`

Experiment checkpoint:

- `Pauper-Spy-Combo-FastT5ReturnContrastive-20260508`

Training run:

- `local-training/local_pbt/value_rl/20260508_spy_fastt5_return_contrastive_r24_b4`

Training details:

- 5,000 generated episodes
- 24 runners
- hybrid opponent sampler: 70% self-play, 30% CP1
- terminal rewards only
- separate critic encoder enabled
- `RETURN_CONTRASTIVE_POLICY_LOSS_COEF=0.50`
- `RETURN_CONTRASTIVE_CRITICAL_ONLY=1`
- critical action definition: selected spell candidate or mulligan head

## Results

Full CP1 eval:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_fastt5_return_contrastive_5k_cp1_eval32`
- Result: `0/128`

| Matchup | Wins | Games | Winrate |
| --- | ---: | ---: | ---: |
| Spy mirror | 0 | 32 | 0.00% |
| Jund Wildfire | 0 | 32 | 0.00% |
| Mono Red Rally | 0 | 32 | 0.00% |
| Grixis Affinity | 0 | 32 | 0.00% |
| Total | 0 | 128 | 0.00% |

Logged CP1 eval:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260508_spy_fastt5_return_contrastive_5k_cp1_logged8`
- Result: `0/32`
- Analysis: `local-training/analysis/value_head_20260508_return_contrastive_logged8`

Action health:

| Metric | Result |
| --- | ---: |
| Mulligan keep | 32/67 decisions |
| Kept hands with 0-1 land | 28/32 |
| Land play opportunities | 364 |
| Land play selected | 0/364 |
| Pass over land | 364/364 |
| Spy cast opportunities | 0 |
| Dread Return flashbacks | 0 |

Value-head AUC could not be computed because all 32 logged games were losses.

## Interpretation

The loss was destructive. It did not learn "do not cast Spy yet"; it escaped by becoming unable to execute basic development, especially land drops.

This is different from the earlier separate-critic checkpoint, which played lands normally but fired combo actions at the wrong time. The contrastive policy loss directly perturbed action probabilities without an action-conditioned value model, so negative terminal signal was too blunt.

## Conclusion

Do not continue this variant at the current formulation.

The next experiment should train an explicit candidate value/Q head from terminal returns, then use it to rerank actions. That keeps the terminal-only thesis but changes the mechanism from "push policy away from selected lost actions" to "learn Q(s,a), then choose high-Q actions."
