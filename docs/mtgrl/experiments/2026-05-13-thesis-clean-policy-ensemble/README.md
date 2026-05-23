# Thesis-clean policy ensemble diagnostic - 2026-05-13

## Question

Rejected thesis-clean branches sometimes improved one hard matchup while hurting another. This experiment tested whether their useful action-level signal survives as an eval-only probability ensemble with the accepted checkpoint, instead of destructive tensor interpolation.

This is thesis-clean as a diagnostic: the mechanism averages generic candidate-policy distributions and does not use card names, action regexes, Spy-specific terminal labels, heuristic step rewards, or selective MCTS.

## Implementation

Added optional bridge env vars:

- `POLICY_ENSEMBLE_MODEL_PATHS`: semicolon/comma/newline separated frozen companion checkpoint paths.
- `POLICY_ENSEMBLE_WEIGHTS`: optional weights. Length may be `companions` or `primary + companions`.

When configured, `scoreCandidatesPolicyFlat` averages the primary model's candidate probabilities and scalar value with the companion models, then renormalizes over valid candidates. Default behavior is unchanged when `POLICY_ENSEMBLE_MODEL_PATHS` is empty.

## Profiles

Registry:

- `local-training/local_pbt/thesis_clean/20260513_policy_ensemble_pressure_registry.json`

Eval-only profiles:

- `Pauper-Spy-Combo-Value-EnsembleSemFlags-20260513`
- `Pauper-Spy-Combo-Value-EnsembleBranchTraj-20260513`

Both use the accepted checkpoint as the primary model. The accepted checkpoint is hardlinked into each profile's `models/model_latest.pt`.

Weights:

- primary accepted checkpoint: `0.75`
- companion checkpoint: `0.25`

## Results

Semantic-effect-flags companion:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260513_policy_ensemble_semflags_pressure_g8`
- CP7 Rally: `2/8 = 25.00%`
- CP7 Grixis Affinity: `1/8 = 12.50%`
- Combined pressure screen: `3/16 = 18.75%`

Generic branch-trajectory-policy companion:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260513_policy_ensemble_branchtraj_pressure_g16`
- CP7 Rally: `7/16 = 43.75%`
- CP7 Grixis Affinity: `1/16 = 6.25%`
- Combined pressure screen: `8/32 = 25.00%`

Accepted pressure reference:

- CP7 Rally accepted reference: `16/64 = 25.00%`
- CP7 Grixis Affinity accepted reference: `13/62 = 20.97%`

## Verdict

Rejected. Probability ensembling did not recover a promotable policy. The branch companion improved Rally in the small screen, but it badly damaged Affinity. The semantic companion was worse on both pressure matchups.

The negative result is still useful: the failed tensor soups were not only a weight-space interpolation problem. The non-accepted branches appear to carry matchup-specialized policy preferences that do not combine cleanly with the accepted policy through simple probability averaging.

## Next

Stop spending local cycles on simple checkpoint combination unless a future branch first demonstrates a clear all-matchup advantage. Move back to mechanisms that can produce new generic policy improvement:

- train-time generic search targets;
- public-information/card-belief modeling;
- longer thesis-clean continuation from the accepted line;
- MCTS speed/scaling work only after a local small-budget gate clears.
