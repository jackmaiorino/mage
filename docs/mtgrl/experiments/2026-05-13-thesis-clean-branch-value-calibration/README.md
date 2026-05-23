# Thesis-Clean Branch Value Calibration

Date: 2026-05-13

## Question

MCTS and prefix search are only useful if the value head ranks branch states well enough to act as a leaf evaluator. The previous MCTS probes failed even after fixing root action mapping, so the next question was whether accepted `Pauper-Spy-Combo-Value` already assigns higher values to generic winning forced branches than to generic losing forced branches.

## Branch Value Probe

Implemented `--branch-value-probe` in `ActionCounterfactualTrainer`. For baseline-losing-alternative examples, it replays the forced baseline and alternative branches, then scores the resulting branch state with the existing value head. This uses only terminal win/loss labels and generic action candidates.

Runs:

- `local-training/local_pbt/action_counterfactual/20260513_branch_value_probe_cp7_smoke16`
- `local-training/local_pbt/action_counterfactual/20260513_branch_value_probe_cp7_smoke8_fixed`

The first probe found 9 records and 4 comparable win/loss decision pairs. The value head preferred the winning branch in only 1/4 pairs. Among rows with finite values, winning branches averaged `0.187849`, losing branches averaged `0.222264`.

The first implementation used an internal termination signal that was caught by `ComputerPlayerRL.priority()`, causing warning spam and slow continuation. The fixed implementation stops cleanly; the smaller fixed validation had too few comparable pairs, but confirmed the instrumentation no longer floods logs.

Conclusion: current value estimates are not reliable enough to justify scaling generic MCTS. The failure is not just branching factor or root action indexing.

## Branch Trajectory Export

Implemented `--branch-trajectory-mode`. Forced branches can now be exported as ordinary terminal-return trajectory episodes via the existing serialized trajectory path. This is thesis-clean: no Spy terminal mode, no card-name regex, no action-text filter, no heuristic step rewards.

Wrapper additions in `scripts/run_action_counterfactual.ps1`:

- `-BranchTrajectoryMode`
- `-ExportTrajectoryDataFile`
- `-ImportTrajectoryDataPath`
- `-TrajectoryFinalReward`
- `-PolicyLossCoef`
- `-ValueLossCoef`
- `-UseMcReturns`

Collection:

- Run: `20260513_branch_traj_valuecal_collect8`
- Scenarios: 8
- Branch records: 82
- Candidate examples: 5
- Serialized branch trajectories: 52 episodes, `145 MB`

Value-only import:

- Profile clone: `Pauper-Spy-Combo-Value-BranchTrajValue-20260513`
- Source: accepted `Pauper-Spy-Combo-Value`
- Import run: `20260513_branch_traj_valuecal_import_e1`
- Training: policy loss `0`, value loss `2`, MC returns, 1 epoch, capped at 512 steps
- Imported: 26 episodes, 525 decision steps

CP7 smoke:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260513_branch_traj_valuecal_cp7_g4`
- Overall: 7/16 = 43.75%
- Affinity: 1/4
- Jund: 2/4
- Rally: 1/4
- Spy mirror: 3/4

This is neutral against the accepted CP7 anchor (108/242 = 44.63%), and the sample is too small for promotion. It does validate that branch-trajectory value-only training can run without immediate policy collapse.

## Decision

Do not promote the value-calibrated profile. Keep the infrastructure, but reject this first value-only calibration recipe.

## Larger Local Pass

Collection:

- Run: `20260513_branch_traj_valuecal_collect24`
- Scenarios: 24
- Branch records: 305
- Serialized branch trajectories: 185 episodes, `566 MB`

Value-only import:

- Run: `20260513_branch_traj_valuecal_import24_e2`
- Profile was reset from accepted before import.
- Requested 2 epochs, capped at 2048 steps; importer cap is global across epochs, so effective training was one pass.
- Imported: 93 episodes, 2056 decision steps

CP7 smoke:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260513_branch_traj_valuecal_import24_cp7_g4`
- Overall: 6/16 = 37.50%
- Affinity: 0/4
- Jund: 2/4
- Rally: 2/4
- Spy mirror: 2/4

This regresses the accepted CP7 anchor and especially the weak Affinity matchup. The naive value-only branch-trajectory import is rejected.

## Next

The useful result is diagnostic, not a candidate model: branch states expose value misranking, but blunt value-only fine-tuning on all forced branch trajectories degrades Affinity. The next value-target experiment should be more selective: train on paired win/loss branch states or decision-local contrastive value targets, not every downstream state in noisy forced trajectories. Only after value calibration improves CP7 or branch-ranking diagnostics should MCTS scaling be reconsidered.

## First-Post-Target Variant

To reduce downstream rollout noise, `--branch-trajectory-first-post-target-only` was added. It exports at most the first later `ACTIVATE_ABILITY_OR_SPELL` decision state after the forced target action, labeled by the forced branch terminal result.

Collection:

- Run: `20260513_branch_traj_firstpost_collect24`
- Scenarios: 24
- Branch records: 313
- Serialized branch trajectories: 185 one-step episodes, `28 MB`

Value-only import:

- Run: `20260513_branch_traj_firstpost_import_e4`
- Profile was reset from accepted before import.
- Imported: 185 one-step episodes, 740 train passes over 4 epochs

CP7 smoke:

- Run: `local-training/local_pbt/cp7_eval_sweeps/20260513_branch_traj_firstpost_cp7_g4`
- Overall: 7/16 = 43.75%
- Affinity: 0/4
- Jund: 2/4
- Rally: 2/4
- Spy mirror: 3/4

This is still not useful because the weakest accepted matchup remains worse. Reject the current value-only branch-state calibration family.

Final decision for this line: keep the instrumentation, but do not scale on HPC and do not continue local variants without a different value objective.
