# Spy Zone-Count Feature Probe

Date: 2026-04-29

## Question

Can generic zone-count features help Spy Combo learn when `Balustrade Spy` is safe to cast, while keeping training terminal-reward-only?

The prior 80,669 checkpoint usually cast `Balustrade Spy` when available, but logged health metrics showed every Spy cast happened with at least one true land still estimated in library. That means the model was firing Spy before the library was prepared, so `Dread Return` often had no `Lotleth Giant` target.

## Implementation

Added an opt-in feature gate in `StateSequenceBuilder`:

- `RL_ZONE_COUNT_FEATURES_ENABLE=1`
- player-stat slots 17-22:
  - hand land count
  - hand creature count
  - library land count
  - library creature count
  - graveyard land count
  - graveyard creature count

These are generic zone counts, not Spy-specific rules, action constraints, or reward shaping. Default remains off so existing checkpoints are behaviorally unchanged unless a run explicitly enables the gate.

Before training, the new input columns 17-22 were zeroed in `input_proj.weight` and `critic_input_proj.weight` for the active Spy checkpoint to avoid random unused weights perturbing behavior.

Backups:

- pre-zero baseline: `local-training/local_pbt/model_backups/20260429_spy_80669_pre_zonecount_zero`
- post-training checkpoint: `local-training/local_pbt/model_backups/20260429_spy_90669_zonecounts_cp1heavy`

## Training

Run:

- `20260429_spy_zonecounts_cp1heavy_to90669_r144`

Shape:

- start: 80,669 episodes
- target: 90,669 episodes
- profile: `Pauper-Spy-Combo-Value`
- runners: 144
- CP1-heavy hybrid:
  - `OPPONENT_SAMPLER=meta_hybrid`
  - `META_HYBRID_META_P=0.25`
  - `SKILL_MIX=1:1.0`
- `RL_ZONE_COUNT_FEATURES_ENABLE=1`
- terminal reward only

Throughput:

- CPU generally 93-100%
- GPU memory stable around 4.7-6.8 GB
- eps/s usually 1.4-2.4, with brief restart bursts
- trainer restarted twice with `rc=1`; orchestrator recovered automatically
- run reached exactly 90,669 and shut down cleanly

## Results

Pre-training feature-on smoke:

- 40-game CP1 eval: `20260429_spy_80669_zonecounts_on_cp1_eval_g10_p16`
- result: 12/40 = 30.0%

Post-training CP1 eval:

- run: `20260429_spy_90669_zonecounts_cp1_eval_g10_p16`
- result: 10/38 = 26.3%
- one Grixis chunk returned 0 games, but even ignoring that, the direction was negative

Post-training logged health:

- run: `20260429_spy_90669_zonecounts_cp1_combo_health_g5_p24`
- result: 4/20 = 20.0%
- `spy_cast_opportunities`: 12
- `spy_casts`: 10/12 = 83.3%
- `spy_cast_no_hidden_lands_opportunities`: 0/12 = 0.0%
- `spy_casts_no_hidden_lands`: 0/10 = 0.0%
- `spy_cast_hidden_lands_est_avg`: 3.42
- `dread_return_target_no_lotleth_available`: 3/6 = 50.0%
- `dread_return_target_lotleth_when_available`: 3/3 = 100.0%

Weight movement after training:

- `input_proj.weight[:,17:23]` moved from 0 to mean abs ~0.010
- `critic_input_proj.weight[:,17:23]` stayed exactly 0

## Conclusion

Negative.

The model did learn some actor-side weights for the new count features, but the critic did not use them, and live play still cast every Spy with lands estimated in library. Aggregate CP1 performance regressed relative to the restored 80,669 checkpoint.

This does not disprove representation features in general, but this specific terminal-only fine-tune did not create the needed value/timing signal. The active model was restored to the stronger 80,669 CP1-heavy checkpoint after this experiment.

## Next Implication

The immediate Spy failure is not "does the action head know to cast Spy"; it does. The failure is temporal:

- prepare the library first,
- then cast Spy,
- then flash back Dread only when Lotleth is available.

More generic PPO from the current distribution is unlikely to solve this quickly because safe Spy states are rare in the model's own rollouts. The next experiment should target state distribution, not just feature visibility.
