# Affinity Replay Metadata Hook

Date: 2026-05-18

## Objective

Make the next accepted-policy CP7 Grixis Affinity compact-log sample convertible
into ActionCounterfactual replay/search probes. Older compact logs identify
decision ordinals and pressure contexts, but not the scenario seed and deck
ordering needed to reproduce the game state family.

## Changes

- `scripts/run_cp7_eval_sweep.py` now has `--replay-metadata` and
  `--replay-seed-base`.
- When enabled, each matchup chunk receives a stable seed-base offset and passes
  `EVAL_REPLAY_METADATA=1` plus `EVAL_REPLAY_SEED_BASE=<seed>` into league bench.
- `RLTrainer` league-bench evals now use ActionCounterfactual-compatible seeded
  deck order only when `EVAL_REPLAY_METADATA=1`, force that library order into
  the game, set `skipInitShuffling`, and log:

```text
REPLAY: scenario=<n> seed=<seed> agent_deck=<file> opp_deck=<file> action_counterfactual_compatible=true
```

- `scripts/build_affinity_replay_anchor_manifest.py` parses the new `REPLAY:`
  line and marks anchors replay-ready when scenario, seed, and deck names exist.

## Validation

```powershell
python -m py_compile scripts\run_cp7_eval_sweep.py scripts\build_affinity_replay_anchor_manifest.py
python scripts\build_affinity_replay_anchor_manifest.py
python C:\Users\Jack\.codex\skills\mage-research-agent\scripts\airl_maven.py compile
python scripts\run_cp7_eval_sweep.py --help
```

The current 2026-05-18 corpus still reports `Replay-ready anchors: 0/20`
because it was collected before the replay metadata hook existed.

## Launch Blocker

The detached controller attempted to launch the 4-game smoke, but this sandbox
could not create a fresh output directory under either:

- `local-training\local_pbt\cp7_eval_sweeps\20260518_affinity_replay_metadata_smoke_g4`
- `C:\tmp\mage-cp7-eval-sweeps`

No process was started. A script-only fallback validated
`parse_replay_metadata()` against a synthetic `REPLAY:` line containing
`scenario`, `seed`, `agent_deck`, `opp_deck`, and
`action_counterfactual_compatible=true`.

## Next Unit

Collect a small accepted-policy Affinity compact-log sample with replay metadata
enabled, then rebuild the corpus and manifest:

```powershell
python scripts\run_cp7_eval_sweep.py --registry local-training\local_pbt\thesis_clean\20260514_accepted_affinity_log_registry.json --run-id 20260518_affinity_replay_metadata_smoke_g4 --profiles Pauper-Spy-Combo-Value --opponents "Grixis Affinity" --games-per-matchup 4 --games-per-job 1 --parallel 1 --ai-threads 8 --eval-game-logging --game-log-format compact --replay-metadata --maven-offline --skip-compile
```

If league-bench execution is blocked, keep using the existing corpus and build
the replay CSV converter against synthetic metadata rows before attempting a
fresh run again.
