# Repository Guidelines

## Scope
This guide applies to the MageRL plugin module at `Mage.Server.Plugins/Mage.Player.AIRL`.

## Project Structure & Module Organization
- Java entry points:
  - `src/mage/player/ai/ComputerPlayerRL.java` (runtime RL player behavior)
  - `src/mage/player/ai/rl/RLTrainer.java` and `DraftTrainer.java` (training loops)
- Python bridge/model code: `src/mage/player/ai/rl/MLPythonCode/` (`py4j_entry_point.py`, model/training utilities).
- Data/config assets:
  - Deck inputs in `src/mage/player/ai/decks/` (`Pauper`, `PauperSubset`, `Vintage`)
  - League/profile state in `src/mage/player/ai/rl/league*` and `src/mage/player/ai/rl/profiles/`
  - Design and run docs in `docs/`.

## Build, Train, and Evaluation Commands
- Compile module only:
  - `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`
- Start training:
  - `$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"`
  - `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train"`
- Run evaluation:
  - `$env:EVAL_EPISODES="20"; $env:MODE="eval"`
  - `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=eval"`
- Scripted workflow from repo root:
  - `powershell -File scripts/rl-train.ps1 -Profile perf -TotalEpisodes 2000 -NumGameRunners 16`
  - `powershell -File scripts/rl-stop.ps1`

## Zaratan HPC Workflow (Slurm)
- Prefer `scripts/hpc/` wrappers for cluster runs (`discover_slurm.sh`, `build_venv.sh`, `submit_spy_pbt.slurm`, `run_spy_pbt.sh`).
- Submit jobs from login node; do not run training loops directly on login nodes.
- For GPU jobs, specify partition in the batch script (for example `#SBATCH --partition=gpu` or `gpu-a100`/`gpu-h100`) and request GPU resources (for example `#SBATCH --gres=gpu:1`).
- Avoid forcing `--qos=...` unless your Slurm association explicitly includes that QoS; unauthorized QoS values fail with `Invalid qos specification`.
- Before launching long runs, verify access and cluster state:
  - `sacctmgr show assoc where user=$USER format=User,Account,Partition,QOS,GrpTRESMins,GrpTRES,MaxTRES,MaxWall -P`
  - `scontrol show partition`
  - `sinfo -o "%P %a %l %D %c %m %G"`

## Coding Style & Naming Conventions
- Java: Java 8-compatible, 4-space indentation, `PascalCase` classes, `camelCase` methods/fields.
- Python: PEP 8-style formatting, `snake_case` functions/modules.
- Add new runtime knobs via `EnvConfig` and environment variables; avoid hard-coded local paths.
- Keep concurrency-safe logic explicit (`Atomic*`, thread-local state, synchronized boundaries where needed).

## Testing & Validation Guidelines
- There are currently no dedicated unit tests under this module's `src/test`; validate via targeted smoke runs.
- Minimum check before PR:
  - Compile succeeds.
  - Short eval run (`EVAL_EPISODES=10-20`) completes.
  - For training changes, run a short train pass (`TOTAL_EPISODES=20-100`) and confirm no bridge/runtime errors.

## Commit & Pull Request Guidelines
- Prefer concise, imperative subjects, typically RL-scoped (example: `RL: Fix activation filtering in chooseMode`).
- Keep commits focused (trainer logic, bridge changes, deck updates, or scripts; not all mixed).
- PR description should include:
  - Behavioral change summary
  - Exact commands used for validation
  - Any key metric deltas (winrate, episode throughput, stability)
- Do not commit generated artifacts/logs/venvs/models (for example under `src/mage/player/ai/rl/models`, `profiles/*/models`, `profiles/*/logs`, `MLPythonCode/venv`).
