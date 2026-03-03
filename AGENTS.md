# Repository Guidelines

## Project Structure & Module Organization
- This repository is a multi-module Maven project (`pom.xml`) centered on XMage.
- Core modules: `Mage` (engine), `Mage.Common`, `Mage.Server`, `Mage.Client`, and `Mage.Sets`.
- Extension/reporting modules: `Mage.Plugins`, `Mage.Server.Plugins`, `Mage.Reports`, `Mage.Verify`.
- Tests live primarily in `Mage.Tests` (`src/test/java/org/mage/test/...`).
- Supporting folders include `db/` (local H2 data), `scripts/` (automation and RL workflows), `docs/`, `k8s/`, and `monitoring/`.

## Build, Test, and Development Commands
- `mvn clean`: remove build outputs across modules.
- `mvn install package -DskipTests`: full reactor build (also used by `make build`).
- `mvn test`: run all tests through Surefire/JUnit platform.
- `mvn -pl Mage.Tests test`: run the dedicated gameplay/regression suite only.
- `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`: compile RL plugin and required dependencies.
- `make package`: create `mage-server.zip` and `mage-client.zip` and copy them to `deploy/` (override with `TARGET_DIR=...`).
- CI-style local run: `mvn test -B -Dxmage.dataCollectors.printGameLogs=false`.

## UMD Zaratan / Slurm Notes
- Use login nodes for code sync, environment prep, and job submission only; run training via Slurm jobs.
- For GPU jobs, set a GPU partition in the job script (for example `#SBATCH --partition=gpu`, `gpu-a100`, or `gpu-h100`).
- Request GPUs explicitly (for example `#SBATCH --gres=gpu:1` or `#SBATCH --gres=gpu:a100:1`).
- Do not assume `--qos=...` is required; using an unauthorized QoS causes `Invalid qos specification`.
- Check effective account permissions before submitting:
  - `sacctmgr show assoc where user=$USER format=User,Account,Partition,QOS,GrpTRESMins,GrpTRES,MaxTRES,MaxWall -P`
  - `scontrol show partition`
  - `sinfo -o "%P %a %l %D %c %m %G"`

## Zaratan Session Learnings (2026-03)
- Keep one canonical remote checkout path per session. In this environment the active path was `/home/jmaior/scratch.msml603/jmaior/mage` (not `/scratch.msml603/jmaior/mage`).
- Keep artifacts (runtime bundles, logs, checkpoints, DB) in scratch-backed project paths, and submit/watch jobs from the same checkout that owns those artifact paths.
- `scp` exit code interpretation:
  - `255`: authentication failure (before transfer), often keyboard-interactive / Duo flow mismatch.
  - `1`: transfer path failure (for example remote destination directory does not exist).
- For Zaratan SSH/SCP from Windows OpenSSH, `keyboard-interactive` + Duo works; disabling `GSSAPIAuthentication` may reduce auth confusion.
- PowerShell multiline paste frequently breaks long commands (split tokens like `TRAIN_PROFILES` or local file names). Prefer single-line commands or script files for submit/upload.
- In PowerShell, `$HOME` expands locally before reaching remote shell unless escaped/quoted carefully; avoid relying on inline `$HOME` in remote command strings when debugging paths.
- For runtime-bundle upload via `scripts/hpc/build_rl_runtime_bundle.ps1`:
  - Set `-ScpCreateRemoteDir:$true` when targeting new remote directories.
  - Ensure `-ScpDestination` exactly matches the active remote checkout path.
- Smoke-test baseline used here:
  - Submit with `HPC_NATIVE_ORCH=1`, `TOTAL_EPISODES=40`, `TRAIN_PROFILES=1`, short walltime.
  - Monitor with `scripts/hpc/watch_job.sh <job_id>`.
- Use `scripts/hpc/zaratan_spy_smoke.sh` to avoid copy/paste breakage. It auto-selects newest bundle from `BUNDLE_DIR` (default: `<repo>/local-training/hpc/bundles`) and submits then watches the job.

## Coding Style & Naming Conventions
- Java source/target compatibility is `1.8`; keep files UTF-8 encoded.
- Match existing style: 4-space indentation, same-line braces, concise class-level Javadocs.
- Naming: `PascalCase` classes, `camelCase` methods/fields, lowercase package names (for example, `mage.abilities.*`).
- Keep changes module-local when possible (UI in `Mage.Client`, rules/engine in `Mage`/`Mage.Sets`, server logic in `Mage.Server`).

## Testing Guidelines
- The project runs JUnit 5 with JUnit 4 vintage compatibility; AssertJ is also available.
- Add new regression tests under `Mage.Tests/src/test/java/org/mage/test/...` and name files `*Test.java`.
- Prefer deterministic tests (set seeds for random behavior) and keep logs minimal.
- JaCoCo is disabled by default; enable coverage with `-Djacoco.skip=false` when needed.

## Commit & Pull Request Guidelines
- Follow the repository's commit style: short, imperative subjects, optionally scoped (example: `RL: Fix mulligan logging race condition`).
- Keep each commit focused on one behavioral change.
- PRs should include: what changed, why, affected modules, and exact test commands executed.
- Link related issues and include screenshots for `Mage.Client` UI updates.
- Do not commit generated/local artifacts (DB files, logs, model outputs, virtualenvs).
