#!/bin/bash
#SBATCH --job-name=rally-skill7-eval
#SBATCH --account=msml603-class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=local-training/hpc/bundles/rally_skill7_%A_%a.out
#SBATCH --error=local-training/hpc/bundles/rally_skill7_%A_%a.err
#
# Deterministic skill-7 Rally gauntlet eval, one Slurm array task per opponent
# (8 tasks == the 8 decks in decklist.all_opponents.txt: Grixis Affinity,
# Mono-Red Burn, Mono Red Rally [mirror], Mono-Blue Faeries, Mono-Blue Terror,
# Jund Wildfire, Caw-Gates, Elves). Each task runs
# scripts/run_cp7_eval_sweep.py serially against ONE opponent at
# --skill 7 --games-per-matchup 256 --deterministic-eval --replay-seed-base 5151,
# profile Pauper-Rally-Anchor-Value. The registry each task evaluates against is
# generated at runtime the same way scripts/run_rally_baseline_det.ps1 does it:
# take a base registry, override Pauper-Rally-Anchor-Value's deck_path to
# decklist.all_opponents.txt, write a fresh copy (see rally_skill7_chunk.sh).
#
# DETERMINISM CONSTRAINT: --deterministic-eval only reproduces bit-identical
# games when each (JVM, gpu_service_host.py) pair is its own OS process with
# its own ports/DB/registry -- confirmed empirically that INTRA-process
# parallelism (multiple --parallel workers or multiple opponents sharing one
# GPU-service instance) breaks it via shared inference-request-ordering, while
# SEPARATE PROCESSES do not interfere with each other. This array gives us that
# for free: 8 independent single-JVM/single-GPU-service tasks running in
# parallel across nodes, each fully serial (and therefore deterministic)
# internally. See the header of rally_skill7_chunk.sh for the full rationale
# and the invariants that must be preserved (no shared ports/registry/run-id
# across tasks; do not add --parallel).
#
# BEFORE SUBMITTING, upload/hot-patch the remote checkout:
#   1. Rebuild + upload the runtime bundle (Java changed today and this eval
#      uses the bundle's classpath, not `mvn compile` on the compute node):
#        .\scripts\hpc\build_rl_runtime_bundle.ps1
#      Files that changed and need to be in that bundle: ComputerPlayerRL.java,
#      GameHealthMonitor.java, MulliganLogger.java, PythonEntryPoint.java,
#      PythonMLBatchManager.java, RLTrainer.java, StateSequenceBuilder.java,
#      TerminalPrefixSearch.java.
#   2. Hot-patch the Python files that changed today (these load straight from
#      the checkout at runtime -- no bundle rebuild needed, just upload):
#      scripts/run_cp7_eval_sweep.py, .../MLPythonCode/py4j_entry_point.py,
#      .../MLPythonCode/metrics_collector.py, .../MLPythonCode/model_persistence.py,
#      .../MLPythonCode/mtg_transformer.py, .../MLPythonCode/snapshot_manager.py.
#      Use scripts/hpc/upload_via_ssh_stdin.py per the CLAUDE.md hot-patch recipe.
#   3. Upload the untracked opponent-pool file (new locally, not yet committed):
#      Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.all_opponents.txt
#   4. Upload the Rally checkpoint you intend to gate to
#      Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/Pauper-Rally-Anchor-Value/models/model.pt
#      (this script does NOT copy one in for you; rally_skill7_chunk.sh fails
#      fast if it's missing). The currently-running local eval
#      (local-training/rally_baseline_det.log, do not touch) is comparing
#      E:\mage-training\backups\rally_dominant_c1_20260623\model.pt vs
#      E:\mage-training\backups\rally_latest_armF_20260630\model.pt -- pick
#      whichever arm you want gated at scale and upload that one.
#   5. Confirm Mage.Sets card DB exists remotely (db/cards.h2.mv.db or
#      db_eval/cards.h2.mv.db in the checkout) -- run_cp7_eval_sweep.py copies
#      per-task slot DBs from whichever is present; it does not build it.
#
# Submit with: sbatch scripts/hpc/rally_skill7_array.sbatch
# (optionally --export=ALL,MAGE_RL_RUNTIME_TARBALL=<path> to pin a specific bundle)
set -euo pipefail

REPO_ROOT="${SOURCE_REPO_ROOT:-/home/jmaior/scratch.msml603/jmaior/mage}"
cd "$REPO_ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] rally_skill7_array task ${SLURM_ARRAY_TASK_ID:-0} of job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}} starting"

exec bash "$REPO_ROOT/scripts/hpc/rally_skill7_chunk.sh"
