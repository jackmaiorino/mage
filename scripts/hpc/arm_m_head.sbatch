#!/bin/bash
#SBATCH --job-name=spy-armM
#SBATCH --account=msml603-class
#SBATCH --time=08:00:00
#SBATCH --output=local-training/hpc/bundles/armM_%j.out
#SBATCH --error=local-training/hpc/bundles/armM_%j.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
# Arm-M (training_sweep_plan_jun3): FRESH size-M (256/4) Spy Combo, self-play, on a
# 2-GPU SPLIT-DEVICE head (cuda:0 inference / cuda:1 training) so a CPU-satellite
# fleet can scale past the single-GPU inference ceiling found in the smoke. A100
# (3.2x cheaper than H100, partition wide open). Size + entropy come from the
# Pauper-Spy-Combo-M profile train_env (selected by TRAIN_PROFILES=1, priority 0).
set -euo pipefail

TARBALL="$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)"
if [[ -z "$TARBALL" || ! -f "$TARBALL" ]]; then echo "FATAL: no tarball" >&2; exit 1; fi
export MAGE_RL_RUNTIME_TARBALL="$TARBALL"

export HPC_NATIVE_ORCH=1
export MULTI_PROFILE_JVM=1
export PY_SERVICE_MODE=shared_gpu
export GPU_SERVICE_BIND_HOST="0.0.0.0"
export TOTAL_EPISODES=10000000
export TRAIN_PROFILES=1            # -> Pauper-Spy-Combo-M (priority 0, name tiebreak over -Value)
export GAME_LOG_FREQUENCY=0
export OPPONENT_SAMPLER=self
export WINRATE_WINDOW=200

# 2-GPU split-device: a dedicated inference GPU lifts the satellite ceiling
export GPU_SERVICE_SPLIT_ROLES=1

# head runs few local runners; satellites generate the bulk of games
export WRITE_SATELLITE_ENV=1
export PRIMARY_RUNNER_CAP=8
export RUNNER_OVERSUBSCRIPTION_FACTOR=1

# fixes carried from the single-node + satellite smokes
export SCORE_WORKER_THREADS=0     # auto-scale inference workers
export GAME_TIMEOUT_SEC=900       # survive the ~7s cold-start warmup
export PY_BATCH_MAX_SIZE=1024
export PY_BATCH_TIMEOUT_MS=25

# conservative auto-satellites: size-M inference is ~8x heavier than the size-S
# smoke, so start at 2 and scale live (add_satellites.sh) after watching duty/pending
export AUTO_SATELLITES=2
export SAT_CPUS=32
export SAT_MEM=32G
export SAT_TIME=08:00:00
export SAT_JVMS_PER_NODE=2
export SAT_RUNNERS_PER_JVM=32

# clean raw PPO (size + entropy schedule come from the profile train_env)
export SEARCH_OP_ENABLE=0
export MCTS_TRAINING_ENABLE=0

echo "arm-M head starting on $(hostname) job=$SLURM_JOB_ID at $(date -u)"
echo "TARBALL=$TARBALL"
exec bash scripts/hpc/run_spy_pbt.sh
