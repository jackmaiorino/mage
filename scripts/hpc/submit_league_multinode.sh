#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
repo_root="$(pwd)"

BUNDLE="${1:-$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)}"
if [[ -z "$BUNDLE" ]]; then
  echo "ERROR: No bundle found. Pass path as first argument or place in local-training/hpc/bundles/" >&2
  exit 1
fi
echo "Bundle: $BUNDLE"

export HPC_NATIVE_ORCH=1
export MAGE_RL_RUNTIME_TARBALL="$BUNDLE"
export LEAGUE_MODE=rl_only
export TRAIN_PROFILES=10
export TOTAL_EPISODES=10000000
export PY_SERVICE_MODE=shared_gpu
export GAME_LOG_FREQUENCY=500
export RUNNER_OVERSUBSCRIPTION_FACTOR=20
export GAME_CPU_CORES=128
export TRAINER_JVM_XMX_MB=40000
export MULTI_PROFILE_JVM=1

sbatch --export=ALL scripts/hpc/league_multinode.sbatch

echo "Submitted. Check with: squeue -u \$USER"
