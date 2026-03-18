#!/bin/bash
# Wrapper that sets env vars and calls run_spy_pbt.sh
# Avoids --export=ALL which triggers signal 53 on Zaratan
export HPC_NATIVE_ORCH=1
export MAGE_RL_RUNTIME_TARBALL="$1"
export MULTI_PROFILE_JVM=1
export RUNNER_OVERSUBSCRIPTION_FACTOR=5
export TRAIN_PROFILES=3
export TOTAL_EPISODES=10000000
export PY_SERVICE_MODE=shared_gpu
export GAME_LOG_FREQUENCY=0
export WINRATE_WINDOW=200
export OPPONENT_SAMPLER=self
export GPU_SERVICE_SPLIT_ROLES=1
exec bash "$(dirname "$0")/run_spy_pbt.sh"
