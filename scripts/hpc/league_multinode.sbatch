#!/bin/bash
#SBATCH --job-name=league-multinode
#SBATCH --account=msml603-class
#SBATCH --time=12:00:00
#SBATCH --output=local-training/hpc/bundles/league-multinode_%j.out
#SBATCH --error=local-training/hpc/bundles/league-multinode_%j.err
#
# Het-group 0: GPU node (inference + training)
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#
#SBATCH hetjob
#
# Het-group 1: CPU node (game simulation)
#SBATCH --partition=standard
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#
# Multi-JVM scaling: the orchestrator launches multiple smaller JVMs per CPU
# node (TARGET_RUNNERS_PER_JVM, default 150) instead of one large JVM.
# Env vars: TARGET_RUNNERS_PER_JVM, AUX_JVMS_PER_NODE (0=auto), PRIMARY_RUNNER_CAP (200)
#
# For more CPU nodes, add additional hetjob components or submit separate
# coordinated jobs:
#   1. sbatch GPU-only job (this file without het-group)
#   2. Wait for GPU node hostname: squeue -j $GPU_JOB -h -o "%N"
#   3. sbatch CPU jobs with GPU_SERVICE_ENDPOINT=<gpu-node>:26100

set -euo pipefail
echo "League multinode job starting on $(hostname) at $(date -u)"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# This script runs on het-group 0 (GPU node).
# run_spy_pbt.sh auto-detects multi-node via SLURM_JOB_NODELIST.
exec bash scripts/hpc/run_spy_pbt.sh
