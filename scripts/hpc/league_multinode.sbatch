#!/bin/bash
#SBATCH --job-name=league-multinode
#SBATCH --account=msml603-class
#SBATCH --time=12:00:00
#SBATCH --output=local-training/hpc/bundles/league-multinode_%j.out
#SBATCH --error=local-training/hpc/bundles/league-multinode_%j.err
#
# Het-group 0: GPU node (inference + training)
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#
#SBATCH hetjob
#
# Het-group 1: CPU node (game simulation)
#SBATCH --partition=standard
#SBATCH --cpus-per-task=128
#SBATCH --mem=0

set -euo pipefail
echo "League multinode job starting on $(hostname) at $(date -u)"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# This script runs on het-group 0 (GPU node).
# run_spy_pbt.sh auto-detects multi-node via SLURM_JOB_NODELIST.
exec bash scripts/hpc/run_spy_pbt.sh
