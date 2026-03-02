#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
reports_root="$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/discovery"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="$reports_root/$run_id"
mkdir -p "$out_dir"

run_capture() {
  local name="$1"
  shift
  local out_file="$out_dir/${name}.txt"
  {
    echo "# utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# command: $*"
    echo
    "$@"
  } >"$out_file" 2>&1 || true
}

run_capture "identity" bash -lc 'whoami; id; hostname -f; uname -a'
run_capture "which_slurm" bash -lc 'which sinfo || true; which squeue || true; which sbatch || true; which srun || true'
run_capture "sinfo" sinfo
run_capture "scontrol_show_partition" scontrol show partition
run_capture "sacctmgr_assoc" sacctmgr show assoc user="$USER" format=Account,Partition,GrpTRES,MaxJobs,MaxWall,QOS
run_capture "squeue_user" squeue -u "$USER"
run_capture "scratch_env" bash -lc 'echo "SLURM_TMPDIR=${SLURM_TMPDIR:-}"; echo "TMPDIR=${TMPDIR:-}"; echo "SCRATCH=${SCRATCH:-}"'

summary_file="$out_dir/summary.md"
cat >"$summary_file" <<EOF
# Zartan Slurm Discovery Snapshot

- Run ID: \`$run_id\`
- User: \`$USER\`
- Output dir: \`$out_dir\`

## Files
- \`identity.txt\`
- \`which_slurm.txt\`
- \`sinfo.txt\`
- \`scontrol_show_partition.txt\`
- \`sacctmgr_assoc.txt\`
- \`squeue_user.txt\`
- \`scratch_env.txt\`

## Tune-gate checklist
- Account
- Partition
- QoS
- Max walltime
- GPU per job
- CPU per job
- Memory per job
- Local scratch path
EOF

echo "Wrote Slurm discovery snapshot: $out_dir"

