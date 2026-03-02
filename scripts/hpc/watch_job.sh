#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
orchestrator_dir="$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/pauper/orchestrator"
jobs_root="$repo_root/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league_reports/hpc/jobs"
job_id="${1:-}"
follow_mode="${FOLLOW_MODE:-1}"

if [[ "${2:-}" == "--no-follow" ]]; then
  follow_mode=0
fi

if [[ -z "$job_id" ]]; then
  if [[ -d "$jobs_root" ]]; then
    job_id="$(ls -1 "$jobs_root" 2>/dev/null | sort | tail -n 1 || true)"
  fi
fi

job_dir=""
if [[ -n "$job_id" && -d "$jobs_root/$job_id" ]]; then
  job_dir="$jobs_root/$job_id"
fi

echo "Repo root: $repo_root"
echo "Job ID: ${job_id:-n/a}"
if [[ -n "$job_dir" ]]; then
  echo "Job reports: $job_dir"
fi
echo

if [[ -n "$job_id" ]] && command -v squeue >/dev/null 2>&1; then
  echo "== squeue =="
  squeue -j "$job_id" -o "%.18i %.10T %.9M %.12l %.8C %.8m %.20R" || true
  echo
fi

pbt_state="$orchestrator_dir/pbt_state.json"
if [[ -f "$pbt_state" ]]; then
  echo "== Latest PBT events =="
  python3 - "$pbt_state" <<'PY'
import json,sys
path=sys.argv[1]
try:
    data=json.load(open(path,"r",encoding="utf-8"))
except Exception as e:
    print(f"failed to parse {path}: {e}")
    raise SystemExit(0)
events=data.get("events",[])
for ev in events[-5:]:
    ts=ev.get("timestamp_utc","")
    grp=ev.get("population_group","")
    w=ev.get("winner","")
    l=ev.get("loser","")
    ww=ev.get("winner_wr","")
    lw=ev.get("loser_wr","")
    seed=ev.get("new_seed","")
    print(f"{ts} group={grp} winner={w} loser={l} winner_wr={ww} loser_wr={lw} seed={seed}")
if not events:
    print("no PBT events yet")
PY
  echo
fi

orchestrator_log=""
if [[ -n "$job_dir" && -f "$job_dir/orchestrator.log" ]]; then
  orchestrator_log="$job_dir/orchestrator.log"
fi
if [[ -n "$orchestrator_log" ]]; then
  echo "== Latest rolling status line =="
  grep -E "Rolling winrates|Concurrent training profiles running" "$orchestrator_log" | tail -n 2 || true
  echo
fi

if [[ "$follow_mode" -eq 0 ]]; then
  exit 0
fi

files_to_tail=()
if [[ -n "$orchestrator_log" ]]; then
  files_to_tail+=("$orchestrator_log")
fi
if [[ -d "$orchestrator_dir/trainers" ]]; then
  while IFS= read -r f; do
    files_to_tail+=("$f")
  done < <(find "$orchestrator_dir/trainers" -maxdepth 1 -type f \( -name "*.stdout.log" -o -name "*.stderr.log" \) | sort)
fi

if [[ "${#files_to_tail[@]}" -eq 0 ]]; then
  echo "No log files found to tail."
  exit 0
fi

echo "== tail -F logs =="
tail -n 80 -F "${files_to_tail[@]}"

