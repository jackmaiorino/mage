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

if [[ -f "$pbt_state" ]]; then
  echo "== Throughput (Games/sec) =="
  python3 - "$repo_root" "$pbt_state" <<'PY'
import csv
import json
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1])
state_path = pathlib.Path(sys.argv[2])
stats_root = repo_root / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles"
recent_window = 200

try:
    state = json.load(open(state_path, "r", encoding="utf-8"))
except Exception as exc:
    print(f"failed to parse pbt state: {exc}")
    raise SystemExit(0)

profiles = []
for item in state.get("profiles", []):
    name = str(item.get("profile", "")).strip()
    if name:
        profiles.append(name)

if not profiles:
    print("no active profiles in pbt_state yet")
    raise SystemExit(0)

agg_recent_eps = 0
agg_recent_sec = 0.0
agg_total_eps = 0
agg_total_sec = 0.0

for profile in profiles:
    stats_csv = stats_root / profile / "logs/stats/training_stats.csv"
    if not stats_csv.exists():
        print(f"{profile}: no stats yet")
        continue

    episode_seconds = []
    try:
        with stats_csv.open("r", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw = row.get("episode_seconds", "")
                try:
                    sec = float(raw)
                except Exception:
                    continue
                if sec > 0:
                    episode_seconds.append(sec)
    except Exception as exc:
        print(f"{profile}: failed to read stats ({exc})")
        continue

    if not episode_seconds:
        print(f"{profile}: no episode_seconds data yet")
        continue

    total_eps = len(episode_seconds)
    total_sec = sum(episode_seconds)
    recent = episode_seconds[-recent_window:]
    recent_eps = len(recent)
    recent_sec = sum(recent)

    total_gps = (total_eps / total_sec) if total_sec > 0 else 0.0
    recent_gps = (recent_eps / recent_sec) if recent_sec > 0 else 0.0

    agg_recent_eps += recent_eps
    agg_recent_sec += recent_sec
    agg_total_eps += total_eps
    agg_total_sec += total_sec

    print(
        f"{profile}: recent({recent_eps})={recent_gps:.3f} g/s  "
        f"lifetime={total_gps:.3f} g/s  episodes={total_eps}"
    )

if agg_recent_sec > 0:
    print(f"aggregate_recent({agg_recent_eps})={(agg_recent_eps / agg_recent_sec):.3f} g/s")
else:
    print("aggregate_recent: no data")
if agg_total_sec > 0:
    print(f"aggregate_lifetime({agg_total_eps})={(agg_total_eps / agg_total_sec):.3f} g/s")
else:
    print("aggregate_lifetime: no data")
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
