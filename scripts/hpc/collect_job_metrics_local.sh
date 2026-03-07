#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: bash scripts/hpc/collect_job_metrics_local.sh --status-path <path> [--orchestrator-log <path>] [--trainer-ports \"19100 19101\"] [--gpu-host-ports \"27100\"]" >&2
}

TRAINER_PORT_START="${TRAINER_PORT_START:-19100}"
TRAINER_PORT_END="${TRAINER_PORT_END:-19131}"
GPU_HOST_METRICS_PORT_START="${GPU_HOST_METRICS_PORT_START:-27100}"
GPU_HOST_METRICS_PORT_END="${GPU_HOST_METRICS_PORT_END:-27115}"

status_path=""
orchestrator_log=""
trainer_ports_override=""
gpu_host_ports_override=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --status-path)
      status_path="${2:-}"
      shift 2
      ;;
    --orchestrator-log)
      orchestrator_log="${2:-}"
      shift 2
      ;;
    --trainer-ports)
      trainer_ports_override="${2:-}"
      shift 2
      ;;
    --gpu-host-ports)
      gpu_host_ports_override="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$status_path" && -z "$trainer_ports_override" && -z "$gpu_host_ports_override" ]]; then
  usage
  exit 2
fi

discover_ports() {
  python3 - "$status_path" "$orchestrator_log" <<'PY'
import json
import pathlib
import re
import sys

status_path = pathlib.Path(sys.argv[1]) if sys.argv[1] else None
orchestrator_log = pathlib.Path(sys.argv[2]) if sys.argv[2] else None
trainer_ports = []
gpu_host_ports = []

if status_path is not None and status_path.exists():
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8-sig"))
    except Exception:
        payload = {}
    for row in payload.get("trainers", []):
        if not isinstance(row, dict):
            continue
        try:
            port = int(row.get("metrics_port", 0))
        except Exception:
            port = 0
        if port > 0:
            trainer_ports.append(port)
    for row in payload.get("shared_gpu_hosts", []):
        if not isinstance(row, dict):
            continue
        try:
            port = int(row.get("metrics_port", 0))
        except Exception:
            port = 0
        if port > 0:
            gpu_host_ports.append(port)

if orchestrator_log is not None and orchestrator_log.exists():
    try:
        text = orchestrator_log.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = ""
    if text:
        trainer_ports.extend(int(m.group(1)) for m in re.finditer(r"metricsPort=(\d+)", text))
        gpu_host_ports.extend(int(m.group(1)) for m in re.finditer(r"gpuServiceMetricsPort=(\d+)", text))

print(" ".join(str(p) for p in sorted(set(p for p in trainer_ports if p > 0))))
print(" ".join(str(p) for p in sorted(set(p for p in gpu_host_ports if p > 0))))
PY
}

trainer_ports="$trainer_ports_override"
gpu_host_ports="$gpu_host_ports_override"
if [[ -z "$trainer_ports" || -z "$gpu_host_ports" ]]; then
  mapfile -t discovered_ports < <(discover_ports)
  if [[ -z "$trainer_ports" ]]; then
    trainer_ports="${discovered_ports[0]:-}"
  fi
  if [[ -z "$gpu_host_ports" ]]; then
    gpu_host_ports="${discovered_ports[1]:-}"
  fi
fi

if [[ -z "$trainer_ports" ]]; then
  trainer_ports="$(seq "${TRAINER_PORT_START}" "${TRAINER_PORT_END}" | tr '\n' ' ' | xargs)"
fi
if [[ -z "$gpu_host_ports" ]]; then
  gpu_host_ports="$(seq "${GPU_HOST_METRICS_PORT_START}" "${GPU_HOST_METRICS_PORT_END}" | tr '\n' ' ' | xargs)"
fi

echo "hostname=$(hostname -f)"
echo "collected_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "== Trainer metrics =="
echo "ports=${trainer_ports}"
for p in ${trainer_ports}; do
  curl -fsS "http://127.0.0.1:${p}/metrics" 2>/dev/null || true
done | awk '
  /^mage_config_py_batch_timeout_ms / {timeout=$2}
  /^mage_episodes_completed_total / {episodes_total+=$2; trainer_seen=1}
  /^mage_training_updates_total / {updates_total+=$2; trainer_seen=1}
  /^mage_active_episodes / {active_total+=$2; trainer_seen=1}
  /^mage_infer_batch_avg_size / {infer_avg+=$2; infer_n++; trainer_seen=1}
  /^mage_infer_batch_size_p50 / {infer_p50+=$2; infer_p50_n++; trainer_seen=1}
  /^mage_infer_flushes_timeout_total / {timeout_flush+=$2; trainer_seen=1}
  /^mage_infer_flushes_full_total / {full_flush+=$2; trainer_seen=1}
  /^mage_infer_latency_avg_ms / {infer_lat+=$2; infer_lat_n++; trainer_seen=1}
  /^mage_infer_latency_recent_avg_ms / {infer_lat_recent+=$2; infer_lat_recent_n++; trainer_seen=1}
  /^mage_infer_latency_p50_ms / {infer_lat_p50+=$2; infer_lat_p50_n++; trainer_seen=1}
  /^mage_infer_latency_p95_ms / {infer_lat_p95+=$2; infer_lat_p95_n++; trainer_seen=1}
  /^mage_train_queue_depth / {q+=$2; trainer_seen=1}
  /^mage_train_queue_dropped_total / {qd+=$2; trainer_seen=1}
  /^mage_train_batch_avg_episodes / {train_ep+=$2; train_ep_n++; trainer_seen=1}
  /^mage_train_latency_avg_ms / {train_lat+=$2; train_lat_n++; trainer_seen=1}
  /^mage_train_latency_recent_avg_ms / {train_lat_recent+=$2; train_lat_recent_n++; trainer_seen=1}
  /^mage_train_latency_p50_ms / {train_lat_p50+=$2; train_lat_p50_n++; trainer_seen=1}
  /^mage_train_latency_p95_ms / {train_lat_p95+=$2; train_lat_p95_n++; trainer_seen=1}
  /^mage_infer_time_ms / {infer_time+=$2; infer_time_n++; trainer_seen=1}
  /^mage_train_time_ms / {train_time+=$2; train_time_n++; trainer_seen=1}
  END {
    if (!trainer_seen) {
      print "warning=no_trainer_metrics_detected";
      exit 0;
    }
    printf "batch_timeout_ms=%s\n", timeout;
    printf "episodes_completed_total=%.0f\n", episodes_total;
    printf "training_updates_total=%.0f\n", updates_total;
    printf "active_episodes_total=%.0f\n", active_total;
    printf "infer_batch_avg_size_mean=%.2f\n", infer_n ? infer_avg / infer_n : 0;
    printf "infer_batch_p50_mean=%.2f\n", infer_p50_n ? infer_p50 / infer_p50_n : 0;
    printf "infer_flushes_timeout_total=%.0f\n", timeout_flush;
    printf "infer_flushes_full_total=%.0f\n", full_flush;
    printf "infer_latency_avg_ms_mean=%.2f\n", infer_lat_n ? infer_lat / infer_lat_n : 0;
    printf "infer_latency_recent_avg_ms_mean=%.2f\n", infer_lat_recent_n ? infer_lat_recent / infer_lat_recent_n : 0;
    printf "infer_latency_p50_ms_mean=%.2f\n", infer_lat_p50_n ? infer_lat_p50 / infer_lat_p50_n : 0;
    printf "infer_latency_p95_ms_mean=%.2f\n", infer_lat_p95_n ? infer_lat_p95 / infer_lat_p95_n : 0;
    printf "train_queue_depth_total=%.0f\n", q;
    printf "train_queue_dropped_total=%.0f\n", qd;
    printf "train_batch_avg_episodes_mean=%.2f\n", train_ep_n ? train_ep / train_ep_n : 0;
    printf "train_latency_avg_ms_mean=%.2f\n", train_lat_n ? train_lat / train_lat_n : 0;
    printf "train_latency_recent_avg_ms_mean=%.2f\n", train_lat_recent_n ? train_lat_recent / train_lat_recent_n : 0;
    printf "train_latency_p50_ms_mean=%.2f\n", train_lat_p50_n ? train_lat_p50 / train_lat_p50_n : 0;
    printf "train_latency_p95_ms_mean=%.2f\n", train_lat_p95_n ? train_lat_p95 / train_lat_p95_n : 0;
    printf "infer_time_ms_mean=%.2f\n", infer_time_n ? infer_time / infer_time_n : 0;
    printf "train_time_ms_mean=%.2f\n", train_time_n ? train_time / train_time_n : 0;
  }
'

echo
echo "== Shared GPU host metrics =="
echo "ports=${gpu_host_ports}"
for p in ${gpu_host_ports}; do
  curl -fsS "http://127.0.0.1:${p}/metrics" 2>/dev/null || true
done | awk '
  /^gpu_service_batch_timeout_ms / {batch_timeout+=$2; batch_timeout_n++; host_seen=1}
  /^gpu_service_batch_max_size / {batch_max+=$2; batch_max_n++; host_seen=1}
  /^gpu_service_profiles / {profiles+=$2; host_seen=1}
  /^gpu_service_pending_scores / {pending_scores+=$2; host_seen=1}
  /^gpu_service_pending_scores_oldest_ms / {if ($2 > pending_scores_oldest) pending_scores_oldest=$2; host_seen=1}
  /^gpu_service_pending_trains / {pending_trains+=$2; host_seen=1}
  /^gpu_service_pending_trains_oldest_ms / {if ($2 > pending_trains_oldest) pending_trains_oldest=$2; host_seen=1}
  /^gpu_service_score_batches_total / {score_batches+=$2; host_seen=1}
  /^gpu_service_train_batches_total / {train_batches+=$2; host_seen=1}
  /^gpu_service_score_failures_total / {score_failures+=$2; host_seen=1}
  /^gpu_service_train_failures_total / {train_failures+=$2; host_seen=1}
  /^gpu_service_model_publishes_total / {model_publishes+=$2; host_seen=1}
  /^gpu_service_model_reloads_total / {model_reloads+=$2; host_seen=1}
  /^gpu_service_last_error_info\{/ {
    host_seen=1
    if (match($0, /message="([^"]*)"/, m)) {
      last_error=m[1]
    }
  }
  /^gpu_service_infer_flush_timeout_total / {infer_flush_timeout+=$2; host_seen=1}
  /^gpu_service_infer_flush_full_total / {infer_flush_full+=$2; host_seen=1}
  /^gpu_service_infer_batch_avg_size / {infer_batch_avg+=$2; infer_batch_avg_n++; host_seen=1}
  /^gpu_service_infer_batch_p50_size / {infer_batch_p50+=$2; infer_batch_p50_n++; host_seen=1}
  /^gpu_service_infer_batch_p95_size / {infer_batch_p95+=$2; infer_batch_p95_n++; host_seen=1}
  /^gpu_service_infer_latency_recent_avg_ms / {infer_lat_recent+=$2; infer_lat_recent_n++; host_seen=1}
  /^gpu_service_infer_latency_p50_ms / {infer_lat_p50+=$2; infer_lat_p50_n++; host_seen=1}
  /^gpu_service_infer_latency_p95_ms / {infer_lat_p95+=$2; infer_lat_p95_n++; host_seen=1}
  /^gpu_service_infer_service_recent_avg_ms / {infer_service+=$2; infer_service_n++; host_seen=1}
  /^gpu_service_infer_service_p50_ms / {infer_service_p50+=$2; infer_service_p50_n++; host_seen=1}
  /^gpu_service_infer_service_p95_ms / {infer_service_p95+=$2; infer_service_p95_n++; host_seen=1}
  /^gpu_service_train_batch_avg_episodes / {train_batch_ep+=$2; train_batch_ep_n++; host_seen=1}
  /^gpu_service_train_batch_avg_steps / {train_batch_steps+=$2; train_batch_steps_n++; host_seen=1}
  /^gpu_service_train_latency_recent_avg_ms / {train_lat_recent+=$2; train_lat_recent_n++; host_seen=1}
  /^gpu_service_train_latency_p50_ms / {train_lat_p50+=$2; train_lat_p50_n++; host_seen=1}
  /^gpu_service_train_latency_p95_ms / {train_lat_p95+=$2; train_lat_p95_n++; host_seen=1}
  /^gpu_service_train_service_recent_avg_ms / {train_service+=$2; train_service_n++; host_seen=1}
  /^gpu_service_train_service_p50_ms / {train_service_p50+=$2; train_service_p50_n++; host_seen=1}
  /^gpu_service_train_service_p95_ms / {train_service_p95+=$2; train_service_p95_n++; host_seen=1}
  END {
    if (!host_seen) {
      print "warning=no_shared_gpu_host_metrics_detected";
      print "hint=verify PY_SERVICE_MODE=shared_gpu and curl a single host metrics port directly (for example 27100)";
      exit 0;
    }
    printf "batch_timeout_ms_mean=%.2f\n", batch_timeout_n ? batch_timeout / batch_timeout_n : 0;
    printf "batch_max_size_mean=%.2f\n", batch_max_n ? batch_max / batch_max_n : 0;
    printf "registered_profiles_total=%.0f\n", profiles;
    printf "pending_scores_total=%.0f\n", pending_scores;
    printf "pending_scores_oldest_ms_max=%.2f\n", pending_scores_oldest;
    printf "pending_trains_total=%.0f\n", pending_trains;
    printf "pending_trains_oldest_ms_max=%.2f\n", pending_trains_oldest;
    printf "score_batches_total=%.0f\n", score_batches;
    printf "train_batches_total=%.0f\n", train_batches;
    printf "score_failures_total=%.0f\n", score_failures;
    printf "train_failures_total=%.0f\n", train_failures;
    printf "model_publishes_total=%.0f\n", model_publishes;
    printf "model_reloads_total=%.0f\n", model_reloads;
    printf "infer_flush_timeout_total=%.0f\n", infer_flush_timeout;
    printf "infer_flush_full_total=%.0f\n", infer_flush_full;
    printf "infer_batch_avg_size_mean=%.2f\n", infer_batch_avg_n ? infer_batch_avg / infer_batch_avg_n : 0;
    printf "infer_batch_p50_size_mean=%.2f\n", infer_batch_p50_n ? infer_batch_p50 / infer_batch_p50_n : 0;
    printf "infer_batch_p95_size_mean=%.2f\n", infer_batch_p95_n ? infer_batch_p95 / infer_batch_p95_n : 0;
    printf "infer_latency_recent_avg_ms_mean=%.2f\n", infer_lat_recent_n ? infer_lat_recent / infer_lat_recent_n : 0;
    printf "infer_latency_p50_ms_mean=%.2f\n", infer_lat_p50_n ? infer_lat_p50 / infer_lat_p50_n : 0;
    printf "infer_latency_p95_ms_mean=%.2f\n", infer_lat_p95_n ? infer_lat_p95 / infer_lat_p95_n : 0;
    printf "infer_service_recent_avg_ms_mean=%.2f\n", infer_service_n ? infer_service / infer_service_n : 0;
    printf "infer_service_p50_ms_mean=%.2f\n", infer_service_p50_n ? infer_service_p50 / infer_service_p50_n : 0;
    printf "infer_service_p95_ms_mean=%.2f\n", infer_service_p95_n ? infer_service_p95 / infer_service_p95_n : 0;
    printf "train_batch_avg_episodes_mean=%.2f\n", train_batch_ep_n ? train_batch_ep / train_batch_ep_n : 0;
    printf "train_batch_avg_steps_mean=%.2f\n", train_batch_steps_n ? train_batch_steps / train_batch_steps_n : 0;
    printf "train_latency_recent_avg_ms_mean=%.2f\n", train_lat_recent_n ? train_lat_recent / train_lat_recent_n : 0;
    printf "train_latency_p50_ms_mean=%.2f\n", train_lat_p50_n ? train_lat_p50 / train_lat_p50_n : 0;
    printf "train_latency_p95_ms_mean=%.2f\n", train_lat_p95_n ? train_lat_p95 / train_lat_p95_n : 0;
    printf "train_service_recent_avg_ms_mean=%.2f\n", train_service_n ? train_service / train_service_n : 0;
    printf "train_service_p50_ms_mean=%.2f\n", train_service_p50_n ? train_service_p50 / train_service_p50_n : 0;
    printf "train_service_p95_ms_mean=%.2f\n", train_service_p95_n ? train_service_p95 / train_service_p95_n : 0;
    printf "last_error=%s\n", last_error;
  }
'
