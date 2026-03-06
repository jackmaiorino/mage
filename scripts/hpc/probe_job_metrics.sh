#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-${JOB:-}}"
if [[ -z "${JOB_ID}" ]]; then
  echo "usage: bash scripts/hpc/probe_job_metrics.sh <job_id>" >&2
  exit 2
fi

TRAINER_PORT_START="${TRAINER_PORT_START:-19100}"
TRAINER_PORT_END="${TRAINER_PORT_END:-19131}"
GPU_HOST_METRICS_PORT_START="${GPU_HOST_METRICS_PORT_START:-27100}"
GPU_HOST_METRICS_PORT_END="${GPU_HOST_METRICS_PORT_END:-27115}"

srun --jobid="${JOB_ID}" --overlap -N1 -n1 -c1 bash -s -- \
  "${TRAINER_PORT_START}" \
  "${TRAINER_PORT_END}" \
  "${GPU_HOST_METRICS_PORT_START}" \
  "${GPU_HOST_METRICS_PORT_END}" <<'EOF'
set -euo pipefail

trainer_port_start="$1"
trainer_port_end="$2"
gpu_host_metrics_port_start="$3"
gpu_host_metrics_port_end="$4"

hostname

echo "== Trainer metrics =="
for p in $(seq "${trainer_port_start}" "${trainer_port_end}"); do
  curl -fsS "http://127.0.0.1:${p}/metrics" 2>/dev/null || true
done | awk '
  /^mage_config_py_batch_timeout_ms / {timeout=$2}
  /^mage_infer_batch_avg_size / {infer_avg+=$2; infer_n++}
  /^mage_infer_batch_size_p50 / {infer_p50+=$2; infer_p50_n++}
  /^mage_infer_flushes_timeout_total / {timeout_flush+=$2}
  /^mage_infer_flushes_full_total / {full_flush+=$2}
  /^mage_infer_latency_avg_ms / {infer_lat+=$2; infer_lat_n++}
  /^mage_infer_latency_recent_avg_ms / {infer_lat_recent+=$2; infer_lat_recent_n++}
  /^mage_infer_latency_p50_ms / {infer_lat_p50+=$2; infer_lat_p50_n++}
  /^mage_infer_latency_p95_ms / {infer_lat_p95+=$2; infer_lat_p95_n++}
  /^mage_train_queue_depth / {q+=$2}
  /^mage_train_queue_dropped_total / {qd+=$2}
  /^mage_train_batch_avg_episodes / {train_ep+=$2; train_ep_n++}
  /^mage_train_latency_avg_ms / {train_lat+=$2; train_lat_n++}
  /^mage_train_latency_recent_avg_ms / {train_lat_recent+=$2; train_lat_recent_n++}
  /^mage_train_latency_p50_ms / {train_lat_p50+=$2; train_lat_p50_n++}
  /^mage_train_latency_p95_ms / {train_lat_p95+=$2; train_lat_p95_n++}
  /^mage_infer_time_ms / {infer_time+=$2; infer_time_n++}
  /^mage_train_time_ms / {train_time+=$2; train_time_n++}
  END {
    printf "batch_timeout_ms=%s\n", timeout;
    printf "infer_batch_avg_size_mean=%.2f\n", infer_n ? infer_avg / infer_n : 0;
    printf "infer_batch_p50_mean=%.2f\n", infer_p50_n ? infer_p50 / infer_p50_n : 0;
    printf "infer_flushes_timeout_total=%d\n", timeout_flush;
    printf "infer_flushes_full_total=%d\n", full_flush;
    printf "infer_latency_avg_ms_mean=%.2f\n", infer_lat_n ? infer_lat / infer_lat_n : 0;
    printf "infer_latency_recent_avg_ms_mean=%.2f\n", infer_lat_recent_n ? infer_lat_recent / infer_lat_recent_n : 0;
    printf "infer_latency_p50_ms_mean=%.2f\n", infer_lat_p50_n ? infer_lat_p50 / infer_lat_p50_n : 0;
    printf "infer_latency_p95_ms_mean=%.2f\n", infer_lat_p95_n ? infer_lat_p95 / infer_lat_p95_n : 0;
    printf "train_queue_depth_total=%d\n", q;
    printf "train_queue_dropped_total=%d\n", qd;
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
for p in $(seq "${gpu_host_metrics_port_start}" "${gpu_host_metrics_port_end}"); do
  curl -fsS "http://127.0.0.1:${p}/metrics" 2>/dev/null || true
done | awk '
  /^gpu_service_batch_timeout_ms / {batch_timeout+=$2; batch_timeout_n++}
  /^gpu_service_batch_max_size / {batch_max+=$2; batch_max_n++}
  /^gpu_service_profiles / {profiles+=$2}
  /^gpu_service_pending_scores / {pending_scores+=$2}
  /^gpu_service_pending_scores_oldest_ms / {if ($2 > pending_scores_oldest) pending_scores_oldest=$2}
  /^gpu_service_pending_trains / {pending_trains+=$2}
  /^gpu_service_pending_trains_oldest_ms / {if ($2 > pending_trains_oldest) pending_trains_oldest=$2}
  /^gpu_service_score_batches_total / {score_batches+=$2}
  /^gpu_service_train_batches_total / {train_batches+=$2}
  /^gpu_service_infer_flush_timeout_total / {infer_flush_timeout+=$2}
  /^gpu_service_infer_flush_full_total / {infer_flush_full+=$2}
  /^gpu_service_infer_batch_avg_size / {infer_batch_avg+=$2; infer_batch_avg_n++}
  /^gpu_service_infer_batch_p50_size / {infer_batch_p50+=$2; infer_batch_p50_n++}
  /^gpu_service_infer_batch_p95_size / {infer_batch_p95+=$2; infer_batch_p95_n++}
  /^gpu_service_infer_latency_recent_avg_ms / {infer_lat_recent+=$2; infer_lat_recent_n++}
  /^gpu_service_infer_latency_p50_ms / {infer_lat_p50+=$2; infer_lat_p50_n++}
  /^gpu_service_infer_latency_p95_ms / {infer_lat_p95+=$2; infer_lat_p95_n++}
  /^gpu_service_infer_service_recent_avg_ms / {infer_service+=$2; infer_service_n++}
  /^gpu_service_infer_service_p50_ms / {infer_service_p50+=$2; infer_service_p50_n++}
  /^gpu_service_infer_service_p95_ms / {infer_service_p95+=$2; infer_service_p95_n++}
  /^gpu_service_train_batch_avg_episodes / {train_batch_ep+=$2; train_batch_ep_n++}
  /^gpu_service_train_batch_avg_steps / {train_batch_steps+=$2; train_batch_steps_n++}
  /^gpu_service_train_latency_recent_avg_ms / {train_lat_recent+=$2; train_lat_recent_n++}
  /^gpu_service_train_latency_p50_ms / {train_lat_p50+=$2; train_lat_p50_n++}
  /^gpu_service_train_latency_p95_ms / {train_lat_p95+=$2; train_lat_p95_n++}
  /^gpu_service_train_service_recent_avg_ms / {train_service+=$2; train_service_n++}
  /^gpu_service_train_service_p50_ms / {train_service_p50+=$2; train_service_p50_n++}
  /^gpu_service_train_service_p95_ms / {train_service_p95+=$2; train_service_p95_n++}
  END {
    if (batch_timeout_n == 0 && batch_max_n == 0 && profiles == 0 && score_batches == 0 && train_batches == 0) {
      print "warning=no_shared_gpu_host_metrics_detected";
      print "hint=verify PY_SERVICE_MODE=shared_gpu and curl a single host metrics port directly (for example 27100)";
      exit 0;
    }
    printf "batch_timeout_ms_mean=%.2f\n", batch_timeout_n ? batch_timeout / batch_timeout_n : 0;
    printf "batch_max_size_mean=%.2f\n", batch_max_n ? batch_max / batch_max_n : 0;
    printf "registered_profiles_total=%d\n", profiles;
    printf "pending_scores_total=%d\n", pending_scores;
    printf "pending_scores_oldest_ms_max=%.2f\n", pending_scores_oldest;
    printf "pending_trains_total=%d\n", pending_trains;
    printf "pending_trains_oldest_ms_max=%.2f\n", pending_trains_oldest;
    printf "score_batches_total=%d\n", score_batches;
    printf "train_batches_total=%d\n", train_batches;
    printf "infer_flush_timeout_total=%d\n", infer_flush_timeout;
    printf "infer_flush_full_total=%d\n", infer_flush_full;
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
  }
'
EOF
