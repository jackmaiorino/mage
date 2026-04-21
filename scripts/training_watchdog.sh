#!/usr/bin/env bash
# External watchdog: detects when overnight training stalls (eps/s=0 for >5 min
# OR GPU service process gone) and restarts the whole pipeline.
#
# Run as a background loop alongside training. Logs to local-training/local_pbt/watchdog.log
set -uo pipefail

cd "$(dirname "$0")/.."
LOG="local-training/local_pbt/watchdog.log"
mkdir -p local-training/local_pbt

while true; do
    # Find the most recent run log
    latest_log=$(ls -t local-training/local_pbt/run*_*.log 2>/dev/null | head -1)
    [ -n "$latest_log" ] || { echo "$(date) waiting for run log..." >> "$LOG"; sleep 30; continue; }

    # Count distinct eps/s readings in last 10 lines that are non-zero
    last_eps=$(tail -10 "$latest_log" 2>/dev/null | grep -oE "eps/s: [0-9.]+" | tail -10 | grep -cE "eps/s: [1-9]")

    # Check GPU service alive
    py_count=$(tasklist 2>/dev/null | grep -c "python3.12.exe" || true)

    if [ "$last_eps" -eq 0 ] && [ "$py_count" -lt 2 ]; then
        echo "$(date) STALL DETECTED: last_eps=$last_eps py_count=$py_count -> restarting" >> "$LOG"
        # Kill anything alive
        tasklist 2>/dev/null | grep -iE "(java|python3)" | awk '{print $2}' | while read pid; do
            taskkill //PID "$pid" //T //F 2>&1 | head -1 >> "$LOG"
        done
        sleep 5
        # Restart with same config
        run_log="local-training/local_pbt/run_watchdog_restart_$(date +%Y%m%d_%H%M%S).log"
        nohup env TRAIN_PROFILES=1 NUM_GAME_RUNNERS=48 PY_BATCH_TIMEOUT_MS=25 \
            py -3.12 scripts/run_local_pbt.py > "$run_log" 2>&1 &
        echo "$(date) RESTARTED: log=$run_log pid=$!" >> "$LOG"
        # Wait longer before next check to let it stabilize
        sleep 120
    else
        echo "$(date) HEALTHY: nonzero_eps=$last_eps py_count=$py_count" >> "$LOG"
    fi
    sleep 60
done
