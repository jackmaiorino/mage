#!/usr/bin/env bash
# Copies current eval game logs into a timestamped snapshot dir to defeat the
# in-place 50-file rotation cap. Idempotent: only copies files not already present.
set -euo pipefail

cd "$(dirname "$0")/.."

PROFILE="${MODEL_PROFILE:-Pauper-Standard}"
SRC="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/profiles/$PROFILE/logs/games/evaluation"
DST="local-training/local_pbt/eval_archive/$PROFILE"
mkdir -p "$DST"

count_new=0
for f in "$SRC"/*.txt; do
    [ -f "$f" ] || continue
    base=$(basename "$f")
    if [ ! -f "$DST/$base" ]; then
        cp "$f" "$DST/$base"
        count_new=$((count_new + 1))
    fi
done

archive_total=$(ls "$DST"/*.txt 2>/dev/null | wc -l)
echo "snapshot: copied $count_new new file(s); archive total = $archive_total"
