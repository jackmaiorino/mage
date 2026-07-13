#!/usr/bin/env bash
# Fail-closed verification gate (Sol #86). Run before EVERY commit touching
# Java RL code or the kernel. No unconditional success echoes, no relative
# paths, no grep-filtered error detection: bare exit codes only.
set -euo pipefail

REPO="/c/Users/Jack/IdeaProjects/mage"
cd "$REPO"

echo "[1/4] Maven install (AIRL + dependency modules; keeps local-repo jars fresh)..."
# install, not compile: bare exec:java invocations resolve dependency modules
# from installed jars. A stale jar (June 9) silently masked a month of Mage-core
# fixes until corpus identity checks caught it. Installing here guarantees the
# local repo always matches HEAD.
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install

echo "[2/4] Python syntax (ML service files)..."
py -3.12 -m py_compile \
  "$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py" \
  "$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_host.py" \
  "$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_core.py"

echo "[3/4] Kernel tests (release)..."
cargo test --release --all-targets --manifest-path "$REPO/kernel/Cargo.toml"

echo "[4/4] Kernel clippy (-D warnings)..."
cargo clippy --release --all-targets --manifest-path "$REPO/kernel/Cargo.toml" -- -D warnings

echo "VERIFY_ALL: PASS (all gates on bare exit codes)"
