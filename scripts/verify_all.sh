#!/usr/bin/env bash
# Fail-closed verification gate (Sol #86). Run before EVERY commit touching
# Java RL code or the kernel. No unconditional success echoes, no relative
# paths, no grep-filtered error detection: bare exit codes only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

if [[ -z "${JAVA_HOME:-}" ]]; then
  for candidate in \
    "/mnt/c/Program Files/Java/jdk-23" \
    "/mnt/c/Program Files/Java/jdk-1.8"; do
    if [[ -x "$candidate/bin/java.exe" || -x "$candidate/bin/java" ]]; then
      export JAVA_HOME="$candidate"
      export PATH="$JAVA_HOME/bin:$PATH"
      break
    fi
  done
fi

USE_WINDOWS_TOOLS=0
if command -v wslpath >/dev/null 2>&1 && command -v powershell.exe >/dev/null 2>&1; then
  USE_WINDOWS_TOOLS=1
  VERIFY_REPO_WIN="$(wslpath -w "$REPO")"
  VERIFY_REPO_WIN_PS="${VERIFY_REPO_WIN//\'/\'\'}"
fi

if [[ "$USE_WINDOWS_TOOLS" == "0" ]]; then
  PYTHON_BIN=""
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python 3 is required for verification" >&2
    exit 1
  fi
fi

run_ps_gate() {
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "\$ErrorActionPreference = 'Stop'; \$repo = '$VERIFY_REPO_WIN_PS'; Set-Location -LiteralPath \$repo; $1; if (\$LASTEXITCODE -ne \$null) { exit \$LASTEXITCODE }"
}

echo "[1/6] Maven install (AIRL + dependency modules; keeps local-repo jars fresh)..."
# install, not compile: bare exec:java invocations resolve dependency modules
# from installed jars. A stale jar (June 9) silently masked a month of Mage-core
# fixes until corpus identity checks caught it. Installing here guarantees the
# local repo always matches HEAD.
if [[ "$USE_WINDOWS_TOOLS" == "1" ]]; then
  run_ps_gate '& mvn.cmd -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install'
else
  mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests install
fi

echo "[2/6] Python syntax (ML service files)..."
if [[ "$USE_WINDOWS_TOOLS" == "1" ]]; then
  run_ps_gate '& py -3.12 -m py_compile (Join-Path $repo "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\MLPythonCode\py4j_entry_point.py") (Join-Path $repo "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\MLPythonCode\gpu_service_host.py") (Join-Path $repo "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\MLPythonCode\gpu_service_core.py")'
else
  "$PYTHON_BIN" -m py_compile \
    "$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/py4j_entry_point.py" \
    "$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_host.py" \
    "$REPO/Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode/gpu_service_core.py"
fi

echo "[3/6] Pauper manifest generator check..."
if [[ "$USE_WINDOWS_TOOLS" == "1" ]]; then
  run_ps_gate '& py -3.12 (Join-Path $repo "kernel\python\tools\generate_pauper_manifests.py") --check'
else
  "$PYTHON_BIN" "$REPO/kernel/python/tools/generate_pauper_manifests.py" --check
fi

echo "[4/6] Pauper manifest tests..."
if [[ "$USE_WINDOWS_TOOLS" == "1" ]]; then
  run_ps_gate '& py -3.12 -m unittest discover -v -s (Join-Path $repo "kernel\python\tests") -p "test_pauper_pool_manifest.py"'
else
  "$PYTHON_BIN" -m unittest discover -v -s "$REPO/kernel/python/tests" -p "test_pauper_pool_manifest.py"
fi

echo "[5/6] Kernel tests (release)..."
if [[ "$USE_WINDOWS_TOOLS" == "1" ]]; then
  run_ps_gate '& cargo test --release --all-targets --manifest-path (Join-Path $repo "kernel\Cargo.toml")'
else
  cargo test --release --all-targets --manifest-path "$REPO/kernel/Cargo.toml"
fi

echo "[6/6] Kernel clippy (-D warnings)..."
if [[ "$USE_WINDOWS_TOOLS" == "1" ]]; then
  run_ps_gate '& cargo clippy --release --all-targets --manifest-path (Join-Path $repo "kernel\Cargo.toml") -- -D warnings'
else
  cargo clippy --release --all-targets --manifest-path "$REPO/kernel/Cargo.toml" -- -D warnings
fi

echo "VERIFY_ALL: PASS (all gates on bare exit codes)"
