#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
default_venv="$repo_root/.mtgrl_venv_hpc"
venv_path="$default_venv"
python_bin="${PYTHON_BIN:-python3}"
torch_index_url="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-path)
      venv_path="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --torch-index-url)
      torch_index_url="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "Repo root: $repo_root"
echo "Venv path: $venv_path"
echo "Python bin: $python_bin"
echo "Torch index: $torch_index_url"

if [[ ! -d "$venv_path" ]]; then
  "$python_bin" -m venv "$venv_path"
fi

venv_python="$venv_path/bin/python"
venv_pip="$venv_path/bin/pip"

"$venv_pip" install --upgrade pip setuptools wheel
"$venv_pip" install --upgrade --index-url "$torch_index_url" "torch>=2.2.0"
"$venv_pip" install --upgrade \
  "py4j" \
  "numpy>=1.21.0" \
  "tensorboard>=2.12.0" \
  "torch-tb-profiler>=0.4.0" \
  "transformers>=4.30.0"

"$venv_python" - <<'PY'
import importlib
mods = ["py4j", "numpy", "torch", "tensorboard", "transformers"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append((m, str(exc)))
if missing:
    for m, e in missing:
        print(f"[FAIL] import {m}: {e}")
    raise SystemExit(1)
print("[OK] imports validated")
PY

echo
echo "Venv build complete."
echo "Use these env vars in Slurm jobs:"
echo "  export MTG_VENV_PATH=\"$venv_path\""
echo "  export PY_BRIDGE_CREATE_VENV=0"
echo "  export PY_BRIDGE_INSTALL_DEPS=0"

