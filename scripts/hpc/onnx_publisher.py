#!/usr/bin/env python3
"""ONNX weight-publisher sidecar for the self-serve-satellite architecture.

Runs on the GPU training HEAD, decoupled from the learner. Watches each trained
profile's model_latest.pt and, when it changes, re-exports the full ONNX head set
to a fresh versioned dir under models/onnx/, atomically flips .active_dir, and
prunes old versions. CPU-self-serve satellites (PY_SERVICE_MODE=hybrid +
ONNX_FORCE_CPU=1) read these via OnnxInferenceModel's mtime reload over the shared
(Lustre) filesystem -> fresh weights with a <= poll-interval lag, no per-decision
round-trip.

Mirrors the proven export logic in scripts/run_local_pbt.py:export_onnx_models.
Env:
  ONNX_PUBLISH_PROFILES   comma list (default: TRAIN_PROFILES_LIST)
  RL_ARTIFACTS_ROOT       artifacts root holding profiles/<P>/models
  ONNX_PUBLISH_INTERVAL_S poll seconds (default 20)
  MODEL_D_MODEL/MODEL_NUM_LAYERS  per-profile arch (export_all_heads reads these)
"""
import os
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MLCODE = REPO / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/MLPythonCode"
sys.path.insert(0, str(MLCODE))
ONNX_EXPORT_VERSION = "3"


def log(msg):
    print(f"[ONNX_PUB {datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def _resolve_active(onnx_dir: Path) -> Path:
    pointer = onnx_dir / ".active_dir"
    if pointer.exists():
        name = pointer.read_text().strip()
        if name:
            p = (onnx_dir / name) if not os.path.isabs(name) else Path(name)
            if p.exists():
                return p
    return onnx_dir


def _prune(onnx_dir: Path, keep=3):
    vers = sorted([d for d in onnx_dir.glob("v*") if d.is_dir()], key=lambda d: d.stat().st_mtime)
    for d in vers[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


def export_profile(profile: str, artifacts_root: Path) -> bool:
    from onnx_export import export_all_heads
    models_dir = artifacts_root / "profiles" / profile / "models"
    model_path = models_dir / "model_latest.pt"
    onnx_dir = models_dir / "onnx"
    if not model_path.exists():
        return False
    active = _resolve_active(onnx_dir)
    action = active / "model_action.onnx"
    ver_file = active / ".export_version"
    cur_ver = ver_file.read_text().strip() if ver_file.exists() else ""
    if (onnx_dir / ".active_dir").exists() and action.exists() \
            and action.stat().st_mtime >= model_path.stat().st_mtime and cur_ver == ONNX_EXPORT_VERSION:
        return False  # up to date
    os.environ["MODEL_PROFILE"] = profile
    os.environ.setdefault("ONNX_EXPORT_FP16", "1")
    stamp = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S_%f")
    stage = onnx_dir / (stamp + "_staging")
    final = onnx_dir / stamp
    if stage.exists():
        shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    export_all_heads(str(model_path), str(stage))
    (stage / ".export_version").write_text(ONNX_EXPORT_VERSION, encoding="utf-8")
    # mulligan (optional)
    mull = models_dir / "mulligan_model.pt"
    if mull.exists():
        try:
            from onnx_export_mulligan import export_mulligan
            export_mulligan(str(mull), str(stage))
        except Exception as e:
            log(f"{profile}: mulligan export failed: {e}")
    stage.rename(final)
    (onnx_dir / ".active_dir").write_text(final.name, encoding="utf-8")
    _prune(onnx_dir)
    log(f"{profile}: published {final.name} in {time.time()-t0:.1f}s")
    return True


def main():
    profiles = os.getenv("ONNX_PUBLISH_PROFILES", os.getenv("TRAIN_PROFILES_LIST", "")).strip()
    profiles = [p.strip() for p in profiles.split(",") if p.strip()]
    artifacts = Path(os.getenv("RL_ARTIFACTS_ROOT",
                               str(REPO / "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl")))
    interval = int(os.getenv("ONNX_PUBLISH_INTERVAL_S", "20"))
    if not profiles:
        log("no profiles (set ONNX_PUBLISH_PROFILES / TRAIN_PROFILES_LIST); exiting")
        return 1
    once = str(os.getenv("ONNX_PUBLISH_ONCE", "")).strip().lower() in ("1", "true", "yes")
    if once:
        log(f"one-shot export of {profiles} under {artifacts}")
        rc = 0
        for p in profiles:
            try:
                export_profile(p, artifacts)
            except Exception as e:
                log(f"{p}: export error: {e}"); rc = 1
        return rc
    log(f"watching {profiles} under {artifacts} every {interval}s")
    while True:
        for p in profiles:
            try:
                export_profile(p, artifacts)
            except Exception as e:
                log(f"{p}: export error: {e}")
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
