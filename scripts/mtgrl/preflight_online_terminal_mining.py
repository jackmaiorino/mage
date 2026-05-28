#!/usr/bin/env python3
"""Preflight checks for online terminal-mining HPC launches.

The goal is to fail before a long Slurm run when the checkout, registry,
Python, Maven, or compute-node environment cannot support the mining harness.
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
PROFILES_ROOT = (
    REPO
    / "Mage.Server.Plugins"
    / "Mage.Player.AIRL"
    / "src"
    / "mage"
    / "player"
    / "ai"
    / "rl"
    / "profiles"
)

REQUIRED_SCRIPTS = [
    "scripts/mtgrl/run_online_terminal_mining_loop.py",
    "scripts/mtgrl/run_value_tree_shards.py",
    "scripts/mtgrl/export_terminal_line_value_targets.py",
    "scripts/mtgrl/summarize_terminal_line_search.py",
    "scripts/run_cp7_eval_sweep.py",
    "scripts/run_live_checkpoint_branch_miner.py",
]


def repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def add_result(results: List[Dict[str, object]], name: str, ok: bool, detail: str = "") -> None:
    results.append({"name": name, "ok": ok, "detail": detail})
    status = "OK" if ok else "FAIL"
    print(f"{status} {name}: {detail}", flush=True)


def normalize_registry(data: object, registry_path: Path) -> List[dict]:
    if isinstance(data, list):
        raw_entries = data
    elif isinstance(data, dict) and isinstance(data.get("profiles"), list):
        raw_entries = data["profiles"]
    elif isinstance(data, dict) and "profile" in data:
        raw_entries = [data]
    else:
        raise RuntimeError(
            "registry must be a list, an object with profiles[], or a single profile object: "
            + str(registry_path)
        )
    entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    if len(entries) != len(raw_entries):
        raise RuntimeError("registry contains non-object entries")
    return entries


def active_entries(registry_path: Path) -> List[dict]:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [entry for entry in normalize_registry(data, registry_path) if entry.get("active", False)]
    if not entries:
        raise RuntimeError(f"no active profiles in {registry_path}")
    return entries


def filter_entries(entries: List[dict], profiles: str) -> List[dict]:
    if not profiles.strip():
        return entries
    wanted = {profile.strip() for profile in profiles.split(",") if profile.strip()}
    filtered = [entry for entry in entries if str(entry.get("profile", "")) in wanted]
    present = {str(entry.get("profile", "")) for entry in filtered}
    missing = wanted - present
    if missing:
        raise RuntimeError(f"requested profile(s) not active in registry: {sorted(missing)}")
    return filtered


def read_deck_pool(deck_path: Path) -> List[Path]:
    if deck_path.suffix.lower() != ".txt":
        return [deck_path]
    decks: List[Path] = []
    for raw in deck_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = deck_path.parent / candidate
        decks.append(candidate.resolve())
    return decks


def filter_decks(decks: Iterable[Path], opponents: str) -> List[Path]:
    tokens = [token.strip().lower() for token in opponents.split(",") if token.strip()]
    if not tokens:
        return list(decks)
    out: List[Path] = []
    for deck in decks:
        haystack = f"{deck.stem.lower()} {deck.name.lower()} {str(deck).lower()}"
        if any(token in haystack for token in tokens):
            out.append(deck)
    return out


def check_py_compile(results: List[Dict[str, object]], scripts: Sequence[str]) -> None:
    for script in scripts:
        path = repo_path(script)
        if not path.is_file():
            add_result(results, f"script_exists:{script}", False, str(path))
            continue
        add_result(results, f"script_exists:{script}", True, str(path))
        try:
            py_compile.compile(str(path), doraise=True)
            add_result(results, f"py_compile:{script}", True, sys.executable)
        except py_compile.PyCompileError as exc:
            add_result(results, f"py_compile:{script}", False, str(exc))


def check_registry(results: List[Dict[str, object]], registry: Path, profiles: str, opponents: str) -> None:
    try:
        entries = filter_entries(active_entries(registry), profiles)
        add_result(results, "registry_load", True, f"active_selected={len(entries)} path={registry}")
    except Exception as exc:
        add_result(results, "registry_load", False, repr(exc))
        return
    for entry in entries:
        profile = str(entry.get("profile", ""))
        deck_path = repo_path(str(entry.get("deck_path", "")))
        add_result(results, f"profile_name:{profile}", bool(profile), profile)
        add_result(results, f"deck_path:{profile}", deck_path.is_file(), str(deck_path))
        try:
            decks = read_deck_pool(deck_path)
            existing = [deck for deck in decks if deck.is_file()]
            add_result(results, f"deck_pool:{profile}", len(existing) == len(decks) and bool(decks),
                       f"existing={len(existing)} total={len(decks)}")
            matched = filter_decks(existing, opponents)
            add_result(results, f"opponent_filter:{profile}", bool(matched),
                       f"opponents={opponents} matched={len(matched)}")
        except Exception as exc:
            add_result(results, f"deck_pool:{profile}", False, repr(exc))
        train_env = entry.get("train_env") or {}
        agent_deck = repo_path(str(train_env.get("RL_AGENT_DECK_LIST", "")))
        add_result(results, f"agent_deck:{profile}", agent_deck.is_file(), str(agent_deck))
        model_latest = PROFILES_ROOT / profile / "models" / "model_latest.pt"
        add_result(results, f"model_latest:{profile}", model_latest.is_file(), str(model_latest))


def run_probe(command: Sequence[str], timeout_sec: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_sec,
    )


def check_executables(results: List[Dict[str, object]], require_maven: bool, check_gpu: bool) -> None:
    python_path = Path(sys.executable)
    add_result(results, "python_executable", python_path.is_file(), str(python_path))
    mvn = shutil.which("mvn") or shutil.which("mvn.cmd")
    if mvn:
        try:
            proc = run_probe([mvn, "-v"], timeout_sec=60)
            add_result(results, "maven_executable", proc.returncode == 0, (proc.stdout or "").splitlines()[0])
        except Exception as exc:
            add_result(results, "maven_executable", False, repr(exc))
    else:
        add_result(results, "maven_executable", not require_maven, "mvn not found")
    if check_gpu:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            add_result(results, "gpu_visible", False, "nvidia-smi not found")
        else:
            try:
                proc = run_probe([nvidia_smi, "-L"], timeout_sec=30)
                add_result(results, "gpu_visible", proc.returncode == 0 and bool(proc.stdout.strip()),
                           (proc.stdout or "").strip().replace("\n", " | "))
            except Exception as exc:
                add_result(results, "gpu_visible", False, repr(exc))


def check_maven_reactor_modules(results: List[Dict[str, object]], enabled: bool) -> None:
    if not enabled:
        return
    pom = REPO / "pom.xml"
    if not pom.is_file():
        add_result(results, "maven_reactor_pom", False, str(pom))
        return
    add_result(results, "maven_reactor_pom", True, str(pom))
    try:
        root = ET.parse(str(pom)).getroot()
        ns = {"m": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        module_nodes = root.findall("./m:modules/m:module", ns) if ns else root.findall("./modules/module")
        modules = [str(node.text or "").strip() for node in module_nodes if str(node.text or "").strip()]
    except Exception as exc:
        add_result(results, "maven_reactor_modules_parse", False, repr(exc))
        return
    if not modules:
        add_result(results, "maven_reactor_modules_parse", False, "no modules found")
        return
    missing = [module for module in modules if not (REPO / module).is_dir()]
    add_result(
        results,
        "maven_reactor_modules",
        not missing,
        f"modules={len(modules)} missing={','.join(missing) if missing else 'none'}",
    )


def check_maven_compile(results: List[Dict[str, object]], enabled: bool, online: bool, timeout_sec: int) -> None:
    if not enabled:
        return
    mvn = shutil.which("mvn") or shutil.which("mvn.cmd")
    if not mvn:
        add_result(results, "maven_compile", False, "mvn not found")
        return
    command = [mvn]
    if not online:
        command.append("-o")
    command.extend([
        "-q",
        "-pl",
        "Mage.Server.Plugins/Mage.Player.AIRL",
        "-am",
        "-DskipTests",
        "compile",
    ])
    try:
        proc = run_probe(command, timeout_sec=max(1, int(timeout_sec)))
        mode = "online" if online else "offline"
        detail = f"{mode} compile passed" if proc.returncode == 0 else (proc.stdout or "")[-4000:]
        add_result(results, f"maven_{mode}_compile", proc.returncode == 0, detail.strip())
    except Exception as exc:
        mode = "online" if online else "offline"
        add_result(results, f"maven_{mode}_compile", False, repr(exc))


def check_runtime_bundle(results: List[Dict[str, object]]) -> None:
    raw = os.environ.get("MAGE_RL_RUNTIME_DIR", "").strip()
    if not raw:
        return
    runtime_dir = repo_path(raw)
    add_result(results, "runtime_dir", runtime_dir.is_dir(), str(runtime_dir))
    app_jars = sorted((runtime_dir / "app").glob("*.jar")) if runtime_dir.is_dir() else []
    lib_jars = sorted((runtime_dir / "lib").glob("*.jar")) if runtime_dir.is_dir() else []
    add_result(results, "runtime_app_jars", bool(app_jars), f"count={len(app_jars)}")
    add_result(results, "runtime_lib_jars", bool(lib_jars), f"count={len(lib_jars)}")


def check_eval_database(results: List[Dict[str, object]]) -> None:
    override = os.environ.get("CP7_EVAL_DB_SOURCE", "").strip()
    candidates: List[Path] = []
    if override:
        path = repo_path(override)
        candidates.append(path / "cards.h2.mv.db" if path.is_dir() else path)
    candidates.extend([
        REPO / "db_eval" / "cards.h2.mv.db",
        REPO / "db" / "cards.h2.mv.db",
    ])
    existing = [path for path in candidates if path.is_file()]
    detail = str(existing[0]) if existing else "checked=" + ",".join(str(path) for path in candidates)
    add_result(results, "cards_db", bool(existing), detail)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--profiles", default="")
    parser.add_argument("--opponents", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--require-maven", action="store_true")
    parser.add_argument("--check-gpu", action="store_true")
    parser.add_argument("--check-maven-compile", action="store_true")
    parser.add_argument("--maven-online", action="store_true", help="Allow Maven to fetch dependencies during compile preflight.")
    parser.add_argument(
        "--maven-compile-timeout-sec",
        type=int,
        default=1800,
        help="Timeout for the Maven compile preflight.",
    )
    parser.add_argument(
        "--check-reactor-modules",
        action="store_true",
        help="Verify all modules listed by the root Maven reactor exist in this checkout.",
    )
    parser.add_argument("--extra-script", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    results: List[Dict[str, object]] = []
    scripts = REQUIRED_SCRIPTS + list(args.extra_script)
    check_py_compile(results, scripts)
    registry = repo_path(args.registry)
    add_result(results, "registry_exists", registry.is_file(), str(registry))
    if registry.is_file():
        check_registry(results, registry, args.profiles, args.opponents)
    check_executables(results, args.require_maven, args.check_gpu)
    check_runtime_bundle(results)
    check_eval_database(results)
    check_maven_reactor_modules(results, args.check_reactor_modules or args.check_maven_compile)
    check_maven_compile(results, args.check_maven_compile, args.maven_online, args.maven_compile_timeout_sec)
    ok = all(bool(result["ok"]) for result in results)
    payload = {
        "ok": ok,
        "cwd": str(REPO),
        "python": sys.executable,
        "results": results,
    }
    if args.output_json:
        out = repo_path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": ok, "checks": len(results)}, sort_keys=True), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
