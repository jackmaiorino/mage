#!/usr/bin/env python3
"""Post-hoc statistical analysis for cp7 eval sweeps.

Turns raw eval run directories (produced by scripts/run_cp7_eval_sweep.py) into
an interpretable report: winrate with confidence intervals, the Spy-combo
execution funnel/rate parsed from eval game logs, and -- when given a candidate
and a source run -- a paired comparison (McNemar exact + bootstrap CI) so a
delta is reported with a p-value instead of "+1 game".

Thesis note: winrate is the objective. Combo-execution rate is a correlated
proxy instrumented alongside it; divergence between the two is a debugging
signal, not a separate reward.

This tool reads existing artifacts only. It does not launch games, change the
harness, or touch the engine.

Usage:
  # Summarize one or more runs (absolute winrate + combo funnel, per matchup)
  py -3.12 scripts/mtgrl/eval_report.py --run <run_dir> [<run_dir> ...]

  # Paired candidate-vs-source comparison (aligned by agent/opponent/chunk seed)
  py -3.12 scripts/mtgrl/eval_report.py --candidate <cand_dir> --source <src_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest

# --- Combo signatures (mirrors scripts/mtgrl/summarize_terminal_line_search.py) ---
# Only the Spy-combo deck can cast these cards, so matching by chosen text alone
# isolates the agent's decisions without needing to disambiguate the player.
COMBO_FUNNEL = [
    # Either Dread Return resolution (flashback or hard cast) executes the combo;
    # match both so full_combo is not undercounted on hard-cast lines.
    ("spy", 3, lambda d: _is_act(d) and _has(d, "Cast Balustrade Spy")),
    ("dread_return", 3, lambda d: _has(d, "Flashback sacrifice three creatures")
        or _has(d, "Cast Dread Return") or _has(d, "Dread Return")),
    ("lotleth_target", 2, lambda d: _is_select(d) and _has(d, "Lotleth Giant")),
    ("lotleth_cast", 1, lambda d: _has(d, "Cast Lotleth Giant")),
]


def _has(decision: dict, needle: str) -> bool:
    blob = " ".join(str(t) for t in decision.get("chosen_texts", []) or [])
    return needle.lower() in blob.lower()


def _is_act(decision: dict) -> bool:
    return decision.get("action_type", "") == "ACTIVATE_ABILITY_OR_SPELL"


def _is_select(decision: dict) -> bool:
    # SELECT_TARGETS where the chosen/candidate set references Lotleth Giant.
    if decision.get("action_type", "") != "SELECT_TARGETS":
        return False
    blob = " ".join(
        str(t)
        for t in (decision.get("chosen_texts", []) or []) + (decision.get("candidate_texts", []) or [])
    )
    return "lotleth giant" in blob.lower()


# ---------------------------------------------------------------------------
# Game-log parsing
# ---------------------------------------------------------------------------
def parse_game_log(path: Path) -> dict:
    """Extract per-game outcome and combo funnel flags from one eval game log."""
    flags = {name: False for name, _, _ in COMBO_FUNNEL}
    outcome = "unknown"
    decisions = 0
    seen_replay = False  # current-format combo detection requires REPLAY_DECISION_JSON
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("REPLAY_DECISION_JSON:"):
                    decisions += 1
                    seen_replay = True
                    try:
                        d = json.loads(line.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        continue
                    for name, _w, pred in COMBO_FUNNEL:
                        if not flags[name] and pred(d):
                            flags[name] = True
                elif line.startswith("RESULT:"):
                    token = line.split(":", 1)[1].strip().upper()
                    if token.startswith("WIN"):
                        outcome = "win"
                    elif token.startswith("LOSS") or token.startswith("LOSE"):
                        outcome = "loss"
                    elif token.startswith("DRAW"):
                        outcome = "draw"
    except OSError:
        pass
    if not seen_replay:
        # Legacy log format (no REPLAY_DECISION_JSON): outcome is still parseable
        # from RESULT, but combo signatures are unavailable -- report as unknown
        # rather than a misleading 0.
        return {
            "path": str(path), "outcome": outcome, "won": outcome == "win",
            "decisions": decisions, "combo_score": None, "full_combo": None,
            **{f"has_{name}": None for name, _, _ in COMBO_FUNNEL},
        }
    combo_score = sum(w for name, w, _ in COMBO_FUNNEL if flags[name])
    # full combo == cast Spy + flashback Dread Return + target Lotleth Giant
    full_combo = flags["spy"] and flags["dread_return"] and flags["lotleth_target"]
    return {
        "path": str(path),
        "outcome": outcome,
        "won": outcome == "win",
        "decisions": decisions,
        "combo_score": combo_score,
        "full_combo": full_combo,
        **{f"has_{name}": flags[name] for name, _, _ in COMBO_FUNNEL},
    }


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------
def _read_matchups(run_dir: Path) -> List[dict]:
    path = run_dir / "matchups.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _run_profile(run_dir: Path) -> str:
    man = run_dir / "manifest.json"
    if man.exists():
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
            p = data.get("profiles_filter") or ""
            if p:
                return p
        except ValueError:
            pass
    return ""


def collect_run(run_dir: Path) -> dict:
    """Load one eval run into per-game records keyed by (agent, opponent, chunk).

    Per-game outcome and combo flags come from game logs when present; winrate
    falls back to matchups.csv when logs are absent (combo metrics then unknown).
    """
    run_dir = run_dir.resolve()
    rows = _read_matchups(run_dir)
    games: List[dict] = []
    matchups_only_wins = 0
    matchups_only_total = 0
    for row in rows:
        agent = (row.get("agent_deck") or "").strip()
        opp = (row.get("opponent_deck") or "").strip()
        try:
            chunk = int(row.get("chunk_index") or 0)
        except ValueError:
            chunk = 0
        try:
            wins = int(row.get("wins") or 0)
            total = int(row.get("total") or 0)
        except ValueError:
            wins, total = 0, 0
        matchups_only_wins += wins
        matchups_only_total += total

        # Locate this matchup's game logs: dir == result_file stem.
        result_file = row.get("result_file") or ""
        log_dir = None
        if result_file:
            stem = Path(result_file).stem
            cand = run_dir / "game_logs" / stem
            if cand.is_dir():
                log_dir = cand
        log_files = sorted(log_dir.glob("*.txt")) if log_dir else []

        if log_files:
            for lf in log_files:
                rec = parse_game_log(lf)
                rec.update({"agent_deck": agent, "opponent_deck": opp, "chunk": chunk,
                            "key": (agent, opp, chunk), "has_log": True})
                games.append(rec)
        else:
            # No log: emit one record per game from matchups.csv (combo unknown).
            for i in range(max(total, 0)):
                games.append({
                    "agent_deck": agent, "opponent_deck": opp, "chunk": chunk,
                    "key": (agent, opp, chunk), "has_log": False,
                    "won": i < wins, "outcome": "win" if i < wins else "loss",
                    "combo_score": None, "full_combo": None,
                    **{f"has_{name}": None for name, _, _ in COMBO_FUNNEL},
                })

    return {
        "run_dir": str(run_dir),
        "profile": _run_profile(run_dir),
        "games": games,
        "n_games": len(games),
        "matchups_csv_wins": matchups_only_wins,
        "matchups_csv_total": matchups_only_total,
        "has_logs": any(g.get("has_log") for g in games),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate(flag_games: List[dict], field: str) -> Optional[Tuple[int, int, float]]:
    vals = [g[field] for g in flag_games if g.get(field) is not None]
    if not vals:
        return None
    hits = sum(1 for v in vals if v)
    return (hits, len(vals), hits / len(vals))


def summarize_games(games: List[dict]) -> dict:
    n = len(games)
    wins = sum(1 for g in games if g.get("won"))
    lo, hi = wilson_ci(wins, n)
    out = {
        "n": n,
        "wins": wins,
        "winrate": (wins / n) if n else 0.0,
        "winrate_ci95": [round(lo, 4), round(hi, 4)],
    }
    # Combo funnel (only over games with current-format combo data).
    logged = [g for g in games if g.get("full_combo") is not None]
    if logged:
        funnel = {}
        for name, _, _ in COMBO_FUNNEL:
            r = _rate(logged, f"has_{name}")
            if r:
                funnel[name] = round(r[2], 4)
        full = _rate(logged, "full_combo")
        combo_wins = sum(1 for g in logged if g.get("full_combo") and g.get("won"))
        out["combo"] = {
            "n_logged": len(logged),
            "funnel": funnel,
            "full_combo_rate": round(full[2], 4) if full else None,
            "combo_win_rate": round(combo_wins / len(logged), 4),
            "mean_combo_score": round(
                float(np.mean([g["combo_score"] for g in logged if g.get("combo_score") is not None])), 3
            ),
        }
    return out


def per_matchup(games: List[dict]) -> Dict[str, dict]:
    buckets: Dict[str, List[dict]] = {}
    for g in games:
        key = f"{g['agent_deck']} vs {g['opponent_deck']}"
        buckets.setdefault(key, []).append(g)
    return {k: summarize_games(v) for k, v in sorted(buckets.items())}


def paired_compare(cand: dict, src: dict, n_boot: int, seed: int) -> dict:
    """Align candidate/source by (agent, opponent, chunk) and test the delta."""
    def index(run: dict) -> Dict[tuple, List[dict]]:
        idx: Dict[tuple, List[dict]] = {}
        for g in run["games"]:
            idx.setdefault(g["key"], []).append(g)
        return idx

    ci, si = index(cand), index(src)
    shared = sorted(set(ci) & set(si))
    # Use chunks present in both with exactly one game each side (clean pairing).
    pairs = []
    for k in shared:
        if len(ci[k]) == 1 and len(si[k]) == 1:
            pairs.append((ci[k][0], si[k][0]))

    n = len(pairs)
    if n == 0:
        return {"error": "no aligned single-game chunks between candidate and source",
                "candidate_keys": len(ci), "source_keys": len(si), "shared_keys": len(shared)}

    c_win = np.array([1 if c["won"] else 0 for c, _ in pairs])
    s_win = np.array([1 if s["won"] else 0 for _, s in pairs])
    # McNemar discordant counts.
    b = int(np.sum((c_win == 1) & (s_win == 0)))  # candidate win, source loss
    c = int(np.sum((c_win == 0) & (s_win == 1)))  # candidate loss, source win
    mcnemar_p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0

    rng = np.random.default_rng(seed)
    diff = c_win - s_win
    boot = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1) if n else np.array([0.0])
    delta = float(diff.mean())
    out = {
        "n_pairs": n,
        "candidate_winrate": round(float(c_win.mean()), 4),
        "source_winrate": round(float(s_win.mean()), 4),
        "winrate_delta": round(delta, 4),
        "winrate_delta_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                               round(float(np.percentile(boot, 97.5)), 4)],
        "mcnemar": {"cand_win_src_loss": b, "cand_loss_src_win": c, "exact_p": round(float(mcnemar_p), 4)},
        "significant_at_0.05": bool(mcnemar_p < 0.05),
    }
    # Paired combo-fire delta when both sides have logs.
    if all(c.get("has_log") and s.get("has_log") for c, s in pairs):
        c_combo = np.array([1 if c["full_combo"] else 0 for c, _ in pairs])
        s_combo = np.array([1 if s["full_combo"] else 0 for _, s in pairs])
        cdiff = c_combo - s_combo
        cboot = rng.choice(cdiff, size=(n_boot, n), replace=True).mean(axis=1)
        out["combo_fire_delta"] = round(float(cdiff.mean()), 4)
        out["combo_fire_delta_ci95"] = [round(float(np.percentile(cboot, 2.5)), 4),
                                        round(float(np.percentile(cboot, 97.5)), 4)]
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_run(run: dict) -> List[str]:
    overall = summarize_games(run["games"])
    lines = [
        f"## {run['profile'] or Path(run['run_dir']).name}",
        f"- run_dir: `{run['run_dir']}`",
    ]
    # AUTHORITATIVE winrate = full game count from matchups.csv. Game logs are
    # capped (~50/run), so the logged subset is a NOISY estimator (+/-7-8pp at
    # n=50) -- never quote it as the winrate. Lead with full-n; show the logged
    # subset only as the combo-funnel sample size.
    mcw, mct = run.get("matchups_csv_wins", 0), run.get("matchups_csv_total", 0)
    if mct and mct != overall["n"]:
        lo, hi = wilson_ci(mcw, mct)
        lines.append(f"- winrate (full n, matchups.csv): {mcw}/{mct} "
                     f"winrate={mcw / mct:.3f} CI95=[{lo:.3f}, {hi:.3f}]")
        lines.append(f"- logged subset (combo sample only, NOT the winrate): "
                     f"{overall['wins']}/{overall['n']} wr={overall['winrate']:.3f}")
    else:
        lines.append(f"- overall: {overall['wins']}/{overall['n']} "
                     f"winrate={overall['winrate']:.3f} "
                     f"CI95=[{overall['winrate_ci95'][0]:.3f}, {overall['winrate_ci95'][1]:.3f}]")
    if "combo" in overall:
        cb = overall["combo"]
        lines.append(
            f"- combo: full_combo_rate={cb['full_combo_rate']} "
            f"combo_win_rate={cb['combo_win_rate']} mean_score={cb['mean_combo_score']} "
            f"(n_logged={cb['n_logged']})"
        )
        lines.append(f"  - funnel: {cb['funnel']}")
    elif not run["has_logs"]:
        lines.append("- combo: (no game logs in this run -- winrate from matchups.csv only)")
    else:
        lines.append("- combo: (legacy log format without REPLAY_DECISION_JSON -- combo data unavailable)")
    lines.append("- per matchup:")
    for mk, ms in per_matchup(run["games"]).items():
        seg = (f"    - {mk}: {ms['wins']}/{ms['n']} wr={ms['winrate']:.3f} "
               f"CI=[{ms['winrate_ci95'][0]:.2f},{ms['winrate_ci95'][1]:.2f}]")
        if "combo" in ms:
            seg += f" full_combo={ms['combo']['full_combo_rate']}"
        lines.append(seg)
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="*", default=[], help="One or more run dirs to summarize.")
    ap.add_argument("--candidate", default="", help="Candidate run dir (paired mode).")
    ap.add_argument("--source", default="", help="Source/baseline run dir (paired mode).")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="Write JSON report here (default: <primary>/eval_report.json).")
    args = ap.parse_args()

    report: dict = {"runs": [], "paired": None}
    out_lines: List[str] = ["# Eval report", ""]
    primary_dir: Optional[Path] = None

    run_dirs = list(args.run)
    if args.candidate:
        run_dirs.append(args.candidate)
    if args.source:
        run_dirs.append(args.source)
    if not run_dirs:
        ap.error("provide --run, or --candidate and --source")

    loaded: Dict[str, dict] = {}
    for rd in run_dirs:
        p = Path(rd)
        if not p.exists():
            print(f"WARNING: run dir not found: {rd}", file=sys.stderr)
            continue
        run = collect_run(p)
        loaded[str(p.resolve())] = run
        if primary_dir is None:
            primary_dir = p.resolve()

    for rd in args.run:
        p = Path(rd).resolve()
        if str(p) in loaded:
            run = loaded[str(p)]
            mcw, mct = run.get("matchups_csv_wins", 0), run.get("matchups_csv_total", 0)
            full_n = None
            if mct:
                flo, fhi = wilson_ci(mcw, mct)
                full_n = {"wins": mcw, "n": mct, "winrate": round(mcw / mct, 4),
                          "winrate_ci95": [round(flo, 4), round(fhi, 4)]}
            report["runs"].append({"run_dir": run["run_dir"], "profile": run["profile"],
                                   "winrate_full_n": full_n,
                                   "summary": summarize_games(run["games"]),
                                   "per_matchup": per_matchup(run["games"])})
            out_lines += _fmt_run(run) + [""]

    if args.candidate and args.source:
        cp, sp = Path(args.candidate).resolve(), Path(args.source).resolve()
        if str(cp) in loaded and str(sp) in loaded:
            cand, src = loaded[str(cp)], loaded[str(sp)]
            out_lines += ["## Candidate", *_fmt_run(cand), "",
                          "## Source", *_fmt_run(src), ""]
            paired = paired_compare(cand, src, args.bootstrap, args.seed)
            report["paired"] = {"candidate": cand["run_dir"], "source": src["run_dir"], **paired}
            report["runs"] += [
                {"role": "candidate", "run_dir": cand["run_dir"], "profile": cand["profile"],
                 "summary": summarize_games(cand["games"])},
                {"role": "source", "run_dir": src["run_dir"], "profile": src["profile"],
                 "summary": summarize_games(src["games"])},
            ]
            out_lines.append("## Paired comparison (candidate - source)")
            if "error" in paired:
                out_lines.append(f"- ERROR: {paired['error']}")
            else:
                out_lines.append(
                    f"- n_pairs={paired['n_pairs']} "
                    f"cand_wr={paired['candidate_winrate']:.3f} src_wr={paired['source_winrate']:.3f} "
                    f"delta={paired['winrate_delta']:+.3f} "
                    f"CI95=[{paired['winrate_delta_ci95'][0]:+.3f}, {paired['winrate_delta_ci95'][1]:+.3f}]")
                m = paired["mcnemar"]
                out_lines.append(
                    f"- McNemar: cand-win/src-loss={m['cand_win_src_loss']} "
                    f"cand-loss/src-win={m['cand_loss_src_win']} exact_p={m['exact_p']} "
                    f"-> {'SIGNIFICANT' if paired['significant_at_0.05'] else 'not significant'} at 0.05")
                if "combo_fire_delta" in paired:
                    out_lines.append(
                        f"- combo-fire delta={paired['combo_fire_delta']:+.3f} "
                        f"CI95=[{paired['combo_fire_delta_ci95'][0]:+.3f}, "
                        f"{paired['combo_fire_delta_ci95'][1]:+.3f}]")
            out_lines.append("")

    text = "\n".join(out_lines)
    print(text)

    out_path = Path(args.out) if args.out else (primary_dir / "eval_report.json" if primary_dir else None)
    if out_path:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[wrote {out_path}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
