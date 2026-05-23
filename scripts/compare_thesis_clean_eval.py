#!/usr/bin/env python3
"""Compare thesis-clean eval sweeps against a baseline promotion gate.

The gate is intentionally conservative:
  * CP7 must be at least as good as baseline by default.
  * CP1/CP3 may only give back a small configured tolerance.
  * Any individual matchup regression beyond tolerance fails the gate.

Inputs are run directories produced by scripts/run_cp7_eval_sweep.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class ProfileSummary:
    wins: int
    total: int
    winrate: float


@dataclass
class MatchupSummary:
    opponent_deck: str
    wins: int
    total: int
    winrate: float


@dataclass
class EvalRun:
    label: str
    path: Path
    profile: ProfileSummary
    matchups: Dict[str, MatchupSummary]


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _to_int(row: dict, key: str) -> int:
    raw = str(row.get(key, "")).strip()
    return int(raw)


def _to_float(row: dict, key: str) -> float:
    raw = str(row.get(key, "")).strip()
    return float(raw)


def load_eval_run(label: str, path: Path) -> EvalRun:
    profile_rows = _read_csv(path / "profile_summary.csv")
    if len(profile_rows) != 1:
        raise ValueError(f"{path}: expected exactly one profile_summary row, got {len(profile_rows)}")
    profile_row = profile_rows[0]
    profile = ProfileSummary(
        wins=_to_int(profile_row, "wins"),
        total=_to_int(profile_row, "total"),
        winrate=_to_float(profile_row, "winrate"),
    )

    matchup_rows = _read_csv(path / "matchup_summary.csv")
    matchups: Dict[str, MatchupSummary] = {}
    for row in matchup_rows:
        deck = str(row.get("opponent_deck", "")).strip()
        if not deck:
            continue
        matchups[deck] = MatchupSummary(
            opponent_deck=deck,
            wins=_to_int(row, "wins"),
            total=_to_int(row, "total"),
            winrate=_to_float(row, "winrate"),
        )
    if not matchups:
        raise ValueError(f"{path}: no matchup_summary rows")
    return EvalRun(label=label, path=path, profile=profile, matchups=matchups)


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _run_rows(candidate: Dict[str, EvalRun], baseline: Dict[str, EvalRun]) -> List[dict]:
    rows = []
    for label in ("CP1", "CP3", "CP7"):
        cand = candidate[label]
        base = baseline[label]
        rows.append(
            {
                "skill": label,
                "candidate": cand.profile.winrate,
                "baseline": base.profile.winrate,
                "delta": cand.profile.winrate - base.profile.winrate,
                "candidate_record": f"{cand.profile.wins}/{cand.profile.total}",
                "baseline_record": f"{base.profile.wins}/{base.profile.total}",
            }
        )
    return rows


def _matchup_rows(candidate: Dict[str, EvalRun], baseline: Dict[str, EvalRun]) -> List[dict]:
    rows = []
    for label in ("CP1", "CP3", "CP7"):
        cand = candidate[label]
        base = baseline[label]
        decks = sorted(set(cand.matchups.keys()) | set(base.matchups.keys()))
        for deck in decks:
            c = cand.matchups.get(deck)
            b = base.matchups.get(deck)
            if c is None or b is None:
                rows.append(
                    {
                        "skill": label,
                        "opponent": deck,
                        "candidate": None if c is None else c.winrate,
                        "baseline": None if b is None else b.winrate,
                        "delta": None,
                        "candidate_record": "" if c is None else f"{c.wins}/{c.total}",
                        "baseline_record": "" if b is None else f"{b.wins}/{b.total}",
                    }
                )
            else:
                rows.append(
                    {
                        "skill": label,
                        "opponent": deck,
                        "candidate": c.winrate,
                        "baseline": b.winrate,
                        "delta": c.winrate - b.winrate,
                        "candidate_record": f"{c.wins}/{c.total}",
                        "baseline_record": f"{b.wins}/{b.total}",
                    }
                )
    return rows


def evaluate_gate(
    candidate: Dict[str, EvalRun],
    baseline: Dict[str, EvalRun],
    cp_low_tolerance: float,
    cp7_min_delta: float,
    max_matchup_regression: float,
    min_games_ratio: float,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    for row in _run_rows(candidate, baseline):
        label = row["skill"]
        delta = float(row["delta"])
        cand_total = candidate[label].profile.total
        base_total = baseline[label].profile.total
        required_total = int(round(base_total * min_games_ratio))
        if cand_total < required_total:
            failures.append(
                f"{label} has only {cand_total} games; required at least {required_total} "
                f"({min_games_ratio:.2f}x baseline total)"
            )
        if label == "CP7":
            if delta < cp7_min_delta:
                failures.append(
                    f"CP7 overall delta {_fmt_pct(delta)} is below required {_fmt_pct(cp7_min_delta)}"
                )
        elif delta < -cp_low_tolerance:
            failures.append(
                f"{label} overall regressed {_fmt_pct(-delta)}, beyond tolerance {_fmt_pct(cp_low_tolerance)}"
            )

    for row in _matchup_rows(candidate, baseline):
        if row["candidate"] is None or row["baseline"] is None:
            failures.append(f"{row['skill']} matchup missing in candidate or baseline: {row['opponent']}")
            continue
        delta = float(row["delta"])
        if delta < -max_matchup_regression:
            failures.append(
                f"{row['skill']} {row['opponent']} regressed {_fmt_pct(-delta)}, "
                f"beyond tolerance {_fmt_pct(max_matchup_regression)}"
            )
    return len(failures) == 0, failures


def _markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> List[str]:
    headers = list(headers)
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def render_markdown(candidate: Dict[str, EvalRun], baseline: Dict[str, EvalRun], passed: bool, failures: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Thesis-Clean Eval Gate")
    lines.append("")
    lines.append(f"Verdict: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Skill", "Candidate", "Baseline", "Delta", "Candidate Record", "Baseline Record"],
            [
                [
                    row["skill"],
                    _fmt_pct(row["candidate"]),
                    _fmt_pct(row["baseline"]),
                    _fmt_pct(row["delta"]),
                    row["candidate_record"],
                    row["baseline_record"],
                ]
                for row in _run_rows(candidate, baseline)
            ],
        )
    )
    lines.append("")
    lines.append("## Matchups")
    lines.append("")
    lines.extend(
        _markdown_table(
            ["Skill", "Opponent", "Candidate", "Baseline", "Delta", "Candidate Record", "Baseline Record"],
            [
                [
                    row["skill"],
                    row["opponent"],
                    "" if row["candidate"] is None else _fmt_pct(row["candidate"]),
                    "" if row["baseline"] is None else _fmt_pct(row["baseline"]),
                    "" if row["delta"] is None else _fmt_pct(row["delta"]),
                    row["candidate_record"],
                    row["baseline_record"],
                ]
                for row in _matchup_rows(candidate, baseline)
            ],
        )
    )
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(f"- {failure}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cp1", type=Path, required=True)
    parser.add_argument("--candidate-cp3", type=Path, required=True)
    parser.add_argument("--candidate-cp7", type=Path, required=True)
    parser.add_argument("--baseline-cp1", type=Path, required=True)
    parser.add_argument("--baseline-cp3", type=Path, required=True)
    parser.add_argument("--baseline-cp7", type=Path, required=True)
    parser.add_argument("--cp-low-tolerance", type=float, default=0.02)
    parser.add_argument("--cp7-min-delta", type=float, default=0.0)
    parser.add_argument("--max-matchup-regression", type=float, default=0.10)
    parser.add_argument("--min-games-ratio", type=float, default=0.90)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    candidate = {
        "CP1": load_eval_run("CP1", args.candidate_cp1),
        "CP3": load_eval_run("CP3", args.candidate_cp3),
        "CP7": load_eval_run("CP7", args.candidate_cp7),
    }
    baseline = {
        "CP1": load_eval_run("CP1", args.baseline_cp1),
        "CP3": load_eval_run("CP3", args.baseline_cp3),
        "CP7": load_eval_run("CP7", args.baseline_cp7),
    }
    passed, failures = evaluate_gate(
        candidate,
        baseline,
        cp_low_tolerance=args.cp_low_tolerance,
        cp7_min_delta=args.cp7_min_delta,
        max_matchup_regression=args.max_matchup_regression,
        min_games_ratio=args.min_games_ratio,
    )
    markdown = render_markdown(candidate, baseline, passed, failures)
    print(markdown)

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.json_out:
        payload = {
            "passed": passed,
            "failures": failures,
            "overall": _run_rows(candidate, baseline),
            "matchups": _matchup_rows(candidate, baseline),
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
