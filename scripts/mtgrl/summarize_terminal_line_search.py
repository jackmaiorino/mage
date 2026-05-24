#!/usr/bin/env python3
"""Summarize terminal-line checkpoint search traces into compact training-gate artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


TERMINAL_LINE_CSV = "terminal_line_search.csv"
TRACE_RE = re.compile(
    r"^(?P<step>\d+):(?P<role>[^:]*):action=(?P<action>.*?):indices=(?P<indices>.*?):"
    r"texts=(?P<texts>.*?):candidates=(?P<candidates>\d+):candidate_hash=(?P<candidate_hash>.*?):"
    r"state_hash=(?P<state_hash>.*?):reason=(?P<reason>.*)$"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize terminal-line branch search CSVs and extract compact combo markers."
    )
    parser.add_argument("run_dir", help="Run directory containing terminal_line_search.csv artifacts.")
    parser.add_argument("--out-json", default="", help="Optional JSON output path.")
    parser.add_argument("--out-md", default="", help="Optional Markdown output path.")
    parser.add_argument("--out-csv", default="", help="Optional compact row CSV output path.")
    parser.add_argument(
        "--prefer-merged",
        action="store_true",
        help="Use root-level terminal_line_search.csv when present instead of recursive shard CSVs.",
    )
    return parser.parse_args(argv)


def read_csvs(run_dir: Path, prefer_merged: bool) -> List[dict]:
    root_file = run_dir / TERMINAL_LINE_CSV
    if prefer_merged and root_file.exists():
        paths = [root_file]
    else:
        paths = [
            path for path in sorted(run_dir.rglob(TERMINAL_LINE_CSV))
            if path.is_file() and path.stat().st_size > 0
        ]
        if root_file.exists() and len(paths) > 1:
            paths = [path for path in paths if path != root_file]
    rows: List[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                out = dict(row)
                out["_artifact"] = str(path)
                rows.append(out)
    return rows


def boolish(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def as_int(row: dict, key: str, default: int = 0) -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def pct(numer: int, denom: int) -> float:
    return 0.0 if denom <= 0 else round(numer / denom, 6)


def split_trace(raw: str) -> List[dict]:
    out: List[dict] = []
    if not raw:
        return out
    for chunk in str(raw).split("|"):
        text = chunk.strip()
        if not text:
            continue
        match = TRACE_RE.match(text)
        if not match:
            out.append({"raw": text, "texts": text, "action": "", "step": -1})
            continue
        event = match.groupdict()
        try:
            event["step"] = int(event.get("step", "-1"))
        except ValueError:
            event["step"] = -1
        out.append(event)
    return out


def contains(text: str, needle: str) -> bool:
    return needle.lower() in str(text or "").lower()


def first_step(events: Iterable[dict], predicate) -> int:
    for event in events:
        if predicate(event):
            return int(event.get("step", -1))
    return -1


def trace_features(row: dict) -> dict:
    events = split_trace(row.get("line_trace", ""))
    texts = [str(event.get("texts", "")) for event in events]
    actions = [str(event.get("action", "")) for event in events]

    spy_step = first_step(events, lambda e: contains(e.get("texts", ""), "Cast Balustrade Spy"))
    dread_step = first_step(events, lambda e: contains(e.get("texts", ""), "Flashback sacrifice three creatures"))
    lotleth_target_step = first_step(
        events,
        lambda e: contains(e.get("action", ""), "SELECT_TARGETS") and contains(e.get("texts", ""), "Lotleth Giant"),
    )
    lotleth_cast_step = first_step(events, lambda e: contains(e.get("texts", ""), "Cast Lotleth Giant"))
    opponent_target_step = first_step(
        events,
        lambda e: contains(e.get("action", ""), "SELECT_TARGETS") and contains(e.get("texts", ""), "EvalBot"),
    )
    lead_step = first_step(events, lambda e: contains(e.get("texts", ""), "Cast Lead the Stampede"))
    tinder_step = first_step(events, lambda e: contains(e.get("texts", ""), "Cast Tinder Wall"))
    battlement_step = first_step(events, lambda e: contains(e.get("texts", ""), "Cast Overgrown Battlement"))

    pass_count = sum(1 for text in texts if text.strip().lower() == "pass")
    mana_count = sum(1 for text in texts if "Add {" in text or "Add one mana" in text)
    cast_count = sum(1 for text in texts if text.startswith("Cast "))
    select_count = sum(1 for action in actions if action == "SELECT_TARGETS")

    combo_score = 0
    combo_score += 3 if spy_step >= 0 else 0
    combo_score += 2 if opponent_target_step >= 0 else 0
    combo_score += 3 if dread_step >= 0 else 0
    combo_score += 2 if lotleth_target_step >= 0 else 0
    combo_score += 1 if lotleth_cast_step >= 0 else 0

    key_events: List[str] = []
    for label, step in (
        ("spy", spy_step),
        ("opponent_target", opponent_target_step),
        ("dread_return_flashback", dread_step),
        ("lotleth_target", lotleth_target_step),
        ("lotleth_cast", lotleth_cast_step),
        ("lead_the_stampede", lead_step),
        ("tinder_wall", tinder_step),
        ("overgrown_battlement", battlement_step),
    ):
        if step >= 0:
            key_events.append(f"{label}@{step}")

    return {
        "snapshot_path": row.get("snapshot_path", ""),
        "ordinal": row.get("ordinal", ""),
        "decision_number": row.get("decision_number", ""),
        "action_type": row.get("action_type", ""),
        "attempt": row.get("attempt", ""),
        "root_indices": row.get("root_indices", ""),
        "root_texts": row.get("root_texts", ""),
        "outcome": row.get("outcome", ""),
        "terminal": boolish(row, "terminal"),
        "won": boolish(row, "won"),
        "lost": boolish(row, "lost"),
        "error": row.get("error", ""),
        "decision_count": as_int(row, "decision_count"),
        "trace_event_count": len(events),
        "combo_score": combo_score,
        "has_spy": spy_step >= 0,
        "spy_step": spy_step,
        "has_opponent_target": opponent_target_step >= 0,
        "opponent_target_step": opponent_target_step,
        "has_dread_return_flashback": dread_step >= 0,
        "dread_return_step": dread_step,
        "has_lotleth_target": lotleth_target_step >= 0,
        "lotleth_target_step": lotleth_target_step,
        "has_lotleth_cast": lotleth_cast_step >= 0,
        "lotleth_cast_step": lotleth_cast_step,
        "has_lead_the_stampede": lead_step >= 0,
        "has_tinder_wall": tinder_step >= 0,
        "has_overgrown_battlement": battlement_step >= 0,
        "pass_count": pass_count,
        "mana_count": mana_count,
        "cast_count": cast_count,
        "select_target_count": select_count,
        "final_state_hash": row.get("final_state_hash", ""),
        "key_events": "|".join(key_events),
        "_artifact": row.get("_artifact", ""),
    }


def counts(rows: Iterable[dict], key: str) -> Dict[str, int]:
    counter = Counter(str(row.get(key, "")).strip() or "<blank>" for row in rows)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def summarize(run_dir: Path, prefer_merged: bool) -> dict:
    rows = read_csvs(run_dir, prefer_merged)
    compact = [trace_features(row) for row in rows]
    wins = [row for row in compact if row["outcome"] == "terminal_win" and not row["error"]]
    terminal = [
        row for row in compact
        if str(row["outcome"]).startswith("terminal_") and not row["error"]
    ]
    spy_wins = [row for row in wins if row["has_spy"]]
    full_combo_wins = [
        row for row in wins
        if row["has_spy"] and row["has_dread_return_flashback"] and row["has_lotleth_target"]
    ]
    combo_rows = [
        row for row in compact
        if row["combo_score"] > 0 and not row["error"]
    ]
    return {
        "run_dir": str(run_dir),
        "rows": len(compact),
        "terminal_rows": len(terminal),
        "terminal_rate": pct(len(terminal), len(compact)),
        "wins": len(wins),
        "win_rate": pct(len(wins), len(compact)),
        "outcome_counts": counts(compact, "outcome"),
        "root_counts": counts(compact, "root_texts"),
        "spy_rows": sum(1 for row in compact if row["has_spy"]),
        "spy_wins": len(spy_wins),
        "dread_return_rows": sum(1 for row in compact if row["has_dread_return_flashback"]),
        "lotleth_target_rows": sum(1 for row in compact if row["has_lotleth_target"]),
        "full_combo_wins": len(full_combo_wins),
        "max_combo_score": max((int(row["combo_score"]) for row in compact), default=0),
        "top_combo_rows": sorted(
            combo_rows,
            key=lambda row: (int(row["combo_score"]), bool(row["won"]), int(row["decision_count"])),
            reverse=True,
        )[:20],
        "winning_rows": sorted(
            wins,
            key=lambda row: (int(row["combo_score"]), -int(row["decision_count"])),
            reverse=True,
        )[:20],
        "compact_rows": compact,
    }


def write_compact_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [key for key in rows[0].keys() if key != "_artifact"] + ["_artifact"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_report(summary: dict) -> str:
    lines = [
        "# Terminal Line Search Summary",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- rows: `{summary['rows']}`",
        f"- terminal_rate: `{summary['terminal_rate']}`",
        f"- wins: `{summary['wins']}`",
        f"- win_rate: `{summary['win_rate']}`",
        f"- outcome_counts: `{summary['outcome_counts']}`",
        f"- spy_rows: `{summary['spy_rows']}`",
        f"- spy_wins: `{summary['spy_wins']}`",
        f"- dread_return_rows: `{summary['dread_return_rows']}`",
        f"- lotleth_target_rows: `{summary['lotleth_target_rows']}`",
        f"- full_combo_wins: `{summary['full_combo_wins']}`",
        f"- max_combo_score: `{summary['max_combo_score']}`",
        "",
        "## Winning Rows",
        "",
        "| outcome | root | combo_score | decisions | key_events | snapshot |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in summary["winning_rows"][:10]:
        lines.append(
            "| {outcome} | {root} | {score} | {decisions} | {events} | {snapshot} |".format(
                outcome=row["outcome"],
                root=str(row["root_texts"]).replace("|", "/"),
                score=row["combo_score"],
                decisions=row["decision_count"],
                events=str(row["key_events"]).replace("|", "<br>"),
                snapshot=Path(str(row["snapshot_path"])).name,
            )
        )
    lines.extend(["", "## Top Combo Rows", "", "| outcome | root | combo_score | key_events |", "| --- | --- | ---: | --- |"])
    for row in summary["top_combo_rows"][:10]:
        lines.append(
            "| {outcome} | {root} | {score} | {events} |".format(
                outcome=row["outcome"],
                root=str(row["root_texts"]).replace("|", "/"),
                score=row["combo_score"],
                events=str(row["key_events"]).replace("|", "<br>"),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    summary = summarize(run_dir, args.prefer_merged)
    out_json = Path(args.out_json) if args.out_json else run_dir / "terminal_line_summary.json"
    out_md = Path(args.out_md) if args.out_md else run_dir / "terminal_line_summary.md"
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "terminal_line_compact.csv"
    out_json.write_text(
        json.dumps({k: v for k, v in summary.items() if k != "compact_rows"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    out_md.write_text(markdown_report(summary), encoding="utf-8")
    write_compact_csv(out_csv, summary["compact_rows"])
    print(json.dumps({k: v for k, v in summary.items() if k != "compact_rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
