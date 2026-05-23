#!/usr/bin/env python3
"""Summarize value-tree and sequence-tree checkpoint mining artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


VALUE_SUMMARY = "counterfactual_value_tree_summary.csv"
VALUE_ACTIONS = "counterfactual_value_tree.csv"
SEQUENCE_SUMMARY = "counterfactual_sequence_tree_summary.csv"
SEQUENCE_ROWS = "counterfactual_sequence_tree.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MTGRL checkpoint value-tree and sequence-tree run artifacts."
    )
    parser.add_argument("run_dir", help="Run directory containing merged or sharded CSV artifacts.")
    parser.add_argument("--out-json", default="", help="Optional JSON output path.")
    parser.add_argument("--out-md", default="", help="Optional Markdown output path.")
    parser.add_argument(
        "--prefer-merged",
        action="store_true",
        help="Use only root-level merged CSVs when present instead of recursively reading shard CSVs.",
    )
    return parser.parse_args(argv)


def read_csvs(run_dir: Path, filename: str, prefer_merged: bool) -> List[dict]:
    root_file = run_dir / filename
    paths: List[Path]
    if prefer_merged and root_file.exists():
        paths = [root_file]
    else:
        paths = [
            path for path in sorted(run_dir.rglob(filename))
            if path.is_file() and path.stat().st_size > 0
        ]
        if root_file.exists() and len(paths) > 1:
            paths = [path for path in paths if path != root_file]
    rows: List[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row = dict(row)
                row["_artifact"] = str(path)
                rows.append(row)
    return rows


def as_float(row: dict, key: str, default: float = 0.0) -> float:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def boolish(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def counts(rows: Iterable[dict], key: str) -> Dict[str, int]:
    counter = Counter(str(row.get(key, "")).strip() or "<blank>" for row in rows)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def pct(numer: int, denom: int) -> float:
    return 0.0 if denom <= 0 else round(numer / denom, 6)


def top_value_rows(rows: List[dict], limit: int = 10) -> List[dict]:
    ranked = sorted(
        rows,
        key=lambda row: (
            as_float(row, "delta_win_rate"),
            as_float(row, "best_win_rate"),
            as_float(row, "source_loss_rate"),
            as_int(row, "terminal_rollouts"),
        ),
        reverse=True,
    )
    out: List[dict] = []
    for row in ranked[:limit]:
        out.append(
            {
                "snapshot_path": row.get("snapshot_path", ""),
                "ordinal": row.get("ordinal", ""),
                "decision_number": row.get("decision_number", ""),
                "classification": row.get("classification", ""),
                "source_texts": row.get("source_texts", ""),
                "best_texts": row.get("best_texts", ""),
                "source_win_rate": as_float(row, "source_win_rate"),
                "source_loss_rate": as_float(row, "source_loss_rate"),
                "source_terminal_rate": as_float(row, "source_terminal_rate"),
                "best_win_rate": as_float(row, "best_win_rate"),
                "best_loss_rate": as_float(row, "best_loss_rate"),
                "best_terminal_rate": as_float(row, "best_terminal_rate"),
                "delta_win_rate": as_float(row, "delta_win_rate"),
                "importance_score": as_float(row, "importance_score"),
                "actions_evaluated": as_int(row, "actions_evaluated"),
                "terminal_rollouts": as_int(row, "terminal_rollouts"),
            }
        )
    return out


def strict_value_candidates(rows: List[dict]) -> List[dict]:
    candidates = []
    for row in rows:
        if as_float(row, "source_loss_rate") < 1.0:
            continue
        if as_float(row, "source_terminal_rate") < 1.0:
            continue
        if as_float(row, "best_terminal_rate") < 1.0:
            continue
        if as_float(row, "best_win_rate") <= 0.0:
            continue
        if as_float(row, "delta_win_rate") <= 0.0:
            continue
        candidates.append(row)
    return top_value_rows(candidates, limit=50)


def summarize(run_dir: Path, prefer_merged: bool) -> dict:
    value_summaries = read_csvs(run_dir, VALUE_SUMMARY, prefer_merged)
    value_actions = read_csvs(run_dir, VALUE_ACTIONS, prefer_merged)
    sequence_summaries = read_csvs(run_dir, SEQUENCE_SUMMARY, prefer_merged)
    sequence_rows = read_csvs(run_dir, SEQUENCE_ROWS, prefer_merged)

    sequence_complete = sum(1 for row in sequence_rows if boolish(row, "prefix_complete"))
    sequence_terminal = sum(1 for row in sequence_rows if boolish(row, "terminal"))
    sequence_wins = sum(1 for row in sequence_rows if boolish(row, "won"))
    sequence_errors = sum(1 for row in sequence_rows if str(row.get("error", "")).strip())
    sequence_timeouts = sum(
        1 for row in sequence_rows
        if "timeout" in str(row.get("outcome", "")).lower()
        or "timeout" in str(row.get("error", "")).lower()
    )
    value_action_wins = sum(as_int(row, "win_count") for row in value_actions)
    value_action_errors = sum(as_int(row, "error_count") for row in value_actions)
    value_action_not_terminal = sum(as_int(row, "not_terminal_count") for row in value_actions)
    terminal_rollouts = sum(as_int(row, "terminal_rollouts") for row in value_summaries)
    total_rollouts = sum(as_int(row, "total_rollouts") for row in value_summaries)

    return {
        "run_dir": str(run_dir),
        "value_summary_rows": len(value_summaries),
        "value_action_rows": len(value_actions),
        "value_classification_counts": counts(value_summaries, "classification"),
        "value_action_wins": value_action_wins,
        "value_action_errors": value_action_errors,
        "value_action_not_terminal": value_action_not_terminal,
        "value_terminal_rollouts": terminal_rollouts,
        "value_total_rollouts": total_rollouts,
        "value_terminal_rate": pct(terminal_rollouts, total_rollouts),
        "strict_value_candidate_count": len(strict_value_candidates(value_summaries)),
        "top_value_rows": top_value_rows(value_summaries),
        "strict_value_candidates": strict_value_candidates(value_summaries),
        "sequence_summary_rows": len(sequence_summaries),
        "sequence_rows": len(sequence_rows),
        "sequence_classification_counts": counts(sequence_summaries, "classification"),
        "sequence_outcome_counts": counts(sequence_rows, "outcome"),
        "sequence_prefix_reason_counts": counts(sequence_rows, "prefix_reason"),
        "sequence_prefix_complete": sequence_complete,
        "sequence_prefix_completion_rate": pct(sequence_complete, len(sequence_rows)),
        "sequence_terminal": sequence_terminal,
        "sequence_terminal_rate": pct(sequence_terminal, len(sequence_rows)),
        "sequence_wins": sequence_wins,
        "sequence_win_rate": pct(sequence_wins, len(sequence_rows)),
        "sequence_errors": sequence_errors,
        "sequence_error_rate": pct(sequence_errors, len(sequence_rows)),
        "sequence_timeouts": sequence_timeouts,
        "sequence_timeout_rate": pct(sequence_timeouts, len(sequence_rows)),
    }


def markdown_report(summary: dict) -> str:
    lines = [
        "# Value Tree Run Summary",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- value_summary_rows: `{summary['value_summary_rows']}`",
        f"- value_action_rows: `{summary['value_action_rows']}`",
        f"- value_classification_counts: `{summary['value_classification_counts']}`",
        f"- value_terminal_rate: `{summary['value_terminal_rate']}`",
        f"- value_action_wins: `{summary['value_action_wins']}`",
        f"- strict_value_candidate_count: `{summary['strict_value_candidate_count']}`",
        f"- sequence_summary_rows: `{summary['sequence_summary_rows']}`",
        f"- sequence_rows: `{summary['sequence_rows']}`",
        f"- sequence_classification_counts: `{summary['sequence_classification_counts']}`",
        f"- sequence_prefix_completion_rate: `{summary['sequence_prefix_completion_rate']}`",
        f"- sequence_terminal_rate: `{summary['sequence_terminal_rate']}`",
        f"- sequence_win_rate: `{summary['sequence_win_rate']}`",
        f"- sequence_error_rate: `{summary['sequence_error_rate']}`",
        f"- sequence_timeout_rate: `{summary['sequence_timeout_rate']}`",
        "",
        "## Top Value Rows",
        "",
        "| classification | source | best | source_win | best_win | delta | terminal_rollouts |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["top_value_rows"][:10]:
        lines.append(
            "| {classification} | {source_texts} | {best_texts} | {source_win_rate:.3f} | "
            "{best_win_rate:.3f} | {delta_win_rate:.3f} | {terminal_rollouts} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Sequence Outcomes",
            "",
            "```json",
            json.dumps(summary["sequence_outcome_counts"], indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    summary = summarize(run_dir, args.prefer_merged)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out_json:
        Path(args.out_json).write_text(text, encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(markdown_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
