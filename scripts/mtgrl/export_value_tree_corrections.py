#!/usr/bin/env python3
"""Export weighted correction evidence from counterfactual value-tree runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SUMMARY_FILE = "counterfactual_value_tree_summary.csv"
ACTION_FILE = "counterfactual_value_tree.csv"
ADMISSION_GATE = "value_tree_source_loss_best_win_delta"

OUTPUT_FIELDS = [
    "correction_id",
    "manifest_status",
    "admission_gate",
    "input_run_id",
    "summary_csv",
    "action_csv",
    "row_number",
    "snapshot_path",
    "ordinal",
    "decision_number",
    "action_type",
    "candidate_count",
    "candidate_hash",
    "state_hash",
    "rng_state_hash",
    "source_indices",
    "source_texts",
    "target_indices",
    "target_texts",
    "source_win_rate",
    "source_loss_rate",
    "source_terminal_rate",
    "target_win_rate",
    "target_loss_rate",
    "target_terminal_rate",
    "delta_win_rate",
    "importance_score",
    "label_weight",
    "actions_evaluated",
    "total_rollouts",
    "terminal_rollouts",
    "source_outcomes",
    "target_outcomes",
    "classification",
    "reentry_a_matched",
    "reentry_b_matched",
    "reentry_a_candidate_hash",
    "reentry_b_candidate_hash",
    "reentry_a_state_hash",
    "reentry_b_state_hash",
    "reentry_a_reason",
    "reentry_b_reason",
]

REJECT_FIELDS = [
    "input_run_id",
    "summary_csv",
    "row_number",
    "snapshot_path",
    "classification",
    "source_texts",
    "best_texts",
    "source_loss_rate",
    "best_win_rate",
    "delta_win_rate",
    "importance_score",
    "rejection_reasons",
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read counterfactual_value_tree_summary.csv/action CSV pairs and export "
            "weighted first-action correction evidence."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Value-tree run directories or summary CSV files.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported correction manifests.")
    parser.add_argument("--min-delta", type=float, default=0.50)
    parser.add_argument("--min-source-loss-rate", type=float, default=1.0)
    parser.add_argument("--min-target-win-rate", type=float, default=0.50)
    parser.add_argument("--min-source-terminal-rate", type=float, default=1.0)
    parser.add_argument("--min-target-terminal-rate", type=float, default=1.0)
    parser.add_argument(
        "--allowed-classifications",
        default="dominant_correction,strong_correction",
        help="Comma-separated value-tree classifications admitted by this export.",
    )
    parser.add_argument("--expect-admitted", type=int, default=None)
    parser.add_argument("--write-readme", action="store_true")
    return parser.parse_args(argv)


def read_csv(path: Path) -> List[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [(index, row) for index, row in enumerate(reader, start=2)]


def resolve_inputs(inputs: Iterable[str]) -> List[Tuple[Path, Path]]:
    out: List[Tuple[Path, Path]] = []
    for raw in inputs:
        path = Path(raw)
        summary = path / SUMMARY_FILE if path.is_dir() else path
        action = summary.parent / ACTION_FILE
        if not summary.is_file():
            raise FileNotFoundError(f"missing value-tree summary CSV: {summary}")
        if not action.is_file():
            raise FileNotFoundError(f"missing value-tree action CSV: {action}")
        out.append((summary, action))
    return out


def parse_set(raw: str) -> set:
    return {item.strip() for item in (raw or "").split(",") if item.strip()}


def as_float(row: Dict[str, str], field: str) -> float:
    value = (row.get(field) or "").strip()
    if not value:
        return 0.0
    return float(value)


def as_bool(row: Dict[str, str], field: str) -> bool:
    return (row.get(field) or "").strip().lower() == "true"


def action_key(snapshot_path: str, indices: str) -> Tuple[str, str]:
    return (snapshot_path.replace("\\", "/"), (indices or "").strip())


def load_action_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = {}
    for _, row in read_csv(path):
        rows[action_key(row.get("snapshot_path", ""), row.get("action_indices", ""))] = row
    return rows


def rejection_reasons(
        row: Dict[str, str],
        source_action: Dict[str, str],
        target_action: Dict[str, str],
        args: argparse.Namespace,
        allowed: set,
) -> List[str]:
    reasons: List[str] = []
    if (row.get("classification") or "").strip() not in allowed:
        reasons.append("classification_not_allowed")
    if not as_bool(row, "reentry_a_matched"):
        reasons.append("reentry_a_not_matched")
    if not as_bool(row, "reentry_b_matched"):
        reasons.append("reentry_b_not_matched")
    if (row.get("candidate_hash") or "").strip() != (row.get("reentry_a_candidate_hash") or "").strip():
        reasons.append("reentry_a_candidate_hash_mismatch")
    if (row.get("candidate_hash") or "").strip() != (row.get("reentry_b_candidate_hash") or "").strip():
        reasons.append("reentry_b_candidate_hash_mismatch")
    if (row.get("state_hash") or "").strip() != (row.get("reentry_a_state_hash") or "").strip():
        reasons.append("reentry_a_state_hash_mismatch")
    if (row.get("state_hash") or "").strip() != (row.get("reentry_b_state_hash") or "").strip():
        reasons.append("reentry_b_state_hash_mismatch")
    if as_float(row, "source_loss_rate") < args.min_source_loss_rate:
        reasons.append("source_loss_rate_below_threshold")
    if as_float(row, "source_terminal_rate") < args.min_source_terminal_rate:
        reasons.append("source_terminal_rate_below_threshold")
    if as_float(row, "best_win_rate") < args.min_target_win_rate:
        reasons.append("target_win_rate_below_threshold")
    if as_float(row, "best_terminal_rate") < args.min_target_terminal_rate:
        reasons.append("target_terminal_rate_below_threshold")
    if as_float(row, "delta_win_rate") < args.min_delta:
        reasons.append("delta_below_threshold")
    if as_float(row, "importance_score") <= 0.0:
        reasons.append("importance_not_positive")
    if not source_action:
        reasons.append("source_action_row_missing")
    if not target_action:
        reasons.append("target_action_row_missing")
    return reasons


def correction_id(row: Dict[str, str]) -> str:
    seed = "|".join(
        [
            row.get("snapshot_path", ""),
            row.get("candidate_hash", ""),
            row.get("state_hash", ""),
            row.get("rng_state_hash", ""),
            row.get("source_indices", ""),
            row.get("best_indices", ""),
            row.get("delta_win_rate", ""),
            row.get("importance_score", ""),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def admitted_row(
        row: Dict[str, str],
        source_action: Dict[str, str],
        target_action: Dict[str, str],
        summary_path: Path,
        action_path: Path,
        row_number: int,
) -> Dict[str, str]:
    label_weight = max(0.0, min(1.0, as_float(row, "importance_score")))
    result = {
        "correction_id": correction_id(row),
        "manifest_status": "admitted",
        "admission_gate": ADMISSION_GATE,
        "input_run_id": summary_path.parent.name,
        "summary_csv": str(summary_path),
        "action_csv": str(action_path),
        "row_number": str(row_number),
        "snapshot_path": row.get("snapshot_path", ""),
        "ordinal": row.get("ordinal", ""),
        "decision_number": row.get("decision_number", ""),
        "action_type": row.get("action_type", ""),
        "candidate_count": row.get("candidate_count", ""),
        "candidate_hash": row.get("candidate_hash", ""),
        "state_hash": row.get("state_hash", ""),
        "rng_state_hash": row.get("rng_state_hash", ""),
        "source_indices": row.get("source_indices", ""),
        "source_texts": row.get("source_texts", ""),
        "target_indices": row.get("best_indices", ""),
        "target_texts": row.get("best_texts", ""),
        "source_win_rate": row.get("source_win_rate", ""),
        "source_loss_rate": row.get("source_loss_rate", ""),
        "source_terminal_rate": row.get("source_terminal_rate", ""),
        "target_win_rate": row.get("best_win_rate", ""),
        "target_loss_rate": row.get("best_loss_rate", ""),
        "target_terminal_rate": row.get("best_terminal_rate", ""),
        "delta_win_rate": row.get("delta_win_rate", ""),
        "importance_score": row.get("importance_score", ""),
        "label_weight": f"{label_weight:.6f}",
        "actions_evaluated": row.get("actions_evaluated", ""),
        "total_rollouts": row.get("total_rollouts", ""),
        "terminal_rollouts": row.get("terminal_rollouts", ""),
        "source_outcomes": source_action.get("outcomes", ""),
        "target_outcomes": target_action.get("outcomes", ""),
        "classification": row.get("classification", ""),
        "reentry_a_matched": row.get("reentry_a_matched", ""),
        "reentry_b_matched": row.get("reentry_b_matched", ""),
        "reentry_a_candidate_hash": row.get("reentry_a_candidate_hash", ""),
        "reentry_b_candidate_hash": row.get("reentry_b_candidate_hash", ""),
        "reentry_a_state_hash": row.get("reentry_a_state_hash", ""),
        "reentry_b_state_hash": row.get("reentry_b_state_hash", ""),
        "reentry_a_reason": row.get("reentry_a_reason", ""),
        "reentry_b_reason": row.get("reentry_b_reason", ""),
    }
    return {field: result.get(field, "") for field in OUTPUT_FIELDS}


def rejected_row(row: Dict[str, str], summary_path: Path, row_number: int, reasons: Sequence[str]) -> Dict[str, str]:
    return {
        "input_run_id": summary_path.parent.name,
        "summary_csv": str(summary_path),
        "row_number": str(row_number),
        "snapshot_path": row.get("snapshot_path", ""),
        "classification": row.get("classification", ""),
        "source_texts": row.get("source_texts", ""),
        "best_texts": row.get("best_texts", ""),
        "source_loss_rate": row.get("source_loss_rate", ""),
        "best_win_rate": row.get("best_win_rate", ""),
        "delta_win_rate": row.get("delta_win_rate", ""),
        "importance_score": row.get("importance_score", ""),
        "rejection_reasons": "|".join(reasons),
    }


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_readme(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Value-Tree Corrections",
        "",
        "Generated by `scripts/mtgrl/export_value_tree_corrections.py`.",
        "",
        f"- Admission gate: `{ADMISSION_GATE}`",
        f"- Input rows: `{summary['input_rows']}`",
        f"- Admitted rows: `{summary['admitted_rows']}`",
        f"- Rejected rows: `{summary['rejected_rows']}`",
        f"- Min delta: `{summary['min_delta']}`",
        f"- Min target win rate: `{summary['min_target_win_rate']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export(args: argparse.Namespace) -> int:
    allowed = parse_set(args.allowed_classifications)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    admitted: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []
    classification_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    input_rows = 0
    input_files: List[str] = []

    for summary_path, action_path in resolve_inputs(args.inputs):
        input_files.append(str(summary_path))
        actions = load_action_rows(action_path)
        for row_number, row in read_csv(summary_path):
            input_rows += 1
            classification_counts[(row.get("classification") or "").strip() or "<missing>"] += 1
            source_action = actions.get(action_key(row.get("snapshot_path", ""), row.get("source_indices", "")), {})
            target_action = actions.get(action_key(row.get("snapshot_path", ""), row.get("best_indices", "")), {})
            reasons = rejection_reasons(row, source_action, target_action, args, allowed)
            if reasons:
                rejection_counts.update(reasons)
                rejected.append(rejected_row(row, summary_path, row_number, reasons))
            else:
                admitted.append(admitted_row(row, source_action, target_action, summary_path, action_path, row_number))

    write_csv(output_dir / "value_tree_corrections.csv", OUTPUT_FIELDS, admitted)
    write_jsonl(output_dir / "value_tree_corrections.jsonl", admitted)
    write_csv(output_dir / "rejected_value_tree_corrections.csv", REJECT_FIELDS, rejected)

    summary = {
        "admission_gate": ADMISSION_GATE,
        "allowed_classifications": sorted(allowed),
        "input_files": input_files,
        "input_rows": input_rows,
        "admitted_rows": len(admitted),
        "rejected_rows": len(rejected),
        "classification_counts": dict(sorted(classification_counts.items())),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "min_delta": args.min_delta,
        "min_source_loss_rate": args.min_source_loss_rate,
        "min_target_win_rate": args.min_target_win_rate,
        "min_source_terminal_rate": args.min_source_terminal_rate,
        "min_target_terminal_rate": args.min_target_terminal_rate,
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.write_readme:
        write_readme(output_dir / "README.md", summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.expect_admitted is not None and len(admitted) != args.expect_admitted:
        print(f"expected {args.expect_admitted} admitted rows, got {len(admitted)}", file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return export(parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
