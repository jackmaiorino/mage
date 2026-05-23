#!/usr/bin/env python3
"""Export fail-closed checkpoint-derived correction evidence manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROBE_FILE_NAME = "live_checkpoint_branch_probe.csv"
ADMISSION_GATE = "source_loss_alternate_win_isolated_reentry_confirmed"

OUTPUT_FIELDS = [
    "correction_id",
    "manifest_status",
    "admission_gate",
    "input_run_id",
    "input_csv",
    "row_number",
    "snapshot_path",
    "ordinal",
    "decision_number",
    "action_type",
    "candidate_count",
    "selected_indices",
    "selected_texts",
    "alternate_indices",
    "alternate_texts",
    "candidate_hash",
    "state_hash",
    "rng_state_hash",
    "source_terminal",
    "source_won",
    "source_lost",
    "alternate_terminal",
    "alternate_won",
    "alternate_lost",
    "alternate_attempt_count",
    "alternate_terminal_count",
    "alternate_win_count",
    "alternate_outcomes",
    "positive_confirmation_count",
    "positive_confirmation_pass_count",
    "positive_confirmation_outcomes",
    "source_reentry_a_matched",
    "source_reentry_b_matched",
    "reentry_a_candidate_hash",
    "reentry_b_candidate_hash",
    "reentry_a_state_hash",
    "reentry_b_state_hash",
    "reentry_a_reason",
    "reentry_b_reason",
]

REJECT_FIELDS = [
    "input_run_id",
    "input_csv",
    "row_number",
    "classification",
    "snapshot_path",
    "ordinal",
    "decision_number",
    "action_type",
    "candidate_hash",
    "state_hash",
    "rng_state_hash",
    "rejection_reasons",
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read live checkpoint branch probe CSV files or run directories and export "
            "only isolated, deterministic correction evidence rows."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Probe CSV files or directories containing "
            f"{PROBE_FILE_NAME}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifest CSV/JSONL and audit files will be written.",
    )
    parser.add_argument(
        "--min-confirmations",
        type=int,
        default=1,
        help="Minimum positive_confirmation_count required for admission.",
    )
    parser.add_argument(
        "--expect-admitted",
        type=int,
        default=None,
        help="Fail if the admitted row count differs from this value.",
    )
    parser.add_argument(
        "--write-readme",
        action="store_true",
        help="Write a short README.md beside the manifest artifacts.",
    )
    return parser.parse_args(argv)


def as_bool(row: Dict[str, str], field: str) -> bool:
    return (row.get(field) or "").strip().lower() == "true"


def as_int(row: Dict[str, str], field: str) -> int:
    value = (row.get(field) or "").strip()
    if not value:
        return 0
    return int(value)


def has_value(row: Dict[str, str], field: str) -> bool:
    return bool((row.get(field) or "").strip())


def resolve_probe_paths(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            path = path / PROBE_FILE_NAME
        if not path.is_file():
            raise FileNotFoundError(f"missing probe CSV: {path}")
        paths.append(path)
    return paths


def read_probe_rows(path: Path) -> List[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [(index, row) for index, row in enumerate(reader, start=2)]


def confirmation_outcomes_pass(row: Dict[str, str]) -> bool:
    outcomes = (row.get("positive_confirmation_outcomes") or "").strip()
    if not outcomes:
        return False
    parts = [part.strip() for part in outcomes.split("|") if part.strip()]
    if not parts:
        return False
    expected = as_int(row, "positive_confirmation_count")
    if expected and len(parts) < expected:
        return False
    return all(
        "source=terminal_loss" in part
        and "alternate=terminal_win" in part
        and "pass=true" in part
        for part in parts
    )


def admission_rejections(row: Dict[str, str], min_confirmations: int) -> List[str]:
    reasons: List[str] = []

    if (row.get("classification") or "").strip() != "clean_positive":
        reasons.append("classification_not_clean_positive")

    if not has_value(row, "snapshot_path"):
        reasons.append("missing_snapshot_path")

    for field in ("candidate_hash", "state_hash", "rng_state_hash"):
        if not has_value(row, field):
            reasons.append(f"missing_{field}")

    if as_int(row, "candidate_count") < 2:
        reasons.append("candidate_count_lt_2")

    if not has_value(row, "selected_indices") or not has_value(row, "selected_texts"):
        reasons.append("missing_source_choice")

    if not has_value(row, "alternate_indices") or not has_value(row, "alternate_texts"):
        reasons.append("missing_alternate_choice")

    if not as_bool(row, "source_reentry_a_matched"):
        reasons.append("source_reentry_a_not_matched")
    if not as_bool(row, "source_reentry_b_matched"):
        reasons.append("source_reentry_b_not_matched")

    candidate_hash = (row.get("candidate_hash") or "").strip()
    state_hash = (row.get("state_hash") or "").strip()
    if (row.get("reentry_a_candidate_hash") or "").strip() != candidate_hash:
        reasons.append("reentry_a_candidate_hash_mismatch")
    if (row.get("reentry_b_candidate_hash") or "").strip() != candidate_hash:
        reasons.append("reentry_b_candidate_hash_mismatch")
    if (row.get("reentry_a_state_hash") or "").strip() != state_hash:
        reasons.append("reentry_a_state_hash_mismatch")
    if (row.get("reentry_b_state_hash") or "").strip() != state_hash:
        reasons.append("reentry_b_state_hash_mismatch")

    if not as_bool(row, "source_terminal"):
        reasons.append("source_not_terminal")
    if as_bool(row, "source_won"):
        reasons.append("source_won")
    if not as_bool(row, "source_lost"):
        reasons.append("source_not_lost")
    if has_value(row, "source_error"):
        reasons.append("source_error")

    if not as_bool(row, "alternate_terminal"):
        reasons.append("alternate_not_terminal")
    if not as_bool(row, "alternate_won"):
        reasons.append("alternate_not_won")
    if as_bool(row, "alternate_lost"):
        reasons.append("alternate_lost")
    if has_value(row, "alternate_error"):
        reasons.append("alternate_error")

    confirmations = as_int(row, "positive_confirmation_count")
    confirmation_passes = as_int(row, "positive_confirmation_pass_count")
    if confirmations < min_confirmations:
        reasons.append("insufficient_positive_confirmations")
    if confirmation_passes != confirmations:
        reasons.append("positive_confirmation_pass_count_mismatch")
    if not confirmation_outcomes_pass(row):
        reasons.append("positive_confirmation_outcomes_not_all_pass")

    return reasons


def correction_id(row: Dict[str, str], run_id: str) -> str:
    seed = "|".join(
        [
            run_id,
            row.get("snapshot_path", ""),
            row.get("candidate_hash", ""),
            row.get("state_hash", ""),
            row.get("rng_state_hash", ""),
            row.get("selected_indices", ""),
            row.get("alternate_indices", ""),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def output_row(row: Dict[str, str], path: Path, row_number: int) -> Dict[str, str]:
    run_id = path.parent.name
    result: Dict[str, str] = {
        field: row.get(field, "")
        for field in OUTPUT_FIELDS
        if field not in {"correction_id", "manifest_status", "admission_gate", "input_run_id", "input_csv", "row_number"}
    }
    result.update(
        {
            "correction_id": correction_id(row, run_id),
            "manifest_status": "admitted",
            "admission_gate": ADMISSION_GATE,
            "input_run_id": run_id,
            "input_csv": str(path),
            "row_number": str(row_number),
        }
    )
    return {field: result.get(field, "") for field in OUTPUT_FIELDS}


def reject_row(row: Dict[str, str], path: Path, row_number: int, reasons: Sequence[str]) -> Dict[str, str]:
    return {
        "input_run_id": path.parent.name,
        "input_csv": str(path),
        "row_number": str(row_number),
        "classification": row.get("classification", ""),
        "snapshot_path": row.get("snapshot_path", ""),
        "ordinal": row.get("ordinal", ""),
        "decision_number": row.get("decision_number", ""),
        "action_type": row.get("action_type", ""),
        "candidate_hash": row.get("candidate_hash", ""),
        "state_hash": row.get("state_hash", ""),
        "rng_state_hash": row.get("rng_state_hash", ""),
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


def write_readme(path: Path, summary: Dict[str, object], admitted: Sequence[Dict[str, str]]) -> None:
    lines = [
        "# Confirmed Checkpoint Corrections",
        "",
        "This directory is generated by `scripts/mtgrl/export_checkpoint_corrections.py`.",
        "",
        f"- Admission gate: `{ADMISSION_GATE}`",
        f"- Input rows: `{summary['input_rows']}`",
        f"- Admitted rows: `{summary['admitted_rows']}`",
        f"- Rejected rows: `{summary['rejected_rows']}`",
        "",
        "Rows are admitted only when source reentry is deterministic, source continuation is a terminal loss,",
        "the selected sibling continuation is a terminal win, and isolated positive confirmations all pass.",
        "",
        "## Admitted Rows",
        "",
    ]
    if admitted:
        lines.extend(
            [
                "| Run | Ordinal | Decision | Source | Alternate | Candidate Hash | State Hash | RNG Hash |",
                "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for row in admitted:
            lines.append(
                "| {input_run_id} | {ordinal} | {decision_number} | {selected_texts} | {alternate_texts} | "
                "`{candidate_hash}` | `{state_hash}` | `{rng_state_hash}` |".format(**row)
            )
    else:
        lines.append("No rows admitted.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    admitted: List[Dict[str, str]] = []
    rejected: List[Dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    input_rows = 0
    input_files: List[str] = []

    for path in resolve_probe_paths(args.inputs):
        input_files.append(str(path))
        for row_number, row in read_probe_rows(path):
            input_rows += 1
            classification_counts[(row.get("classification") or "").strip() or "<missing>"] += 1
            reasons = admission_rejections(row, args.min_confirmations)
            if reasons:
                rejected.append(reject_row(row, path, row_number, reasons))
                reason_counts.update(reasons)
            else:
                admitted.append(output_row(row, path, row_number))

    write_csv(output_dir / "confirmed_checkpoint_corrections.csv", OUTPUT_FIELDS, admitted)
    write_jsonl(output_dir / "confirmed_checkpoint_corrections.jsonl", admitted)
    write_csv(output_dir / "rejected_checkpoint_corrections.csv", REJECT_FIELDS, rejected)

    summary = {
        "admission_gate": ADMISSION_GATE,
        "min_confirmations": args.min_confirmations,
        "input_files": input_files,
        "input_rows": input_rows,
        "admitted_rows": len(admitted),
        "rejected_rows": len(rejected),
        "classification_counts": dict(sorted(classification_counts.items())),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.write_readme:
        write_readme(output_dir / "README.md", summary, admitted)

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.expect_admitted is not None and len(admitted) != args.expect_admitted:
        print(
            f"expected {args.expect_admitted} admitted rows, got {len(admitted)}",
            file=sys.stderr,
        )
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return export(parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
