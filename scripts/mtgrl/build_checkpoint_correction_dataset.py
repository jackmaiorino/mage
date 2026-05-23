#!/usr/bin/env python3
"""Build a gated supervised dataset from confirmed checkpoint corrections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ADMISSION_GATE = "source_loss_alternate_win_isolated_reentry_confirmed"
DATASET_VERSION = "checkpoint_correction_v1"

CSV_FIELDS = [
    "example_id",
    "dataset_version",
    "snapshot_path",
    "ordinal",
    "decision_number",
    "action_type",
    "candidate_count",
    "source_indices",
    "source_texts",
    "target_indices",
    "target_texts",
    "candidate_hash",
    "state_hash",
    "rng_state_hash",
    "source_outcome",
    "target_outcome",
    "positive_confirmation_count",
    "selected_prob",
    "value_score",
    "turn",
    "own_life",
    "own_graveyard_count",
    "opponent_permanent_count",
    "input_run_id",
    "manifest_row_number",
]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consume confirmed_checkpoint_corrections.csv and emit a small, "
            "fail-closed correction dataset with checkpoint provenance."
        )
    )
    parser.add_argument("--manifest", required=True, help="confirmed_checkpoint_corrections.csv path.")
    parser.add_argument("--output-dir", required=True, help="Directory for dataset outputs.")
    parser.add_argument(
        "--expect-examples",
        type=int,
        default=None,
        help="Fail if the generated example count differs from this value.",
    )
    parser.add_argument(
        "--allow-missing-snapshots",
        action="store_true",
        help="Do not fail when a manifest snapshot path is not present on local disk.",
    )
    parser.add_argument(
        "--allow-missing-selection-metadata",
        action="store_true",
        help="Do not fail when selected_snapshots.csv metadata cannot be joined.",
    )
    parser.add_argument("--write-readme", action="store_true", help="Write a generated README.md.")
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


def parse_indices(value: str) -> List[int]:
    text = (value or "").strip()
    if not text:
        return []
    out: List[int] = []
    for part in text.split("|"):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def norm_path(value: str) -> str:
    return (value or "").replace("\\", "/").strip()


def read_csv(path: Path) -> List[Tuple[int, Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [(index, row) for index, row in enumerate(reader, start=2)]


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def local_snapshot_exists(snapshot_path: str, manifest_path: Path) -> bool:
    path = Path(snapshot_path)
    if path.is_file():
        return True
    if (manifest_path.parent / path).is_file():
        return True
    return False


def confirmation_outcomes_pass(row: Dict[str, str]) -> bool:
    outcomes = (row.get("positive_confirmation_outcomes") or "").strip()
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


def manifest_rejections(row: Dict[str, str], manifest_path: Path, allow_missing_snapshots: bool) -> List[str]:
    reasons: List[str] = []

    if (row.get("manifest_status") or "").strip() != "admitted":
        reasons.append("manifest_status_not_admitted")
    if (row.get("admission_gate") or "").strip() != ADMISSION_GATE:
        reasons.append("admission_gate_mismatch")

    if as_int(row, "candidate_count") < 2:
        reasons.append("candidate_count_lt_2")

    source_indices = parse_indices(row.get("selected_indices", ""))
    target_indices = parse_indices(row.get("alternate_indices", ""))
    candidate_count = as_int(row, "candidate_count")
    if not source_indices or not has_value(row, "selected_texts"):
        reasons.append("missing_source_choice")
    if not target_indices or not has_value(row, "alternate_texts"):
        reasons.append("missing_target_choice")
    if source_indices == target_indices:
        reasons.append("source_and_target_same_indices")
    if any(index < 0 or index >= candidate_count for index in source_indices + target_indices):
        reasons.append("choice_index_out_of_range")

    for field in ("snapshot_path", "candidate_hash", "state_hash", "rng_state_hash"):
        if not has_value(row, field):
            reasons.append(f"missing_{field}")

    if has_value(row, "snapshot_path") and not allow_missing_snapshots:
        if not local_snapshot_exists(row["snapshot_path"], manifest_path):
            reasons.append("snapshot_file_missing")

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
    if not as_bool(row, "alternate_terminal"):
        reasons.append("alternate_not_terminal")
    if not as_bool(row, "alternate_won"):
        reasons.append("alternate_not_won")
    if as_bool(row, "alternate_lost"):
        reasons.append("alternate_lost")

    confirmations = as_int(row, "positive_confirmation_count")
    if confirmations < 1:
        reasons.append("missing_positive_confirmations")
    if as_int(row, "positive_confirmation_pass_count") != confirmations:
        reasons.append("positive_confirmation_pass_count_mismatch")
    if not confirmation_outcomes_pass(row):
        reasons.append("positive_confirmation_outcomes_not_all_pass")

    return reasons


def load_selection_metadata(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    seen_files = set()
    for row in rows:
        input_csv = row.get("input_csv") or ""
        if not input_csv:
            continue
        selection_path = Path(input_csv).parent / "selected_snapshots.csv"
        if selection_path in seen_files or not selection_path.is_file():
            continue
        seen_files.add(selection_path)
        for _, selected in read_csv(selection_path):
            metadata[norm_path(selected.get("snapshot_path", ""))] = selected
    return metadata


def make_example_id(row: Dict[str, str]) -> str:
    seed = "|".join(
        [
            row.get("correction_id", ""),
            row.get("snapshot_path", ""),
            row.get("candidate_hash", ""),
            row.get("state_hash", ""),
            row.get("rng_state_hash", ""),
            row.get("selected_indices", ""),
            row.get("alternate_indices", ""),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def build_example(row: Dict[str, str], selected: Dict[str, str]) -> Dict[str, object]:
    source_indices = parse_indices(row["selected_indices"])
    target_indices = parse_indices(row["alternate_indices"])
    return {
        "dataset_version": DATASET_VERSION,
        "example_id": make_example_id(row),
        "task": "first_action_correction",
        "label": {
            "target_indices": target_indices,
            "target_texts": row.get("alternate_texts", ""),
            "target_outcome": "terminal_win",
        },
        "negative": {
            "source_indices": source_indices,
            "source_texts": row.get("selected_texts", ""),
            "source_outcome": "terminal_loss",
        },
        "decision": {
            "snapshot_path": row.get("snapshot_path", ""),
            "ordinal": as_int(row, "ordinal"),
            "decision_number": as_int(row, "decision_number"),
            "action_type": row.get("action_type", ""),
            "candidate_count": as_int(row, "candidate_count"),
            "candidate_hash": row.get("candidate_hash", ""),
            "state_hash": row.get("state_hash", ""),
            "rng_state_hash": row.get("rng_state_hash", ""),
        },
        "proof": {
            "admission_gate": row.get("admission_gate", ""),
            "source_reentry_a_matched": as_bool(row, "source_reentry_a_matched"),
            "source_reentry_b_matched": as_bool(row, "source_reentry_b_matched"),
            "reentry_a_candidate_hash": row.get("reentry_a_candidate_hash", ""),
            "reentry_b_candidate_hash": row.get("reentry_b_candidate_hash", ""),
            "reentry_a_state_hash": row.get("reentry_a_state_hash", ""),
            "reentry_b_state_hash": row.get("reentry_b_state_hash", ""),
            "positive_confirmation_count": as_int(row, "positive_confirmation_count"),
            "positive_confirmation_pass_count": as_int(row, "positive_confirmation_pass_count"),
            "positive_confirmation_outcomes": row.get("positive_confirmation_outcomes", ""),
            "alternate_outcomes": row.get("alternate_outcomes", ""),
        },
        "ranking_context": {
            "selected_prob": selected.get("selected_prob", ""),
            "value_score": selected.get("value_score", ""),
            "rank": selected.get("rank", ""),
            "selection_score": selected.get("score", ""),
            "turn": selected.get("turn", ""),
            "own_life": selected.get("own_life", ""),
            "own_graveyard_count": selected.get("own_graveyard_count", ""),
            "opponent_permanent_count": selected.get("opponent_permanent_count", ""),
            "score_reasons": selected.get("score_reasons", ""),
        },
        "provenance": {
            "manifest_correction_id": row.get("correction_id", ""),
            "manifest_input_run_id": row.get("input_run_id", ""),
            "manifest_input_csv": row.get("input_csv", ""),
            "manifest_row_number": row.get("row_number", ""),
        },
    }


def csv_row(example: Dict[str, object]) -> Dict[str, str]:
    decision = example["decision"]  # type: ignore[index]
    label = example["label"]  # type: ignore[index]
    negative = example["negative"]  # type: ignore[index]
    proof = example["proof"]  # type: ignore[index]
    ranking = example["ranking_context"]  # type: ignore[index]
    provenance = example["provenance"]  # type: ignore[index]
    return {
        "example_id": str(example["example_id"]),
        "dataset_version": str(example["dataset_version"]),
        "snapshot_path": str(decision["snapshot_path"]),
        "ordinal": str(decision["ordinal"]),
        "decision_number": str(decision["decision_number"]),
        "action_type": str(decision["action_type"]),
        "candidate_count": str(decision["candidate_count"]),
        "source_indices": "|".join(str(value) for value in negative["source_indices"]),
        "source_texts": str(negative["source_texts"]),
        "target_indices": "|".join(str(value) for value in label["target_indices"]),
        "target_texts": str(label["target_texts"]),
        "candidate_hash": str(decision["candidate_hash"]),
        "state_hash": str(decision["state_hash"]),
        "rng_state_hash": str(decision["rng_state_hash"]),
        "source_outcome": str(negative["source_outcome"]),
        "target_outcome": str(label["target_outcome"]),
        "positive_confirmation_count": str(proof["positive_confirmation_count"]),
        "selected_prob": str(ranking["selected_prob"]),
        "value_score": str(ranking["value_score"]),
        "turn": str(ranking["turn"]),
        "own_life": str(ranking["own_life"]),
        "own_graveyard_count": str(ranking["own_graveyard_count"]),
        "opponent_permanent_count": str(ranking["opponent_permanent_count"]),
        "input_run_id": str(provenance["manifest_input_run_id"]),
        "manifest_row_number": str(provenance["manifest_row_number"]),
    }


def write_readme(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Checkpoint Correction Dataset",
        "",
        "Generated by `scripts/mtgrl/build_checkpoint_correction_dataset.py`.",
        "",
        f"- Dataset version: `{DATASET_VERSION}`",
        f"- Admission gate: `{ADMISSION_GATE}`",
        f"- Examples: `{summary['examples']}`",
        f"- Manifest rows: `{summary['manifest_rows']}`",
        "",
        "This is a supervised first-action correction dataset. Each row labels the",
        "terminal-winning sibling action against the accepted-policy source action that",
        "lost from the same serialized checkpoint.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    rows = read_csv(manifest_path)
    manifest_rows = [row for _, row in rows]
    if not rows:
        print("manifest has zero rows", file=sys.stderr)
        return 2

    selection_metadata = load_selection_metadata(manifest_rows)
    examples: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    error_reason_counts: Counter[str] = Counter()
    seen_ids = set()

    for row_number, row in rows:
        reasons = manifest_rejections(row, manifest_path, args.allow_missing_snapshots)
        selected = selection_metadata.get(norm_path(row.get("snapshot_path", "")), {})
        if not selected and not args.allow_missing_selection_metadata:
            reasons.append("selection_metadata_missing")
        if reasons:
            error_reason_counts.update(reasons)
            errors.append(
                {
                    "row_number": row_number,
                    "correction_id": row.get("correction_id", ""),
                    "snapshot_path": row.get("snapshot_path", ""),
                    "reasons": reasons,
                }
            )
            continue
        example = build_example(row, selected)
        example_id = example["example_id"]
        if example_id in seen_ids:
            errors.append(
                {
                    "row_number": row_number,
                    "correction_id": row.get("correction_id", ""),
                    "snapshot_path": row.get("snapshot_path", ""),
                    "reasons": ["duplicate_example_id"],
                }
            )
            error_reason_counts.update(["duplicate_example_id"])
            continue
        seen_ids.add(example_id)
        examples.append(example)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_version": DATASET_VERSION,
        "admission_gate": ADMISSION_GATE,
        "manifest": str(manifest_path),
        "manifest_rows": len(rows),
        "examples": len(examples),
        "errors": len(errors),
        "error_reason_counts": dict(sorted(error_reason_counts.items())),
        "error_rows": errors,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "checkpoint_correction_examples.jsonl", examples)
    write_csv(output_dir / "checkpoint_correction_examples.csv", CSV_FIELDS, [csv_row(example) for example in examples])
    if args.write_readme:
        write_readme(output_dir / "README.md", summary)

    stdout_summary = dict(summary)
    stdout_summary.pop("error_rows", None)
    print(json.dumps(stdout_summary, indent=2, sort_keys=True))

    if errors:
        return 3
    if not examples:
        return 4
    if args.expect_examples is not None and len(examples) != args.expect_examples:
        print(
            f"expected {args.expect_examples} examples, got {len(examples)}",
            file=sys.stderr,
        )
        return 5
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return build(parse_args(argv if argv is not None else sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
