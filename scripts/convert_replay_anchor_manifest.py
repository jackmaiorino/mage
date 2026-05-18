#!/usr/bin/env python3
"""Convert replay-ready anchor manifests into ActionCounterfactual replay CSV.

The input is the `replay_anchor_manifest.csv` produced from compact Affinity
failure logs. The output schema is intentionally compatible with
ActionCounterfactualTrainer's `--replay-file` loader while preserving enough
anchor context to inspect the original decision later.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MANIFEST_CSV = Path(
    "local-training/local_pbt/corpora/20260518_affinity_replay_metadata_smoke_g4/replay_anchor_manifest.csv"
)

ACTION_TYPES = {
    "ACTIVATE_ABILITY_OR_SPELL",
    "SELECT_TARGETS",
    "SELECT_CARD",
    "CHOOSE_USE",
    "CHOOSE_MODE",
    "ANNOUNCE_X",
}

JAVA_REQUIRED_COLUMNS = (
    "scenario",
    "agent_deck",
    "opp_deck",
    "seed",
    "ordinal",
    "action_type",
    "chosen_indices",
    "chosen_texts",
    "best_idx",
    "best_text",
    "first_priority_hand",
    "first_mulligan_hand",
)

OUTPUT_COLUMNS = (
    "scenario",
    "agent_deck",
    "opp_deck",
    "seed",
    "ordinal",
    "action_type",
    "chosen_indices",
    "chosen_texts",
    "best_idx",
    "best_text",
    "first_priority_hand",
    "first_mulligan_hand",
    "anchor_id",
    "anchor_rank",
    "anchor_category",
    "target_action_category",
    "target_action_type",
    "selected_class",
    "selected_text",
    "selected_prob",
    "top_best_text",
    "top_best_prob",
    "candidate_indices",
    "candidate_probs",
    "candidate_texts",
    "source_game_path",
    "source_log_line",
    "decision_line",
    "selected_line",
    "top_line",
    "state_line",
    "value_line",
    "replay_metadata_line",
    "agent_opening_hand",
    "game_started",
    "matchup",
    "mode",
)

SELECTED_RE = re.compile(r"SELECTED\[(\d+)]\s+p=([0-9.eE+-]+)\s+value=[^:]*:\s*(.*)$")
TOP_PART_RE = re.compile(r"^\s*(\*)?\[(\d+)]\s+([0-9.eE+-]+)\s+(.*)$")


class ConversionError(ValueError):
    pass


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def split_cards(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def join_replay_texts(values: Iterable[str]) -> str:
    return "||".join((value or "").replace("||", " ").strip() for value in values)


def parse_selected_line(line: str) -> Tuple[Optional[int], str, str]:
    match = SELECTED_RE.search(line or "")
    if not match:
        return None, "", ""
    return int(match.group(1)), match.group(2), match.group(3).strip()


def parse_top_line(line: str) -> Tuple[List[int], List[str], List[str], Optional[int]]:
    indices: List[int] = []
    probs: List[str] = []
    texts: List[str] = []
    starred: Optional[int] = None
    for raw_part in (line or "").split("|"):
        part = raw_part.strip()
        if part.startswith("TOP:"):
            continue
        match = TOP_PART_RE.match(part)
        if not match:
            continue
        if match.group(1):
            starred = int(match.group(2))
        indices.append(int(match.group(2)))
        probs.append(match.group(3))
        texts.append(match.group(4).strip())
    return indices, probs, texts, starred


def infer_action_type(row: Dict[str, str]) -> str:
    decision_line = (row.get("decision_line") or "").upper()
    phase = (row.get("phase") or "").upper()
    selected_class = (row.get("selected_class") or "").strip()
    combined = decision_line + " " + phase
    if "TARGET_PICK" in combined:
        return "SELECT_TARGETS"
    if "CHOOSE_USE" in combined:
        return "CHOOSE_USE"
    if "CHOOSE_MODE" in combined:
        return "CHOOSE_MODE"
    if "ANNOUNCE_X" in combined:
        return "ANNOUNCE_X"
    if selected_class == "target_or_choice":
        return "SELECT_TARGETS"
    return "ACTIVATE_ABILITY_OR_SPELL"


def row_value(row: Dict[str, str], name: str) -> str:
    return (row.get(name) or "").strip()


def convert_row(row: Dict[str, str], strict: bool, stack_agent_opening_hand: bool) -> Dict[str, str]:
    anchor_id = row_value(row, "anchor_id")
    if strict and not truthy(row.get("replay_ready")):
        raise ConversionError(f"{anchor_id}: anchor is not replay_ready")

    selected_idx, selected_prob, selected_text = parse_selected_line(row_value(row, "selected_line"))
    top_indices, top_probs, top_texts, starred_idx = parse_top_line(row_value(row, "top_line"))
    if selected_idx is None:
        selected_idx = starred_idx
    if not selected_text:
        selected_text = row_value(row, "selected")
    if not selected_prob:
        selected_prob = row_value(row, "selected_prob")
    if selected_idx is None:
        raise ConversionError(f"{anchor_id}: could not parse selected index from selected_line/top_line")

    action_type = infer_action_type(row)
    if action_type not in ACTION_TYPES:
        raise ConversionError(f"{anchor_id}: unsupported action type {action_type}")

    required = {
        "scenario": row_value(row, "replay_scenario"),
        "agent_deck": row_value(row, "replay_agent_deck"),
        "opp_deck": row_value(row, "replay_opp_deck"),
        "seed": row_value(row, "replay_seed"),
        "ordinal": row_value(row, "decision_number"),
        "action_type": action_type,
        "chosen_indices": str(selected_idx),
        "chosen_texts": join_replay_texts([selected_text]),
        "best_idx": str(selected_idx),
        "best_text": selected_text,
        "first_priority_hand": (
            join_replay_texts(split_cards(row_value(row, "agent_opening_hand")))
            if stack_agent_opening_hand
            else ""
        ),
        "first_mulligan_hand": "",
    }
    missing = [name for name, value in required.items() if name not in {"first_priority_hand", "first_mulligan_hand"} and not value]
    if missing:
        raise ConversionError(f"{anchor_id}: missing required replay fields: {', '.join(missing)}")

    out = {
        **required,
        "anchor_id": anchor_id,
        "anchor_rank": row_value(row, "rank"),
        "anchor_category": row_value(row, "category"),
        "target_action_category": row_value(row, "category"),
        "target_action_type": action_type,
        "selected_class": row_value(row, "selected_class"),
        "selected_text": selected_text,
        "selected_prob": selected_prob,
        "top_best_text": row_value(row, "top_best_text"),
        "top_best_prob": row_value(row, "top_best_prob"),
        "candidate_indices": ";".join(str(i) for i in top_indices),
        "candidate_probs": ";".join(top_probs),
        "candidate_texts": join_replay_texts(top_texts),
        "source_game_path": row_value(row, "game_path"),
        "source_log_line": row_value(row, "log_line"),
        "decision_line": row_value(row, "decision_line"),
        "selected_line": row_value(row, "selected_line"),
        "top_line": row_value(row, "top_line"),
        "state_line": row_value(row, "state_line"),
        "value_line": row_value(row, "value_line"),
        "replay_metadata_line": row_value(row, "replay_metadata_line"),
        "agent_opening_hand": row_value(row, "agent_opening_hand"),
        "game_started": row_value(row, "game_started"),
        "matchup": row_value(row, "matchup"),
        "mode": row_value(row, "mode"),
    }
    return out


def load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def filter_rows(rows: Sequence[Dict[str, str]], anchor_ids: Sequence[str]) -> List[Dict[str, str]]:
    wanted = [anchor_id.strip() for anchor_id in anchor_ids if anchor_id.strip()]
    if not wanted:
        return list(rows)
    by_id = {row_value(row, "anchor_id"): row for row in rows}
    missing = [anchor_id for anchor_id in wanted if anchor_id not in by_id]
    if missing:
        raise ConversionError("missing requested anchors: " + ", ".join(missing))
    return [by_id[anchor_id] for anchor_id in wanted]


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-csv",
        default=str(DEFAULT_MANIFEST_CSV),
        help="Input replay_anchor_manifest.csv.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output ActionCounterfactual replay CSV. Defaults beside the manifest.",
    )
    parser.add_argument(
        "--anchor-id",
        action="append",
        default=[],
        help="Only convert this anchor id. Can be passed more than once.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require replay_ready=true and all Java replay loader fields.",
    )
    parser.add_argument(
        "--stack-agent-opening-hand",
        action="store_true",
        help=(
            "Populate first_priority_hand from the manifest. By default the replay CSV leaves it blank "
            "so ActionCounterfactualTrainer uses the recorded seed/deck order unchanged."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv)
    out_csv = Path(args.out_csv) if args.out_csv else manifest_csv.with_name("action_counterfactual_replay.csv")
    source_rows = filter_rows(load_manifest(manifest_csv), args.anchor_id)
    converted = [
        convert_row(row, strict=args.strict, stack_agent_opening_hand=args.stack_agent_opening_hand)
        for row in source_rows
    ]
    write_csv(out_csv, converted)
    summary = {
        "rows": len(converted),
        "out_csv": str(out_csv),
        "manifest_csv": str(manifest_csv),
        "anchors": [row["anchor_id"] for row in converted],
        "action_types": dict(Counter(row["action_type"] for row in converted)),
        "required_columns": list(JAVA_REQUIRED_COLUMNS),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
