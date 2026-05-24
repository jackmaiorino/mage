#!/usr/bin/env python3
"""Export strict teacher labels from terminal-line checkpoint search artifacts."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from summarize_terminal_line_search import TERMINAL_LINE_CSV, read_csvs, trace_features


ADMISSION_GATE = "terminal_line_paired_win_loss_delta"

LABEL_FIELDS = [
    "label_id",
    "manifest_status",
    "admission_gate",
    "input_run_id",
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
    "source_attempts",
    "source_wins",
    "source_losses",
    "source_win_rate",
    "source_loss_rate",
    "source_terminal_rate",
    "target_attempts",
    "target_wins",
    "target_losses",
    "target_win_rate",
    "target_loss_rate",
    "target_terminal_rate",
    "delta_win_rate",
    "combo_score_delta",
    "importance_score",
    "label_weight",
    "source_max_combo_score",
    "target_max_combo_score",
    "source_min_win_decisions",
    "target_min_win_decisions",
    "actions_evaluated",
    "total_terminal_attempts",
    "classification",
    "source_outcomes",
    "target_outcomes",
]

REJECT_FIELDS = [
    "input_run_id",
    "snapshot_path",
    "ordinal",
    "decision_number",
    "action_type",
    "candidate_hash",
    "state_hash",
    "rng_state_hash",
    "actions_evaluated",
    "total_terminal_attempts",
    "best_indices",
    "best_texts",
    "best_win_rate",
    "best_attempts",
    "worst_indices",
    "worst_texts",
    "worst_win_rate",
    "worst_attempts",
    "delta_win_rate",
    "rejection_reasons",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read terminal_line_search.csv artifacts and export strict first-action "
            "teacher labels from paired terminal win/loss roots."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Run directories or terminal_line_search.csv files.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported teacher-label artifacts.")
    parser.add_argument("--prefer-merged", action="store_true", help="Prefer root merged CSVs in run directories.")
    parser.add_argument("--min-actions", type=int, default=2)
    parser.add_argument("--min-attempts-per-action", type=int, default=1)
    parser.add_argument("--min-target-wins", type=int, default=1)
    parser.add_argument("--min-target-win-rate", type=float, default=1.0)
    parser.add_argument("--max-source-win-rate", type=float, default=0.0)
    parser.add_argument("--min-delta", type=float, default=1.0)
    parser.add_argument("--min-source-terminal-rate", type=float, default=1.0)
    parser.add_argument("--min-target-terminal-rate", type=float, default=1.0)
    parser.add_argument(
        "--min-target-combo-score",
        type=int,
        default=0,
        help="Optional diagnostic gate. Default 0 keeps reward terminal-only.",
    )
    parser.add_argument("--expect-admitted", type=int, default=None)
    parser.add_argument("--write-readme", action="store_true")
    return parser.parse_args(argv)


def read_csv_file(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            out = dict(row)
            out["_artifact"] = str(path)
            rows.append(out)
    return rows


def resolve_input_rows(raw_input: str, prefer_merged: bool) -> Tuple[str, List[Dict[str, str]]]:
    path = Path(raw_input)
    if path.is_dir():
        return path.name, read_csvs(path, prefer_merged)
    if path.is_file():
        return path.parent.name, read_csv_file(path)
    raise FileNotFoundError("missing terminal-line input: {}".format(raw_input))


def boolish(row: Dict[str, object], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() == "true"


def as_int(row: Dict[str, object], key: str, default: int = 0) -> int:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def as_float(row: Dict[str, object], key: str, default: float = 0.0) -> float:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def pct(numer: int, denom: int) -> float:
    return 0.0 if denom <= 0 else round(float(numer) / float(denom), 6)


def normalize_path(path: str) -> str:
    return str(path or "").replace("\\", "/")


def checkpoint_key(row: Dict[str, object]) -> Tuple[str, str, str, str, str, str, str]:
    return (
        normalize_path(str(row.get("snapshot_path", ""))),
        str(row.get("ordinal", "")).strip(),
        str(row.get("decision_number", "")).strip(),
        str(row.get("action_type", "")).strip(),
        str(row.get("candidate_hash", "")).strip(),
        str(row.get("state_hash", "")).strip(),
        str(row.get("rng_state_hash", "")).strip(),
    )


def action_key(row: Dict[str, object]) -> Tuple[str, str]:
    return (str(row.get("root_indices", "")).strip(), str(row.get("root_texts", "")).strip())


def valid_terminal(row: Dict[str, object]) -> bool:
    if str(row.get("error", "")).strip():
        return False
    return str(row.get("outcome", "")).strip() in ("terminal_win", "terminal_loss")


def enriched_rows(raw_rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for raw in raw_rows:
        feature = trace_features(raw)
        merged: Dict[str, object] = dict(raw)
        for key, value in feature.items():
            if key not in merged:
                merged[key] = value
        rows.append(merged)
    return rows


def aggregate_action(rows: List[Dict[str, object]]) -> Dict[str, object]:
    attempts = len(rows)
    wins = sum(1 for row in rows if str(row.get("outcome", "")).strip() == "terminal_win")
    losses = sum(1 for row in rows if str(row.get("outcome", "")).strip() == "terminal_loss")
    terminal = wins + losses
    win_decisions = [
        as_int(row, "decision_count")
        for row in rows
        if str(row.get("outcome", "")).strip() == "terminal_win"
    ]
    combo_scores = [as_int(row, "combo_score") for row in rows]
    first = rows[0] if rows else {}
    return {
        "indices": str(first.get("root_indices", "")).strip(),
        "texts": str(first.get("root_texts", "")).strip(),
        "attempts": attempts,
        "wins": wins,
        "losses": losses,
        "terminal": terminal,
        "win_rate": pct(wins, attempts),
        "loss_rate": pct(losses, attempts),
        "terminal_rate": pct(terminal, attempts),
        "max_combo_score": max(combo_scores) if combo_scores else 0,
        "min_win_decisions": min(win_decisions) if win_decisions else 0,
        "outcomes": "|".join(str(row.get("outcome", "")).strip() for row in rows),
    }


def action_sort_key(action: Dict[str, object]) -> Tuple[float, int, int, int, float]:
    return (
        as_float(action, "win_rate"),
        as_int(action, "wins"),
        as_int(action, "max_combo_score"),
        -as_int(action, "min_win_decisions", default=999999),
        -as_float(action, "loss_rate"),
    )


def worst_sort_key(action: Dict[str, object]) -> Tuple[float, int, float, int]:
    return (
        as_float(action, "win_rate"),
        -as_int(action, "losses"),
        -as_float(action, "terminal_rate"),
        as_int(action, "max_combo_score"),
    )


def rejection_reasons(
        actions: List[Dict[str, object]],
        source: Optional[Dict[str, object]],
        target: Optional[Dict[str, object]],
        args: argparse.Namespace,
) -> List[str]:
    reasons: List[str] = []
    if len(actions) < args.min_actions:
        reasons.append("not_enough_actions")
    if source is None or target is None:
        reasons.append("missing_source_or_target")
        return reasons
    if source["indices"] == target["indices"]:
        reasons.append("source_target_same_indices")
    if as_int(source, "attempts") < args.min_attempts_per_action:
        reasons.append("source_attempts_below_threshold")
    if as_int(target, "attempts") < args.min_attempts_per_action:
        reasons.append("target_attempts_below_threshold")
    if as_float(source, "terminal_rate") < args.min_source_terminal_rate:
        reasons.append("source_terminal_rate_below_threshold")
    if as_float(target, "terminal_rate") < args.min_target_terminal_rate:
        reasons.append("target_terminal_rate_below_threshold")
    if as_int(target, "wins") < args.min_target_wins:
        reasons.append("target_wins_below_threshold")
    if as_float(target, "win_rate") < args.min_target_win_rate:
        reasons.append("target_win_rate_below_threshold")
    if as_float(source, "win_rate") > args.max_source_win_rate:
        reasons.append("source_win_rate_above_threshold")
    if as_float(target, "win_rate") - as_float(source, "win_rate") < args.min_delta:
        reasons.append("delta_below_threshold")
    if as_int(target, "max_combo_score") < args.min_target_combo_score:
        reasons.append("target_combo_score_below_threshold")
    return reasons


def label_id(row: Dict[str, object], source: Dict[str, object], target: Dict[str, object]) -> str:
    seed = "|".join(
        [
            normalize_path(str(row.get("snapshot_path", ""))),
            str(row.get("candidate_hash", "")),
            str(row.get("state_hash", "")),
            str(row.get("rng_state_hash", "")),
            str(source.get("indices", "")),
            str(target.get("indices", "")),
            str(source.get("win_rate", "")),
            str(target.get("win_rate", "")),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def classify(source: Dict[str, object], target: Dict[str, object], delta: float) -> str:
    if as_float(source, "loss_rate") >= 1.0 and as_float(target, "win_rate") >= 1.0:
        return "terminal_dominant_win_loss"
    if delta >= 0.75:
        return "terminal_strong_delta"
    return "terminal_moderate_delta"


def admitted_row(
        input_run_id: str,
        base: Dict[str, object],
        source: Dict[str, object],
        target: Dict[str, object],
        actions: List[Dict[str, object]],
) -> Dict[str, object]:
    delta = round(as_float(target, "win_rate") - as_float(source, "win_rate"), 6)
    combo_delta = as_int(target, "max_combo_score") - as_int(source, "max_combo_score")
    confidence = math.sqrt(min(as_int(source, "attempts"), as_int(target, "attempts")))
    importance = round(min(1.0, max(0.0, delta) * confidence), 6)
    return {
        "label_id": label_id(base, source, target),
        "manifest_status": "admitted",
        "admission_gate": ADMISSION_GATE,
        "input_run_id": input_run_id,
        "snapshot_path": base.get("snapshot_path", ""),
        "ordinal": base.get("ordinal", ""),
        "decision_number": base.get("decision_number", ""),
        "action_type": base.get("action_type", ""),
        "candidate_count": base.get("candidate_count", ""),
        "candidate_hash": base.get("candidate_hash", ""),
        "state_hash": base.get("state_hash", ""),
        "rng_state_hash": base.get("rng_state_hash", ""),
        "source_indices": source.get("indices", ""),
        "source_texts": source.get("texts", ""),
        "target_indices": target.get("indices", ""),
        "target_texts": target.get("texts", ""),
        "source_attempts": source.get("attempts", 0),
        "source_wins": source.get("wins", 0),
        "source_losses": source.get("losses", 0),
        "source_win_rate": source.get("win_rate", 0.0),
        "source_loss_rate": source.get("loss_rate", 0.0),
        "source_terminal_rate": source.get("terminal_rate", 0.0),
        "target_attempts": target.get("attempts", 0),
        "target_wins": target.get("wins", 0),
        "target_losses": target.get("losses", 0),
        "target_win_rate": target.get("win_rate", 0.0),
        "target_loss_rate": target.get("loss_rate", 0.0),
        "target_terminal_rate": target.get("terminal_rate", 0.0),
        "delta_win_rate": delta,
        "combo_score_delta": combo_delta,
        "importance_score": importance,
        "label_weight": importance,
        "source_max_combo_score": source.get("max_combo_score", 0),
        "target_max_combo_score": target.get("max_combo_score", 0),
        "source_min_win_decisions": source.get("min_win_decisions", 0),
        "target_min_win_decisions": target.get("min_win_decisions", 0),
        "actions_evaluated": len(actions),
        "total_terminal_attempts": sum(as_int(action, "terminal") for action in actions),
        "classification": classify(source, target, delta),
        "source_outcomes": source.get("outcomes", ""),
        "target_outcomes": target.get("outcomes", ""),
    }


def rejected_row(
        input_run_id: str,
        base: Dict[str, object],
        actions: List[Dict[str, object]],
        source: Optional[Dict[str, object]],
        target: Optional[Dict[str, object]],
        reasons: List[str],
) -> Dict[str, object]:
    source = source or {}
    target = target or {}
    return {
        "input_run_id": input_run_id,
        "snapshot_path": base.get("snapshot_path", ""),
        "ordinal": base.get("ordinal", ""),
        "decision_number": base.get("decision_number", ""),
        "action_type": base.get("action_type", ""),
        "candidate_hash": base.get("candidate_hash", ""),
        "state_hash": base.get("state_hash", ""),
        "rng_state_hash": base.get("rng_state_hash", ""),
        "actions_evaluated": len(actions),
        "total_terminal_attempts": sum(as_int(action, "terminal") for action in actions),
        "best_indices": target.get("indices", ""),
        "best_texts": target.get("texts", ""),
        "best_win_rate": target.get("win_rate", 0.0),
        "best_attempts": target.get("attempts", 0),
        "worst_indices": source.get("indices", ""),
        "worst_texts": source.get("texts", ""),
        "worst_win_rate": source.get("win_rate", 0.0),
        "worst_attempts": source.get("attempts", 0),
        "delta_win_rate": round(as_float(target, "win_rate") - as_float(source, "win_rate"), 6)
        if source and target else 0.0,
        "rejection_reasons": "|".join(reasons),
    }


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def process_input(
        input_run_id: str,
        raw_rows: List[Dict[str, str]],
        args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Counter]:
    rows = [row for row in enriched_rows(raw_rows) if valid_terminal(row)]
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[checkpoint_key(row)].append(row)

    labels: List[Dict[str, object]] = []
    rejects: List[Dict[str, object]] = []
    counters = Counter()
    counters["raw_rows"] += len(raw_rows)
    counters["valid_terminal_rows"] += len(rows)
    counters["checkpoint_groups"] += len(groups)

    for _, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        base = group_rows[0]
        by_action: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in group_rows:
            by_action[action_key(row)].append(row)
        actions = [aggregate_action(action_rows) for action_rows in by_action.values()]
        actions = [action for action in actions if as_int(action, "attempts") > 0]
        target = max(actions, key=action_sort_key) if actions else None
        source = min(actions, key=worst_sort_key) if actions else None
        reasons = rejection_reasons(actions, source, target, args)
        if reasons:
            rejects.append(rejected_row(input_run_id, base, actions, source, target, reasons))
            for reason in reasons:
                counters["reject_" + reason] += 1
            continue
        labels.append(admitted_row(input_run_id, base, source, target, actions))
        counters["admitted"] += 1

    return labels, rejects, counters


def read_all_inputs(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Counter]:
    all_labels: List[Dict[str, object]] = []
    all_rejects: List[Dict[str, object]] = []
    total = Counter()
    for raw_input in args.inputs:
        input_run_id, rows = resolve_input_rows(raw_input, args.prefer_merged)
        labels, rejects, counters = process_input(input_run_id, rows, args)
        all_labels.extend(labels)
        all_rejects.extend(rejects)
        total.update(counters)
    return all_labels, all_rejects, total


def markdown(summary: Dict[str, object], labels: List[Dict[str, object]]) -> str:
    lines = [
        "# Terminal Line Teacher Labels",
        "",
        "- admission_gate: `{}`".format(ADMISSION_GATE),
        "- admitted_labels: `{}`".format(summary["admitted_labels"]),
        "- rejected_groups: `{}`".format(summary["rejected_groups"]),
        "- valid_terminal_rows: `{}`".format(summary["valid_terminal_rows"]),
        "- checkpoint_groups: `{}`".format(summary["checkpoint_groups"]),
        "- classification_counts: `{}`".format(summary["classification_counts"]),
        "- rejection_counts: `{}`".format(summary["rejection_counts"]),
        "",
        "## Top Labels",
        "",
        "| target | source | delta | weight | target_combo | source_combo | snapshot |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    ranked = sorted(
        labels,
        key=lambda row: (
            as_float(row, "label_weight"),
            as_int(row, "target_max_combo_score"),
            -as_int(row, "target_min_win_decisions", default=999999),
        ),
        reverse=True,
    )
    for row in ranked[:15]:
        lines.append(
            "| {target} | {source} | {delta} | {weight} | {tcombo} | {scombo} | {snapshot} |".format(
                target=str(row.get("target_texts", "")).replace("|", "/"),
                source=str(row.get("source_texts", "")).replace("|", "/"),
                delta=row.get("delta_win_rate", ""),
                weight=row.get("label_weight", ""),
                tcombo=row.get("target_max_combo_score", ""),
                scombo=row.get("source_max_combo_score", ""),
                snapshot=Path(str(row.get("snapshot_path", ""))).name,
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, rejects, counters = read_all_inputs(args)
    classification_counts = Counter(str(row.get("classification", "")) for row in labels)
    rejection_counts = Counter()
    for row in rejects:
        for reason in str(row.get("rejection_reasons", "")).split("|"):
            if reason:
                rejection_counts[reason] += 1

    summary = {
        "admission_gate": ADMISSION_GATE,
        "admitted_labels": len(labels),
        "rejected_groups": len(rejects),
        "raw_rows": counters.get("raw_rows", 0),
        "valid_terminal_rows": counters.get("valid_terminal_rows", 0),
        "checkpoint_groups": counters.get("checkpoint_groups", 0),
        "classification_counts": dict(sorted(classification_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "inputs": list(args.inputs),
        "min_actions": args.min_actions,
        "min_attempts_per_action": args.min_attempts_per_action,
        "min_target_wins": args.min_target_wins,
        "min_target_win_rate": args.min_target_win_rate,
        "max_source_win_rate": args.max_source_win_rate,
        "min_delta": args.min_delta,
        "min_target_combo_score": args.min_target_combo_score,
    }

    write_csv(output_dir / "terminal_line_teacher_labels.csv", LABEL_FIELDS, labels)
    write_csv(output_dir / "terminal_line_teacher_rejections.csv", REJECT_FIELDS, rejects)
    write_jsonl(output_dir / "terminal_line_teacher_labels.jsonl", labels)
    (output_dir / "terminal_line_teacher_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.write_readme:
        (output_dir / "README.md").write_text(markdown(summary, labels), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.expect_admitted is not None and len(labels) != args.expect_admitted:
        raise SystemExit(
            "expected {} admitted labels, found {}".format(args.expect_admitted, len(labels))
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
