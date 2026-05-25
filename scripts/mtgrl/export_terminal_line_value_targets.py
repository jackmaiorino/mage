#!/usr/bin/env python3
"""Export soft value targets from terminal-line checkpoint search artifacts."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from export_terminal_line_teacher_labels import (
    action_key,
    aggregate_action,
    as_float,
    as_int,
    checkpoint_key,
    enriched_rows,
    pct,
    resolve_input_rows,
    valid_terminal,
)


DATASET_VERSION = "terminal_line_value_targets_v1"
ADMISSION_GATE = "terminal_line_common_seed_value_target"

VALUE_FIELDS = [
    "example_id",
    "dataset_version",
    "manifest_status",
    "training_status",
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
    "actions_evaluated",
    "eligible_actions",
    "total_terminal_attempts",
    "min_action_attempts",
    "common_samples",
    "value_source",
    "best_indices",
    "best_texts",
    "best_value",
    "best_attempts",
    "best_wins",
    "worst_indices",
    "worst_texts",
    "worst_value",
    "worst_attempts",
    "worst_wins",
    "value_delta",
    "label_weight",
    "confidence",
    "target_temperature",
    "target_distribution",
    "candidate_value_estimates",
    "candidate_attempts",
    "candidate_win_rates",
    "classification",
    "quality_flags",
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
    "eligible_actions",
    "total_terminal_attempts",
    "best_indices",
    "best_texts",
    "best_value",
    "worst_indices",
    "worst_texts",
    "worst_value",
    "value_delta",
    "quality_flags",
    "rejection_reasons",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read terminal_line_search.csv artifacts and export checkpoint-level "
            "soft value targets over root action candidates."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Run directories or terminal_line_search.csv files.")
    parser.add_argument("--output-dir", required=True, help="Directory for exported value-target artifacts.")
    parser.add_argument("--prefer-merged", action="store_true", help="Prefer root merged CSVs in run directories.")
    parser.add_argument("--min-actions", type=int, default=2)
    parser.add_argument("--min-attempts-per-action", type=int, default=2)
    parser.add_argument("--min-terminal-rate", type=float, default=1.0)
    parser.add_argument("--min-value-delta", type=float, default=0.10)
    parser.add_argument(
        "--min-common-samples",
        type=int,
        default=2,
        help="Use common continuation samples when at least this many samples exist for every eligible action.",
    )
    parser.add_argument(
        "--require-common-samples",
        action="store_true",
        help="Reject groups without enough common continuation samples instead of falling back to aggregate win rate.",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=0.25,
        help="Softmax temperature for converting action value estimates into candidate probabilities.",
    )
    parser.add_argument(
        "--confidence-attempts",
        type=int,
        default=8,
        help="Evidence count that maps a full value delta to weight 1.0.",
    )
    parser.add_argument(
        "--include-suspect-pass-best",
        action="store_true",
        help="Keep pass-best rows flagged as suspect in the main manifest instead of excluding them to diagnostics.",
    )
    parser.add_argument(
        "--pass-best-min-common-samples",
        type=int,
        default=16,
        help="Common continuation samples required before pass-over-setup rows stop being low-evidence suspect rows.",
    )
    parser.add_argument(
        "--pass-best-min-value",
        type=float,
        default=0.50,
        help="Pass-best terminal win rate required before pass-over-setup rows stop being low-value suspect rows.",
    )
    parser.add_argument(
        "--pass-best-min-delta",
        type=float,
        default=0.50,
        help="Best-vs-worst value delta required before pass-over-setup rows stop being low-margin suspect rows.",
    )
    parser.add_argument("--expect-examples", type=int, default=None)
    parser.add_argument("--write-readme", action="store_true")
    return parser.parse_args(argv)


def parse_indices(value: object) -> List[int]:
    text = str(value or "").strip()
    if not text:
        return []
    out: List[int] = []
    for part in text.replace(",", "|").split("|"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def format_float(value: float) -> str:
    return "{:.6f}".format(float(value))


def compact_pairs(items: Iterable[Tuple[str, float]]) -> str:
    return "|".join("{}:{}".format(key, format_float(value)) for key, value in items)


def compact_int_pairs(items: Iterable[Tuple[str, int]]) -> str:
    return "|".join("{}:{}".format(key, int(value)) for key, value in items)


def action_identifier(action: Dict[str, object]) -> str:
    indices = str(action.get("indices", "")).strip()
    texts = str(action.get("texts", "")).strip().replace("|", "/")
    return "{}={}".format(indices, texts)


def is_pass_action(action: Optional[Dict[str, object]]) -> bool:
    if action is None:
        return False
    return str(action.get("texts", "")).strip().lower() == "pass"


def is_setup_like_action(action: Optional[Dict[str, object]]) -> bool:
    if action is None:
        return False
    text = str(action.get("texts", "")).strip().lower()
    if not text or text == "pass":
        return False
    setup_hints = (
        "play ",
        "cycling",
        "add {",
        "add one mana",
        "tinder wall",
        "wall of roots",
        "overgrown battlement",
        "saruli caretaker",
        "quirion ranger",
        "gatecreeper vine",
        "generous ent",
    )
    return any(hint in text for hint in setup_hints)


def common_samples(actions: Sequence[Dict[str, object]]) -> List[str]:
    sample_sets: List[set] = []
    for action in actions:
        samples = action.get("sample_outcomes", {})
        if not isinstance(samples, dict):
            return []
        sample_sets.append(set(str(key) for key in samples.keys()))
    if not sample_sets:
        return []
    common = sample_sets[0]
    for sample_set in sample_sets[1:]:
        common = common.intersection(sample_set)
    return sorted(common)


def value_for_common_samples(action: Dict[str, object], samples: Sequence[str]) -> Tuple[int, int, float]:
    sample_outcomes = action.get("sample_outcomes", {})
    if not isinstance(sample_outcomes, dict) or not samples:
        return 0, 0, 0.0
    wins = sum(1 for sample in samples if sample_outcomes.get(sample) == "terminal_win")
    losses = sum(1 for sample in samples if sample_outcomes.get(sample) == "terminal_loss")
    return wins, losses, pct(wins, len(samples))


def choose_value_source(
        actions: Sequence[Dict[str, object]],
        args: argparse.Namespace,
) -> Tuple[str, List[str], Dict[int, float]]:
    samples = common_samples(actions)
    if len(samples) >= args.min_common_samples:
        values: Dict[int, float] = {}
        for index, action in enumerate(actions):
            wins, losses, value = value_for_common_samples(action, samples)
            action["paired_common_wins"] = wins
            action["paired_common_losses"] = losses
            action["paired_common_value"] = value
            values[index] = value
        return "paired_common_win_rate", samples, values
    if args.require_common_samples:
        return "missing_common_samples", samples, {}
    return "aggregate_win_rate", samples, {
        index: as_float(action, "win_rate") for index, action in enumerate(actions)
    }


def action_rank_key(item: Tuple[int, Dict[str, object], float]) -> Tuple[float, int, int, int]:
    _, action, value = item
    return (
        value,
        as_int(action, "wins"),
        as_int(action, "max_combo_score"),
        -as_int(action, "min_win_decisions", default=999999),
    )


def softmax_distribution(
        actions: Sequence[Dict[str, object]],
        values: Dict[int, float],
        candidate_count: int,
        temperature: float,
) -> List[float]:
    max_candidates = max(0, candidate_count)
    out = [0.0 for _ in range(max_candidates)]
    if not actions or max_candidates <= 0:
        return out
    safe_temp = max(0.001, float(temperature))
    logits = [float(values.get(index, 0.0)) / safe_temp for index, _ in enumerate(actions)]
    max_logit = max(logits) if logits else 0.0
    weights = [math.exp(logit - max_logit) for logit in logits]
    denom = sum(weights)
    if denom <= 0.0:
        return out
    for action_index, action in enumerate(actions):
        indices = [idx for idx in parse_indices(action.get("indices", "")) if 0 <= idx < max_candidates]
        if not indices:
            continue
        share = weights[action_index] / denom / float(len(indices))
        for idx in indices:
            out[idx] += share
    total = sum(out)
    if total > 0.0:
        out = [value / total for value in out]
    return out


def candidate_value_map(
        actions: Sequence[Dict[str, object]],
        values: Dict[int, float],
        candidate_count: int,
) -> List[float]:
    sums = [0.0 for _ in range(candidate_count)]
    counts = [0 for _ in range(candidate_count)]
    for action_index, action in enumerate(actions):
        value = float(values.get(action_index, 0.0))
        for idx in parse_indices(action.get("indices", "")):
            if 0 <= idx < candidate_count:
                sums[idx] += value
                counts[idx] += 1
    return [
        (sums[index] / float(counts[index])) if counts[index] else 0.0
        for index in range(candidate_count)
    ]


def candidate_attempt_map(actions: Sequence[Dict[str, object]], candidate_count: int) -> List[int]:
    out = [0 for _ in range(candidate_count)]
    for action in actions:
        for idx in parse_indices(action.get("indices", "")):
            if 0 <= idx < candidate_count:
                out[idx] += as_int(action, "attempts")
    return out


def classify(delta: float, confidence: float) -> str:
    if delta >= 0.75 and confidence >= 0.75:
        return "terminal_value_strong_delta"
    if delta >= 0.25:
        return "terminal_value_moderate_delta"
    return "terminal_value_weak_delta"


def pass_best_quality_flags(
        best: Optional[Dict[str, object]],
        eligible: Sequence[Dict[str, object]],
        best_value: float,
        delta: float,
        common_sample_count: int,
        args: argparse.Namespace,
) -> List[str]:
    if not is_pass_action(best):
        return []
    flags = ["pass_best"]
    if any(is_setup_like_action(action) for action in eligible if action is not best):
        flags.append("pass_best_over_setup")
    if common_sample_count < args.pass_best_min_common_samples:
        flags.append("pass_best_low_common_samples")
    if best_value < args.pass_best_min_value:
        flags.append("pass_best_low_value")
    if delta < args.pass_best_min_delta:
        flags.append("pass_best_low_delta")
    if "pass_best_over_setup" in flags and (
            "pass_best_low_common_samples" in flags
            or "pass_best_low_value" in flags
            or "pass_best_low_delta" in flags):
        flags.append("suspect_pass_best")
    return flags


def make_example_id(base: Dict[str, object], target_distribution: Sequence[float]) -> str:
    payload = "|".join([
        str(base.get("snapshot_path", "")),
        str(base.get("ordinal", "")),
        str(base.get("decision_number", "")),
        str(base.get("candidate_hash", "")),
        ",".join(format_float(value) for value in target_distribution),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def rejection_row(
        input_run_id: str,
        base: Dict[str, object],
        actions: Sequence[Dict[str, object]],
        eligible: Sequence[Dict[str, object]],
        best: Optional[Dict[str, object]],
        worst: Optional[Dict[str, object]],
        delta: float,
        quality_flags: Sequence[str],
        reasons: Sequence[str],
) -> Dict[str, object]:
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
        "eligible_actions": len(eligible),
        "total_terminal_attempts": sum(as_int(action, "terminal") for action in actions),
        "best_indices": "" if best is None else best.get("indices", ""),
        "best_texts": "" if best is None else best.get("texts", ""),
        "best_value": "" if best is None else format_float(as_float(best, "target_value")),
        "worst_indices": "" if worst is None else worst.get("indices", ""),
        "worst_texts": "" if worst is None else worst.get("texts", ""),
        "worst_value": "" if worst is None else format_float(as_float(worst, "target_value")),
        "value_delta": format_float(delta),
        "quality_flags": "|".join(quality_flags),
        "rejection_reasons": "|".join(reasons),
    }


def example_row(
        input_run_id: str,
        base: Dict[str, object],
        actions: Sequence[Dict[str, object]],
        eligible: Sequence[Dict[str, object]],
        values: Dict[int, float],
        value_source: str,
        samples: Sequence[str],
        quality_flags: Sequence[str],
        args: argparse.Namespace,
) -> Dict[str, object]:
    candidate_count = as_int(base, "candidate_count")
    ranked = sorted(
        ((index, action, values.get(index, 0.0)) for index, action in enumerate(eligible)),
        key=action_rank_key,
        reverse=True,
    )
    best_index, best, best_value = ranked[0]
    worst_index, worst, worst_value = ranked[-1]
    delta = max(0.0, best_value - worst_value)
    evidence_n = len(samples) if value_source == "paired_common_win_rate" else min(
        as_int(action, "attempts") for action in eligible
    )
    confidence = min(1.0, math.sqrt(float(max(1, evidence_n)) / float(max(1, args.confidence_attempts))))
    label_weight = min(1.0, delta * confidence)
    distribution = softmax_distribution(eligible, values, candidate_count, args.target_temperature)
    value_map = candidate_value_map(eligible, values, candidate_count)
    attempts_map = candidate_attempt_map(eligible, candidate_count)
    win_rate_map = candidate_value_map(
        eligible,
        {index: as_float(action, "win_rate") for index, action in enumerate(eligible)},
        candidate_count,
    )
    suspect = "suspect_pass_best" in quality_flags
    train_suspect = suspect and args.include_suspect_pass_best
    row = {
        "example_id": make_example_id(base, distribution),
        "dataset_version": DATASET_VERSION,
        "manifest_status": "admitted" if (not suspect or train_suspect) else "suspect",
        "training_status": "trainable" if (not suspect or train_suspect) else "excluded_suspect",
        "admission_gate": ADMISSION_GATE,
        "input_run_id": input_run_id,
        "snapshot_path": base.get("snapshot_path", ""),
        "ordinal": base.get("ordinal", ""),
        "decision_number": base.get("decision_number", ""),
        "action_type": base.get("action_type", ""),
        "candidate_count": candidate_count,
        "candidate_hash": base.get("candidate_hash", ""),
        "state_hash": base.get("state_hash", ""),
        "rng_state_hash": base.get("rng_state_hash", ""),
        "actions_evaluated": len(actions),
        "eligible_actions": len(eligible),
        "total_terminal_attempts": sum(as_int(action, "terminal") for action in actions),
        "min_action_attempts": min(as_int(action, "attempts") for action in eligible),
        "common_samples": len(samples),
        "value_source": value_source,
        "best_indices": best.get("indices", ""),
        "best_texts": best.get("texts", ""),
        "best_value": format_float(best_value),
        "best_attempts": best.get("attempts", 0),
        "best_wins": best.get("wins", 0),
        "worst_indices": worst.get("indices", ""),
        "worst_texts": worst.get("texts", ""),
        "worst_value": format_float(worst_value),
        "worst_attempts": worst.get("attempts", 0),
        "worst_wins": worst.get("wins", 0),
        "value_delta": format_float(delta),
        "label_weight": format_float(label_weight),
        "confidence": format_float(confidence),
        "target_temperature": format_float(args.target_temperature),
        "target_distribution": compact_pairs((str(index), value) for index, value in enumerate(distribution) if value > 0.0),
        "candidate_value_estimates": compact_pairs((str(index), value) for index, value in enumerate(value_map) if value > 0.0),
        "candidate_attempts": compact_int_pairs((str(index), value) for index, value in enumerate(attempts_map) if value > 0),
        "candidate_win_rates": compact_pairs((str(index), value) for index, value in enumerate(win_rate_map) if value > 0.0),
        "classification": classify(delta, confidence),
        "quality_flags": "|".join(quality_flags),
        "_json_detail": {
            "candidate_values": value_map,
            "candidate_attempts": attempts_map,
            "candidate_win_rates": win_rate_map,
            "target_distribution": distribution,
            "actions": [
                {
                    "indices": action.get("indices", ""),
                    "texts": action.get("texts", ""),
                    "attempts": action.get("attempts", 0),
                    "wins": action.get("wins", 0),
                    "losses": action.get("losses", 0),
                    "win_rate": action.get("win_rate", 0.0),
                    "target_value": values.get(index, 0.0),
                    "max_combo_score": action.get("max_combo_score", 0),
                    "min_win_decisions": action.get("min_win_decisions", 0),
                }
                for index, action in enumerate(eligible)
            ],
        },
    }
    # Record these for rejection-row helpers if the caller reuses action dicts.
    best["target_value"] = best_value
    worst["target_value"] = worst_value
    return row


def group_actions(group_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_action: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in group_rows:
        by_action[action_key(row)].append(row)
    actions = [aggregate_action(action_rows) for action_rows in by_action.values()]
    return [action for action in actions if as_int(action, "attempts") > 0]


def process_input(
        input_run_id: str,
        raw_rows: List[Dict[str, str]],
        args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Counter]:
    rows = [row for row in enriched_rows(raw_rows) if valid_terminal(row)]
    groups: Dict[Tuple[str, str, str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[checkpoint_key(row)].append(row)

    examples: List[Dict[str, object]] = []
    rejects: List[Dict[str, object]] = []
    counters = Counter()
    counters["raw_rows"] += len(raw_rows)
    counters["valid_terminal_rows"] += len(rows)
    counters["checkpoint_groups"] += len(groups)

    for _, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        base = group_rows[0]
        actions = group_actions(group_rows)
        eligible = [
            action for action in actions
            if as_int(action, "attempts") >= args.min_attempts_per_action
            and as_float(action, "terminal_rate") >= args.min_terminal_rate
            and parse_indices(action.get("indices", ""))
        ]
        reasons: List[str] = []
        best = None
        worst = None
        delta = 0.0
        quality_flags: List[str] = []
        values: Dict[int, float] = {}
        samples: List[str] = []
        value_source = ""
        if len(actions) < args.min_actions:
            reasons.append("actions_lt_min")
        if len(eligible) < args.min_actions:
            reasons.append("eligible_actions_lt_min")
        if not reasons:
            value_source, samples, values = choose_value_source(eligible, args)
            if value_source == "missing_common_samples":
                reasons.append("common_samples_lt_min")
            elif not values:
                reasons.append("missing_value_estimates")
        if not reasons:
            ranked = sorted(
                ((index, action, values.get(index, 0.0)) for index, action in enumerate(eligible)),
                key=action_rank_key,
                reverse=True,
            )
            best_index, best, best_value = ranked[0]
            worst_index, worst, worst_value = ranked[-1]
            best["target_value"] = best_value
            worst["target_value"] = worst_value
            delta = best_value - worst_value
            quality_flags = pass_best_quality_flags(
                best, eligible, best_value, delta, len(samples), args)
            if delta < args.min_value_delta:
                reasons.append("value_delta_below_threshold")
            if "suspect_pass_best" in quality_flags and not args.include_suspect_pass_best:
                reasons.append("suspect_pass_best")
            if as_int(base, "candidate_count") <= 1:
                reasons.append("candidate_count_lt_2")
            if not softmax_distribution(eligible, values, as_int(base, "candidate_count"), args.target_temperature):
                reasons.append("empty_target_distribution")
        if reasons:
            rejects.append(rejection_row(input_run_id, base, actions, eligible, best, worst, delta, quality_flags, reasons))
            for reason in reasons:
                counters["reject_" + reason] += 1
            continue
        examples.append(example_row(input_run_id, base, actions, eligible, values, value_source, samples, quality_flags, args))
        counters["admitted"] += 1

    return examples, rejects, counters


def read_all_inputs(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Counter]:
    examples: List[Dict[str, object]] = []
    rejects: List[Dict[str, object]] = []
    total = Counter()
    for raw_input in args.inputs:
        input_run_id, rows = resolve_input_rows(raw_input, args.prefer_merged)
        input_examples, input_rejects, counters = process_input(input_run_id, rows, args)
        examples.extend(input_examples)
        rejects.extend(input_rejects)
        total.update(counters)
    return examples, rejects, total


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = dict(row)
            detail = payload.pop("_json_detail", {})
            payload.update(detail)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def markdown(summary: Dict[str, object], examples: Sequence[Dict[str, object]]) -> str:
    lines = [
        "# Terminal Line Value Targets",
        "",
        "Generated by `scripts/mtgrl/export_terminal_line_value_targets.py`.",
        "",
        "- dataset_version: `{}`".format(DATASET_VERSION),
        "- admission_gate: `{}`".format(ADMISSION_GATE),
        "- admitted_examples: `{}`".format(summary["admitted_examples"]),
        "- rejected_groups: `{}`".format(summary["rejected_groups"]),
        "- valid_terminal_rows: `{}`".format(summary["valid_terminal_rows"]),
        "- checkpoint_groups: `{}`".format(summary["checkpoint_groups"]),
        "- classification_counts: `{}`".format(summary["classification_counts"]),
        "- rejection_counts: `{}`".format(summary["rejection_counts"]),
        "",
        "Each admitted row stores a soft target distribution over candidate indices.",
        "The target value is terminal win rate, preferring common continuation samples",
        "when all evaluated root actions share enough sample ids.",
        "",
        "## Top Targets",
        "",
        "| best | worst | delta | weight | source | distribution |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    ranked = sorted(
        examples,
        key=lambda row: (
            as_float(row, "label_weight"),
            as_float(row, "value_delta"),
            as_int(row, "common_samples"),
        ),
        reverse=True,
    )
    for row in ranked[:15]:
        lines.append(
            "| {best} | {worst} | {delta} | {weight} | {source} | {dist} |".format(
                best=str(row.get("best_texts", "")).replace("|", "/"),
                worst=str(row.get("worst_texts", "")).replace("|", "/"),
                delta=row.get("value_delta", ""),
                weight=row.get("label_weight", ""),
                source=row.get("value_source", ""),
                dist=str(row.get("target_distribution", "")).replace("|", "<br>"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples, rejects, counters = read_all_inputs(args)
    classification_counts = Counter(str(row.get("classification", "")) for row in examples)
    rejection_counts = Counter()
    for row in rejects:
        for reason in str(row.get("rejection_reasons", "")).split("|"):
            if reason:
                rejection_counts[reason] += 1
    quality_flag_counts = Counter()
    for row in examples:
        for flag in str(row.get("quality_flags", "")).split("|"):
            if flag:
                quality_flag_counts[flag] += 1
    for row in rejects:
        for flag in str(row.get("quality_flags", "")).split("|"):
            if flag:
                quality_flag_counts[flag] += 1

    summary = {
        "dataset_version": DATASET_VERSION,
        "admission_gate": ADMISSION_GATE,
        "admitted_examples": len(examples),
        "rejected_groups": len(rejects),
        "raw_rows": counters.get("raw_rows", 0),
        "valid_terminal_rows": counters.get("valid_terminal_rows", 0),
        "checkpoint_groups": counters.get("checkpoint_groups", 0),
        "classification_counts": dict(sorted(classification_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "inputs": list(args.inputs),
        "min_actions": args.min_actions,
        "min_attempts_per_action": args.min_attempts_per_action,
        "min_terminal_rate": args.min_terminal_rate,
        "min_value_delta": args.min_value_delta,
        "min_common_samples": args.min_common_samples,
        "require_common_samples": args.require_common_samples,
        "target_temperature": args.target_temperature,
        "confidence_attempts": args.confidence_attempts,
        "quality_flag_counts": dict(sorted(quality_flag_counts.items())),
        "include_suspect_pass_best": args.include_suspect_pass_best,
        "pass_best_min_common_samples": args.pass_best_min_common_samples,
        "pass_best_min_value": args.pass_best_min_value,
        "pass_best_min_delta": args.pass_best_min_delta,
    }

    write_csv(output_dir / "terminal_line_value_targets.csv", VALUE_FIELDS, examples)
    write_csv(output_dir / "terminal_line_value_target_rejections.csv", REJECT_FIELDS, rejects)
    write_jsonl(output_dir / "terminal_line_value_targets.jsonl", examples)
    (output_dir / "terminal_line_value_target_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.write_readme:
        (output_dir / "README.md").write_text(markdown(summary, examples), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.expect_examples is not None and len(examples) != args.expect_examples:
        raise SystemExit("expected {} examples, found {}".format(args.expect_examples, len(examples)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
