#!/usr/bin/env python3
"""Compare forced-prefix replay RNG trace artifacts for two run roots."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


JSON_PREFIXES = {
    "agent_search_trace.log": "REPLAY_AGENT_SEARCH_JSON:",
    "random_util_wrapper_trace.log": "RANDOM_UTIL_WRAPPER_JSON:",
}


def read_json_lines(path, prefix):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line.startswith(prefix):
                continue
            payload = line[len(prefix):].strip()
            try:
                row = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            row["_line"] = line_number
            rows.append(row)
    return rows


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def first_join(value):
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return "" if value is None else str(value)


def load_run(root):
    root = Path(root)
    return {
        "root": root,
        "sample": read_csv(root / "replay_samples.csv"),
        "forced": read_csv(root / "forced_prefix_trace.csv"),
        "search": read_json_lines(root / "agent_search_trace.log", JSON_PREFIXES["agent_search_trace.log"]),
        "wrapper": read_json_lines(root / "random_util_wrapper_trace.log", JSON_PREFIXES["random_util_wrapper_trace.log"]),
    }


def actual_searches(run):
    rows = []
    for row in run["search"]:
        actor_class = row.get("actor_class", "")
        if (
            "ActionCounterfactualTrainer$ActionPlayer" not in actor_class
            and "mage.player.ai.ComputerPlayerRL" not in actor_class
        ):
            continue
        rows.append(row)
    return rows


def search_summary_rows(run):
    rows = []
    for row in actual_searches(run):
        rows.append({
            "seq": row.get("seq", ""),
            "source": row.get("source_name", ""),
            "turn": row.get("turn", ""),
            "actor_class": row.get("actor_class", "").split(".")[-1],
            "before": safe_int(row.get("random_util_count_before_shuffle")),
            "after": safe_int(row.get("random_util_count_after_shuffle")),
            "delta": safe_int(row.get("random_util_delta_shuffle")),
            "chosen": first_join(row.get("chosen_names")),
            "top_before": first_join(row.get("library_top_before_shuffle", [])[:4]),
            "top_after": first_join(row.get("library_top_after_shuffle", [])[:4]),
        })
    return rows


def find_search(rows, source):
    for row in rows:
        if row["source"] == source:
            return row
    return None


def wrapper_between(run, start_inclusive, end_exclusive):
    rows = []
    for row in run["wrapper"]:
        before = safe_int(row.get("wrapper_counter_before"))
        after = safe_int(row.get("wrapper_counter_after"))
        if before is None or after is None:
            continue
        if before >= start_inclusive and after <= end_exclusive and row.get("counted_global_stream", True):
            rows.append(row)
    return rows


def group_wrapper(rows):
    grouped = defaultdict(lambda: {"count": 0, "first": None, "last": None, "bounds": Counter(), "sources": Counter()})
    for row in rows:
        key = row.get("caller", "")
        entry = grouped[key]
        entry["count"] += safe_int(row.get("wrapper_counter_delta"), 0) or 0
        entry["first"] = row if entry["first"] is None else entry["first"]
        entry["last"] = row
        entry["bounds"][row.get("detail", "")] += 1
        entry["sources"][row.get("source_name", "")] += 1
        entry.setdefault("threads", Counter())[row.get("thread", "")] += 1
    return grouped


def format_counter(counter, max_items=4):
    parts = []
    for key, value in counter.most_common(max_items):
        label = key if key else "<blank>"
        parts.append(f"{label} x{value}")
    return ", ".join(parts)


def sample_summary(run):
    if not run["sample"]:
        return {}
    row = run["sample"][0]
    keys = [
        "ordinal",
        "source_decision_number",
        "action_type",
        "actual_text",
        "matched",
        "timed_out",
        "actual_decisions",
        "forced_prefix_count",
        "prefix_failure_ordinal",
        "prefix_failure_reason",
        "error",
    ]
    return {key: row.get(key, "") for key in keys}


def forced_decision_rows(run, decisions):
    by_decision = {}
    for row in run["forced"]:
        decision = row.get("source_decision_number", "")
        if decision in decisions:
            by_decision[decision] = row
    return by_decision


def markdown_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(out)


def build_report(label_a, run_a, label_b, run_b):
    a_search = search_summary_rows(run_a)
    b_search = search_summary_rows(run_b)
    report = []
    report.append(f"# Replay RNG Trace Delta: {label_a} vs {label_b}")
    report.append("")
    report.append("## Replay Sample Summary")
    sample_rows = []
    for label, run in [(label_a, run_a), (label_b, run_b)]:
        summary = sample_summary(run)
        summary["run"] = label
        sample_rows.append(summary)
    report.append(markdown_table(
        ["run", "ordinal", "source_decision_number", "action_type", "actual_text", "matched", "timed_out", "actual_decisions", "forced_prefix_count", "prefix_failure_ordinal", "prefix_failure_reason", "error"],
        sample_rows,
    ))
    report.append("")

    report.append("## ActionPlayer Search/Shuffle Windows")
    search_rows = []
    for label, rows in [(label_a, a_search), (label_b, b_search)]:
        for row in rows:
            row = dict(row)
            row["run"] = label
            search_rows.append(row)
    report.append(markdown_table(
        ["run", "seq", "source", "turn", "before", "after", "delta", "chosen", "top_before", "top_after"],
        search_rows,
    ))
    report.append("")

    decisions = ["39", "40", "41", "42"]
    report.append("## Forced Prefix Rows Around Gatecreeper")
    forced_rows = []
    for label, run in [(label_a, run_a), (label_b, run_b)]:
        rows = forced_decision_rows(run, decisions)
        for decision in decisions:
            row = rows.get(decision)
            if not row:
                continue
            forced_rows.append({
                "run": label,
                "decision": decision,
                "ordinal": row.get("ordinal", ""),
                "expected_action_type": row.get("expected_action_type", ""),
                "actual_action_type": row.get("actual_action_type", ""),
                "expected_indices": row.get("expected_indices", ""),
                "actual_selected_indices": row.get("actual_selected_indices", ""),
                "actual_selected_texts": row.get("actual_selected_texts", ""),
                "stack_before": row.get("stack_before", ""),
                "stack_after": row.get("stack_after", ""),
            })
    report.append(markdown_table(
        ["run", "decision", "ordinal", "expected_action_type", "actual_action_type", "expected_indices", "actual_selected_indices", "actual_selected_texts", "stack_before", "stack_after"],
        forced_rows,
    ))
    report.append("")

    source_after = "Generous Ent"
    gate_source = "Gatecreeper Vine"
    report.append("## Wrapper Consumption Between Forestcycling And Gatecreeper")
    interval_rows = []
    grouped_sections = []
    for label, run, rows in [(label_a, run_a, a_search), (label_b, run_b, b_search)]:
        after_row = find_search(rows, source_after)
        gate_row = find_search(rows, gate_source)
        if not after_row or not gate_row:
            interval_rows.append({"run": label, "start": "missing", "end": "missing", "count": "missing"})
            continue
        start = after_row["after"]
        end = gate_row["before"]
        wrappers = wrapper_between(run, start, end)
        interval_rows.append({
            "run": label,
            "start": start,
            "end": end,
            "count": sum((safe_int(r.get("wrapper_counter_delta"), 0) or 0) for r in wrappers),
            "wrapper_rows": len(wrappers),
        })
        grouped_sections.append((label, start, end, group_wrapper(wrappers)))
    report.append(markdown_table(["run", "start", "end", "count", "wrapper_rows"], interval_rows))
    report.append("")

    for label, start, end, grouped in grouped_sections:
        report.append(f"### {label} wrapper groups [{start}, {end})")
        group_rows = []
        for caller, data in sorted(grouped.items(), key=lambda item: (-item[1]["count"], item[0])):
            first = data["first"]
            last = data["last"]
            group_rows.append({
                "caller": caller,
                "count": data["count"],
                "first_before": first.get("wrapper_counter_before", "") if first else "",
                "last_after": last.get("wrapper_counter_after", "") if last else "",
                "bounds": format_counter(data["bounds"]),
                "source_names": format_counter(data["sources"]),
                "threads": format_counter(data["threads"]),
            })
        report.append(markdown_table(["caller", "count", "first_before", "last_after", "bounds", "threads", "source_names"], group_rows))
        report.append("")

    a_gate = find_search(a_search, gate_source)
    b_gate = find_search(b_search, gate_source)
    if a_gate and b_gate:
        before_delta = b_gate["before"] - a_gate["before"]
        after_delta = b_gate["after"] - a_gate["after"]
        report.append("## Observed Delta")
        report.append("")
        report.append(f"- {label_a} Gatecreeper window: {a_gate['before']}->{a_gate['after']}.")
        report.append(f"- {label_b} Gatecreeper window: {b_gate['before']}->{b_gate['after']}.")
        report.append(f"- {label_b} is {abs(before_delta)} counts {'behind' if before_delta < 0 else 'ahead'} at Gatecreeper start and {abs(after_delta)} counts {'behind' if after_delta < 0 else 'ahead'} after the shuffle.")
    return "\n".join(report).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-label", default="v45")
    parser.add_argument("--left-root", required=True)
    parser.add_argument("--right-label", default="v47")
    parser.add_argument("--right-root", required=True)
    parser.add_argument("--markdown-out")
    args = parser.parse_args()

    left = load_run(args.left_root)
    right = load_run(args.right_root)
    report = build_report(args.left_label, left, args.right_label, right)
    if args.markdown_out:
        out_path = Path(args.markdown_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
