#!/usr/bin/env python3
"""Build a replay/search anchor manifest from mined Affinity failure examples.

The compact game logs are useful for choosing exact failure contexts, but they
do not yet contain enough deterministic replay metadata for
ActionCounterfactualTrainer's replay CSV format. This helper turns the mined
pressure examples into a durable manifest that names the log, decision ordinal,
compact state, top policy alternatives, and the specific metadata still missing
before a terminal-prefix search can target the historical state.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_CORPUS_DIR = Path(
    "local-training/local_pbt/corpora/20260518_affinity_failure_context"
)

DECISION_RE = re.compile(r"^DECISION #(\d+)\b")
HEADER_RE = re.compile(r"^([^:]+):\s*(.*)$")
REPLAY_KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=")


@dataclasses.dataclass(frozen=True)
class DecisionContext:
    line_number: int
    decision_line: str
    selected_line: str
    state_line: str
    top_line: str
    value_line: str
    suffix: List[str]


@dataclasses.dataclass(frozen=True)
class Anchor:
    anchor_id: str
    rank: int
    category: str
    replay_ready: bool
    replay_blocker: str
    replay_scenario: str
    replay_seed: str
    replay_agent_deck: str
    replay_opp_deck: str
    replay_metadata_line: str
    game_path: str
    decision_number: int
    log_line: int
    turn: int
    phase: str
    selected_class: str
    selected: str
    selected_prob: str
    top_best_prob: str
    top_best_text: str
    opp_permanents: int
    own_graveyard_count: int
    own_hand_count: int
    own_life: int
    opp_life: int
    has_cast_spy_option: bool
    has_dread_return_option: bool
    selected_in_hand: bool
    selected_in_graveyard: bool
    dread_return_in_hand: bool
    dread_return_in_graveyard: bool
    agent_opening_hand: str
    game_started: str
    matchup: str
    mode: str
    decision_line: str
    selected_line: str
    state_line: str
    top_line: str
    value_line: str
    suffix: str
    recommended_hook: str


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def contains_card(card_list: str, card_name: str) -> bool:
    if not card_list or not card_name:
        return False
    needle = card_name.strip().lower()
    if needle.startswith("cast "):
        needle = needle[5:].strip()
    return any(part.strip().lower() == needle for part in card_list.split(";"))


def parse_replay_metadata(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    matches = list(REPLAY_KV_RE.finditer(line or ""))
    for idx, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line or "")
        out[match.group(1).strip()] = (line or "")[value_start:value_end].strip()
    return out


def load_pressure_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_log(path: Path, lookahead_decisions: int) -> Tuple[Dict[str, str], Dict[int, DecisionContext]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header: Dict[str, str] = {}
    decisions: Dict[int, Tuple[int, int]] = {}

    for idx, line in enumerate(lines):
        if idx < 20:
            if line.startswith("MODE="):
                header["mode"] = line.split("=", 1)[1].strip()
            m = HEADER_RE.match(line)
            if m:
                header[m.group(1).strip().lower()] = m.group(2).strip()
        m = DECISION_RE.match(line)
        if m:
            decisions[int(m.group(1))] = (idx, idx + 1)

    ordered = sorted(decisions)
    contexts: Dict[int, DecisionContext] = {}
    for pos, number in enumerate(ordered):
        start = decisions[number][0]
        end = decisions[ordered[pos + 1]][0] if pos + 1 < len(ordered) else len(lines)
        block = lines[start:end]
        selected_line = next((line.strip() for line in block if line.strip().startswith("SELECTED[")), "")
        state_line = next((line.strip() for line in block if line.strip().startswith("STATE:")), "")
        top_line = next((line.strip() for line in block if line.strip().startswith("TOP:")), "")
        value_line = next((line.strip() for line in block if line.strip().startswith("VALUE SCORE:")), "")
        suffix: List[str] = []
        suffix_end = min(len(ordered), pos + 1 + lookahead_decisions)
        for suffix_pos in range(pos + 1, suffix_end):
            next_number = ordered[suffix_pos]
            next_start = decisions[next_number][0]
            next_end = decisions[ordered[suffix_pos + 1]][0] if suffix_pos + 1 < len(ordered) else len(lines)
            next_block = lines[next_start:next_end]
            next_selected = next((line.strip() for line in next_block if line.strip().startswith("SELECTED[")), "")
            suffix.append(f"D{next_number:03d}: {next_selected}" if next_selected else f"D{next_number:03d}")
        contexts[number] = DecisionContext(
            line_number=start + 1,
            decision_line=lines[start].strip(),
            selected_line=selected_line,
            state_line=state_line,
            top_line=top_line,
            value_line=value_line,
            suffix=suffix,
        )
    return header, contexts


def opening_hand_from_log(path: Path) -> str:
    keep_hand = ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:80]:
        if line.startswith("MULLIGAN_DECISION:") and "decision=KEEP" in line:
            m = re.search(r"hand=\[(.*)]", line)
            if m:
                keep_hand = m.group(1).strip()
        if line.startswith("TURN 1 START") and keep_hand:
            break
    return keep_hand


def classify(row: Dict[str, str]) -> str:
    selected_class = row.get("selected_class", "")
    selected = row.get("selected", "")
    hand = row.get("own_hand_cards", "")
    graveyard = row.get("own_graveyard", "")
    phase = row.get("phase", "")
    if selected_class == "cast_dread_return" and contains_card(hand, "Dread Return") and not contains_card(graveyard, "Dread Return"):
        return "hard_cast_dread_return_from_hand"
    if selected_class == "flashback":
        return "flashback_followon_target_sequence"
    if "TARGET_PICK" in phase:
        return "target_pick_under_pressure"
    if selected_class == "cast_spy":
        return "spy_cast_under_pressure"
    if selected_class == "cast_other":
        return "setup_spell_under_pressure"
    if selected_class == "mana_ability":
        return "mana_sequence_under_pressure"
    if selected_class == "pass":
        return "pass_under_pressure"
    if contains_card(hand, selected) or contains_card(graveyard, selected):
        return "zone_sensitive_choice"
    return "other_pressure_choice"


def category_priority(category: str) -> int:
    order = {
        "hard_cast_dread_return_from_hand": 0,
        "flashback_followon_target_sequence": 1,
        "target_pick_under_pressure": 2,
        "spy_cast_under_pressure": 3,
        "setup_spell_under_pressure": 4,
        "mana_sequence_under_pressure": 5,
        "zone_sensitive_choice": 6,
        "pass_under_pressure": 7,
        "other_pressure_choice": 8,
    }
    return order.get(category, 99)


def build_anchors(
    rows: Sequence[Dict[str, str]],
    repo: Path,
    max_anchors: int,
    min_opp_permanents: int,
    lookahead_decisions: int,
) -> List[Anchor]:
    filtered: List[Tuple[int, Dict[str, str], str]] = []
    for idx, row in enumerate(rows, start=1):
        if safe_int(row.get("opp_permanents")) < min_opp_permanents:
            continue
        category = classify(row)
        filtered.append((idx, row, category))

    filtered.sort(
        key=lambda item: (
            category_priority(item[2]),
            -safe_int(item[1].get("opp_permanents")),
            float(item[1].get("top_selected_prob") or "1.0"),
            item[0],
        )
    )

    log_cache: Dict[Path, Tuple[Dict[str, str], Dict[int, DecisionContext], str]] = {}
    anchors: List[Anchor] = []
    for source_rank, row, category in filtered[:max_anchors]:
        rel_game_path = Path(row.get("game_path", ""))
        game_path = rel_game_path if rel_game_path.is_absolute() else repo / rel_game_path
        if game_path not in log_cache:
            header, contexts = read_log(game_path, lookahead_decisions)
            opening = opening_hand_from_log(game_path)
            log_cache[game_path] = (header, contexts, opening)
        header, contexts, opening = log_cache[game_path]
        decision_number = safe_int(row.get("decision_number"))
        context = contexts.get(decision_number, DecisionContext(0, "", "", "", "", "", []))
        selected = row.get("selected", "")
        hand = row.get("own_hand_cards", "")
        graveyard = row.get("own_graveyard", "")
        anchor_id = f"{game_path.stem}_D{decision_number:03d}"
        replay_line = header.get("replay", "")
        replay_meta = parse_replay_metadata(replay_line)
        replay_ready = (
            replay_meta.get("action_counterfactual_compatible", "").lower() == "true"
            and bool(replay_meta.get("scenario"))
            and bool(replay_meta.get("seed"))
            and bool(replay_meta.get("agent_deck"))
            and bool(replay_meta.get("opp_deck"))
        )
        anchors.append(
            Anchor(
                anchor_id=anchor_id,
                rank=source_rank,
                category=category,
                replay_ready=replay_ready,
                replay_blocker="" if replay_ready else (
                    "compact log has decision ordinal and opening hand, but no ActionCounterfactual scenario seed "
                    "or restorable engine snapshot"
                ),
                replay_scenario=replay_meta.get("scenario", ""),
                replay_seed=replay_meta.get("seed", ""),
                replay_agent_deck=replay_meta.get("agent_deck", ""),
                replay_opp_deck=replay_meta.get("opp_deck", ""),
                replay_metadata_line=replay_line,
                game_path=str(game_path),
                decision_number=decision_number,
                log_line=context.line_number,
                turn=safe_int(row.get("turn")),
                phase=row.get("phase", ""),
                selected_class=row.get("selected_class", ""),
                selected=selected,
                selected_prob=row.get("top_selected_prob", ""),
                top_best_prob=row.get("top_best_prob", ""),
                top_best_text=row.get("top_best_text", ""),
                opp_permanents=safe_int(row.get("opp_permanents")),
                own_graveyard_count=safe_int(row.get("own_graveyard_count")),
                own_hand_count=safe_int(row.get("own_hand_count")),
                own_life=safe_int(row.get("own_life")),
                opp_life=safe_int(row.get("opp_life")),
                has_cast_spy_option=str(row.get("has_cast_spy_option", "")).lower() == "true",
                has_dread_return_option=str(row.get("has_dread_return_option", "")).lower() == "true",
                selected_in_hand=contains_card(hand, selected),
                selected_in_graveyard=contains_card(graveyard, selected),
                dread_return_in_hand=contains_card(hand, "Dread Return"),
                dread_return_in_graveyard=contains_card(graveyard, "Dread Return"),
                agent_opening_hand=opening,
                game_started=header.get("started", ""),
                matchup=header.get("matchup", ""),
                mode=header.get("mode", ""),
                decision_line=context.decision_line,
                selected_line=context.selected_line,
                state_line=context.state_line,
                top_line=context.top_line,
                value_line=context.value_line,
                suffix=" || ".join(context.suffix),
                recommended_hook=(
                    "export per-decision scenario seed or snapshot key, then run terminal-prefix search from this "
                    "ordinal with generic branch order and terminal-win-only label admission"
                ),
            )
        )
    return anchors


def write_jsonl(path: Path, anchors: Iterable[Anchor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for anchor in anchors:
            fh.write(json.dumps(dataclasses.asdict(anchor), ensure_ascii=True) + "\n")


def write_csv(path: Path, anchors: Sequence[Anchor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field.name for field in dataclasses.fields(Anchor)]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for anchor in anchors:
            writer.writerow(dataclasses.asdict(anchor))


def write_readme(path: Path, anchors: Sequence[Anchor], examples_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = Counter(anchor.category for anchor in anchors)
    lines = [
        "# Affinity Replay Anchor Manifest",
        "",
        f"Source examples: `{examples_csv}`",
        f"Anchors: {len(anchors)}",
        "",
        "## Category Counts",
        "",
    ]
    for category, count in categories.most_common():
        lines.append(f"- `{category}`: {count}")
    ready_count = sum(1 for anchor in anchors if anchor.replay_ready)
    lines.extend(["", "## Replay Readiness", "", f"Replay-ready anchors: {ready_count}/{len(anchors)}", ""])
    if ready_count == len(anchors) and anchors:
        lines.extend(
            [
                "These anchors have ActionCounterfactual-compatible scenario, seed, and deck metadata. "
                "They still need conversion into replay CSV rows with target action types and candidate text.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "These anchors are not yet direct `ActionCounterfactualTrainer --replay-file` rows. "
                "The compact logs preserve the decision ordinal, selected action, top policy choices, "
                "opening hand, and compact state, but older logs do not preserve the scenario seed or a "
                "restorable engine snapshot key.",
                "",
                "Next hook: collect accepted-policy compact logs with `scripts/run_cp7_eval_sweep.py "
                "--eval-game-logging --game-log-format compact --replay-metadata`, then feed those "
                "`anchor_id`/ordinal pairs into generic terminal-prefix search and only admit labels "
                "whose forced suffix reaches a terminal win.",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Anchors",
            "",
            "| Anchor | Category | Decision | Selected | Pressure | Log Line |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for anchor in anchors[:12]:
        selected = anchor.selected.replace("|", "/")
        lines.append(
            f"| `{anchor.anchor_id}` | `{anchor.category}` | {anchor.decision_number} | "
            f"{selected} | {anchor.opp_permanents} | {anchor.log_line} |"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--examples-csv",
        default=str(DEFAULT_CORPUS_DIR / "pressure_failure_examples.csv"),
        help="Input pressure failure examples CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_CORPUS_DIR),
        help="Output directory for replay_anchor_manifest.{jsonl,csv,md}.",
    )
    parser.add_argument("--max-anchors", type=int, default=20)
    parser.add_argument("--min-opp-permanents", type=int, default=0)
    parser.add_argument("--lookahead-decisions", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    examples_csv = Path(args.examples_csv)
    out_dir = Path(args.out_dir)
    rows = load_pressure_rows(examples_csv)
    anchors = build_anchors(
        rows=rows,
        repo=repo,
        max_anchors=max(1, args.max_anchors),
        min_opp_permanents=max(0, args.min_opp_permanents),
        lookahead_decisions=max(0, args.lookahead_decisions),
    )
    write_jsonl(out_dir / "replay_anchor_manifest.jsonl", anchors)
    write_csv(out_dir / "replay_anchor_manifest.csv", anchors)
    write_readme(out_dir / "replay_anchor_manifest.md", anchors, examples_csv)
    print(
        json.dumps(
            {
                "anchors": len(anchors),
                "jsonl": str(out_dir / "replay_anchor_manifest.jsonl"),
                "csv": str(out_dir / "replay_anchor_manifest.csv"),
                "readme": str(out_dir / "replay_anchor_manifest.md"),
                "categories": dict(Counter(anchor.category for anchor in anchors)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
