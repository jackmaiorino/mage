#!/usr/bin/env python3
"""Print a compact action trace from MTGRL GameLogger text logs.

This is a read-only viewer for manual inspection. It accepts either full logs
or compact logs and writes a shorter trace to stdout.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


DECISION_RE = re.compile(r"DECISION #(?P<num>\d+) - Turn (?P<turn>\d+) \((?P<owner>.*?) turn\), (?P<phase>.*?) - (?P<actor>.*)$")
FULL_OPTION_RE = re.compile(r"\[(?P<idx>\d+)\]\s+(?P<prob>[+-]?\d+(?:\.\d+)?)\s+-\s+(?P<action>.*)$")
FULL_ACTION_RE = re.compile(r"\[Turn (?P<turn>\d+), (?P<phase>.*?)\]\s+(?P<actor>.*?):\s+(?P<action>.*)$")


@dataclass
class Option:
    idx: int
    prob: float
    action: str
    selected: bool = False


@dataclass
class Decision:
    header: str
    num: str
    turn: str
    owner: str
    phase: str
    actor: str
    selected: str = ""
    value: str = ""
    selected_idx: Optional[int] = None
    selected_prob: str = ""
    compact_state: str = ""
    compact_top: str = ""
    exploration: str = ""
    state_lines: List[str] = field(default_factory=list)
    options: List[Option] = field(default_factory=list)


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r", " ").replace("\n", " ")).strip()


def trunc(text: str, max_chars: int) -> str:
    text = one_line(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def split_cards(value: str) -> List[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def strip_outer_brackets(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        return raw[1:-1].strip()
    return raw


def compact_zone(label: str, raw: str, zone_chars: int) -> str:
    value = strip_outer_brackets(raw)
    hidden = re.fullmatch(r"(\d+) cards", value)
    if hidden:
        return f"{label}{hidden.group(1)}"
    if not value:
        return f"{label}0"
    cards = split_cards(value)
    return f"{label}{len(cards)}[{trunc(value, zone_chars)}]"


def compact_state_from_full(lines: List[str], zone_chars: int) -> str:
    parts: List[str] = []
    current: Optional[dict] = None
    stack_text = ""

    def flush_current() -> None:
        if not current:
            return
        parts.append(
            f"{current['name']} L{current['life']} "
            f"{current.get('H', 'H?')} {current.get('B', 'B?')} "
            f"{current.get('G', 'G?')} {current.get('X', 'X?')}"
        )

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("[Exploration]"):
            continue
        if line.startswith("[Stack]:"):
            stack_text = "stack=" + line[len("[Stack]:") :].strip()
            continue
        if line.startswith("- [") and stack_text and " top=" not in stack_text:
            stack_text += " top=" + trunc(re.sub(r"^-\s*", "", line), 90)
            continue
        player = re.match(r"\[(?P<name>[^\]]+)\], life = (?P<life>.*)$", line)
        if player:
            flush_current()
            current = {"name": player.group("name"), "life": player.group("life").strip()}
            continue
        if current and line.startswith("-> "):
            zone, sep, raw_value = line[3:].partition(":")
            if not sep:
                continue
            zone = zone.strip()
            raw_value = raw_value.strip()
            if zone == "Hand":
                current["H"] = compact_zone("H", raw_value, zone_chars)
            elif zone == "Permanents":
                current["B"] = compact_zone("B", raw_value, zone_chars)
            elif zone == "Graveyard":
                current["G"] = compact_zone("G", raw_value, zone_chars)
            elif zone == "Exile":
                current["X"] = compact_zone("X", raw_value, zone_chars)

    flush_current()
    if stack_text:
        parts.insert(0, stack_text)
    return " || ".join(parts)


def parse_decision_header(line: str) -> Optional[Decision]:
    m = DECISION_RE.match(line.strip())
    if not m:
        return None
    return Decision(
        header=line.strip(),
        num=m.group("num"),
        turn=m.group("turn"),
        owner=m.group("owner"),
        phase=m.group("phase"),
        actor=m.group("actor"),
    )


def parse_full_option(line: str) -> Optional[Option]:
    text = line.strip()
    selected = text.startswith(">>>")
    if selected:
        text = text[3:].strip()
    m = FULL_OPTION_RE.match(text)
    if not m:
        return None
    return Option(
        idx=int(m.group("idx")),
        prob=float(m.group("prob")),
        action=m.group("action").strip(),
        selected=selected,
    )


def selected_option(decision: Decision) -> Optional[Option]:
    for option in decision.options:
        if option.selected:
            return option
    if decision.selected_idx is not None:
        for option in decision.options:
            if option.idx == decision.selected_idx:
                return option
    selected = one_line(decision.selected)
    if selected:
        for option in decision.options:
            if one_line(option.action) == selected:
                return option
    return None


def top_options(decision: Decision, limit: int, action_chars: int) -> str:
    if decision.compact_top:
        return decision.compact_top
    if not decision.options:
        return ""
    selected = selected_option(decision)
    chosen: List[Option] = []
    if selected is not None:
        chosen.append(selected)
    for option in sorted(decision.options, key=lambda item: item.prob, reverse=True):
        if len(chosen) >= max(1, limit):
            break
        if all(existing.idx != option.idx for existing in chosen):
            chosen.append(option)
    parts = [f"n={len(decision.options)}"]
    selected_idx = selected.idx if selected else None
    for option in chosen:
        mark = "*" if option.idx == selected_idx else ""
        parts.append(f"{mark}[{option.idx}] {option.prob:.4f} {trunc(option.action, action_chars)}")
    return " | ".join(parts)


def emit_decision(decision: Decision, top_n: int, zone_chars: int, action_chars: int) -> None:
    selected = selected_option(decision)
    idx = decision.selected_idx
    prob = decision.selected_prob
    action = decision.selected
    if selected is not None:
        idx = selected.idx
        prob = f"{selected.prob:.4f}"
        action = selected.action
    idx_text = "?" if idx is None else str(idx)
    prob_text = prob or "n/a"
    value_text = decision.value or "n/a"
    print(f"D{int(decision.num):03d} T{decision.turn} {decision.owner}-turn actor={decision.actor} phase={decision.phase}")
    print(f"  selected[{idx_text}] p={prob_text} value={value_text}: {trunc(action, action_chars)}")
    state = decision.compact_state or compact_state_from_full(decision.state_lines, zone_chars)
    if state:
        print(f"  state: {state}")
    tops = top_options(decision, top_n, action_chars)
    if tops:
        print(f"  top: {tops}")
    if decision.exploration:
        print(f"  {decision.exploration}")


def resolve_paths(inputs: List[str], latest: int) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.rglob("game_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
            paths.extend(found[:latest] if latest > 0 else found)
        else:
            paths.append(path)
    return paths


def view_file(path: Path, top_n: int, zone_chars: int, action_chars: int) -> None:
    print(f"== {path} ==")
    current: Optional[Decision] = None
    mode = ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print(f"Could not read {path}: {exc}", file=sys.stderr)
        return

    for line in lines:
        decision = parse_decision_header(line)
        if decision:
            if current:
                emit_decision(current, top_n, zone_chars, action_chars)
            current = decision
            mode = ""
            continue
        if current is None:
            action = FULL_ACTION_RE.match(line.strip())
            if action:
                print(
                    f"ACTION T{action.group('turn')} {action.group('phase')} "
                    f"{action.group('actor')}: {trunc(action.group('action'), action_chars)}"
                )
            elif line.startswith("ACTION T"):
                print(trunc(line, action_chars + 40))
            elif line.startswith("GAME OUTCOME") or line.startswith("Winner:") or line.startswith("Reason:"):
                print(line.strip())
            continue

        stripped = line.strip()
        if stripped == "GAME STATE:":
            mode = "state"
            continue
        if stripped == "OPTIONS & SCORES:":
            mode = "options"
            continue
        if stripped.startswith("SELECTED["):
            m = re.match(r"SELECTED\[(?P<idx>-?\d+)\]\s+p=(?P<prob>\S+)\s+value=(?P<value>\S+):\s+(?P<action>.*)", stripped)
            if m:
                try:
                    current.selected_idx = int(m.group("idx"))
                except ValueError:
                    current.selected_idx = None
                current.selected_prob = m.group("prob")
                current.value = m.group("value")
                current.selected = m.group("action")
            continue
        if stripped.startswith("STATE:"):
            current.compact_state = stripped[len("STATE:") :].strip()
            continue
        if stripped.startswith("TOP:"):
            current.compact_top = stripped[len("TOP:") :].strip()
            continue
        if stripped.startswith("[Exploration]"):
            current.exploration = trunc(stripped, 220)
            continue
        if stripped.startswith("VALUE SCORE:"):
            current.value = stripped[len("VALUE SCORE:") :].strip()
            continue
        if stripped.startswith("SELECTED:"):
            current.selected = stripped[len("SELECTED:") :].strip()
            continue
        if mode == "state":
            if stripped.startswith("-") or stripped.startswith("[") or stripped.startswith("->"):
                current.state_lines.append(line)
            continue
        if mode == "options":
            option = parse_full_option(line)
            if option:
                current.options.append(option)
            continue

        action = FULL_ACTION_RE.match(stripped)
        if action:
            emit_decision(current, top_n, zone_chars, action_chars)
            current = None
            mode = ""
            print(
                f"ACTION T{action.group('turn')} {action.group('phase')} "
                f"{action.group('actor')}: {trunc(action.group('action'), action_chars)}"
            )
        elif stripped.startswith("ACTION T"):
            emit_decision(current, top_n, zone_chars, action_chars)
            current = None
            mode = ""
            print(trunc(stripped, action_chars + 40))
        elif stripped.startswith("GAME OUTCOME"):
            emit_decision(current, top_n, zone_chars, action_chars)
            current = None
            mode = ""
            print(stripped)

    if current:
        emit_decision(current, top_n, zone_chars, action_chars)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Game log files or directories containing game_*.txt logs.")
    parser.add_argument("--latest", type=int, default=1, help="When a path is a directory, read this many newest logs.")
    parser.add_argument("--top-options", type=int, default=5)
    parser.add_argument("--zone-chars", type=int, default=96)
    parser.add_argument("--action-chars", type=int, default=180)
    args = parser.parse_args(argv)

    for path in resolve_paths(args.paths, args.latest):
        view_file(path, args.top_options, args.zone_chars, args.action_chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
