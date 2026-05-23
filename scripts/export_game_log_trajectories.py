#!/usr/bin/env python3
"""Export MTGRL text game logs into compact trajectory JSONL.

This is intended for offline credit-assignment probes. It does not create
training labels by itself; it preserves terminal outcomes and per-decision
metadata so a later script can test synthetic-return / return-decomposition
ideas without modifying the online trainer.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


GAME_RE = re.compile(r"^GAME LOG:\s+(.+?)\s*$")
OPPONENT_RE = re.compile(r"^OPPONENT:\s+(.+?)\s*$")
MATCHUP_RE = re.compile(r"^MATCHUP:\s+agent=(.+?)\s+vs\s+opp=(.+?)\s*$")
MULLIGAN_RE = re.compile(
    r"^MULLIGAN_DECISION:\s+player=(.*?)\s+mulligansTaken=(\d+)\s+handSize=(\d+)\s+"
    r"lands=(\d+)\s+decision=(KEEP|MULLIGAN)\s+P_keep=([-0-9.]+)\s+P_mull=([-0-9.]+)"
    r"(?:\s+hand=\[(.*)\])?\s*$"
)
LONDON_RE = re.compile(
    r"^LONDON_BOTTOM:\s+player=(.*?)\s+mulligansTaken=(\d+)\s+handSize=(\d+)\s+"
    r"bottomN=(\d+)\s+kept=\[(.*?)\]\s+bottomed=\[(.*?)\]\s*$"
)
DECISION_RE = re.compile(r"^DECISION #(\d+) - Turn (\d+) \((.*?) turn\), (.*?) - (.*)$")
OPTION_RE = re.compile(r"^\s*(>>>\s*)?\[(\d+)\]\s+([-0-9.]+)\s+-\s+(.+?)\s*$")
COMPACT_TOP_RE = re.compile(r"^\s*TOP:\s+(.+?)\s*$")
COMPACT_TOP_OPTION_RE = re.compile(r"^(\*)?\[(\d+)\]\s+([-0-9.]+)\s+(.+?)\s*$")
VALUE_RE = re.compile(r"^VALUE SCORE:\s+([-0-9.]+)\s*$")
SELECTED_RE = re.compile(r"^SELECTED:\s+(.+?)\s*$")
WINNER_RE = re.compile(r"^Winner:\s+(.+?)\s*$")
LOSER_RE = re.compile(r"^Loser:\s+(.+?)\s*$")
RESULT_RE = re.compile(r"^RESULT:\s+(WIN|LOSS)\s*$")
TURNS_RE = re.compile(r"^Turns:\s+(\d+)\s*$")
REASON_RE = re.compile(r"^Reason:\s+(.+?)\s*$")
PLAYER_RE = re.compile(r"^\[(.+?)\], life = (-?\d+)")
ZONE_RE = re.compile(r"^-> (Hand|Permanents|Graveyard|Exile): \[(.*)\]$")
COMPACT_STACK_RE = re.compile(r"^stack=(\d+) items(?:\s+top=\[(\d+)\]\s+(.+))?$")
COMPACT_PLAYER_RE = re.compile(r"^(.+?)\s+L(-?\d+)\s+(.*)$")
COMPACT_ZONE_RE = re.compile(r"\b([HBGX])(\d+)(?:\[([^\]]*)\])?")


def iter_logs(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            yield root
        elif root.exists():
            yield from sorted(root.rglob("*.txt"))


def split_cards(raw: str) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(";") if part.strip()]


def action_flags(text: str, options: Optional[List[Dict[str, object]]] = None) -> Dict[str, bool]:
    opts = [str(o.get("text", "")) for o in (options or [])]
    return {
        "selected_pass": text == "Pass",
        "selected_play_land": text.startswith("Play ") and not text.startswith("Play with"),
        "selected_cast_spy": text.startswith("Cast Balustrade Spy"),
        "selected_cast_dread_return": text.startswith("Cast Dread Return"),
        "selected_land_grant": text.startswith("Cast Land Grant"),
        "selected_mana_ability": "Add {" in text,
        "has_land_play_option": any(o.startswith("Play ") and not o.startswith("Play with") for o in opts),
        "has_cast_spy_option": any(o.startswith("Cast Balustrade Spy") for o in opts),
        "has_dread_return_option": any(o.startswith("Cast Dread Return") for o in opts),
        "has_nonpass_option": any(o != "Pass" for o in opts),
    }


def parse_compact_top(raw: str) -> List[Dict[str, object]]:
    """Parse compact `TOP:` option summaries emitted by GameLogger.

    Compact logs only show the selected option plus the highest-probability
    alternatives, so this is not a full legal-action list. It is still enough
    for diagnostics that need to identify visible alternatives in terse logs.
    """
    options: List[Dict[str, object]] = []
    seen = set()
    for part in raw.split(" | "):
        part = part.strip()
        if not part or part.startswith("n="):
            continue
        match = COMPACT_TOP_OPTION_RE.match(part)
        if not match:
            continue
        marker, idx, prob, text = match.groups()
        index = int(idx)
        if index in seen:
            continue
        seen.add(index)
        options.append({
            "index": index,
            "prob": float(prob),
            "selected_marker": bool(marker),
            "text": text,
        })
    return options


def summarize_state(lines: List[str]) -> Dict[str, object]:
    """Return a small visible-state summary from a GAME STATE block."""
    players: Dict[str, Dict[str, object]] = {}
    stack: Dict[str, object] = {}
    current_player: Optional[str] = None
    for line in lines:
        compact = summarize_compact_state(line)
        if compact:
            return compact
        player = PLAYER_RE.match(line)
        if player:
            current_player = player.group(1)
            players.setdefault(current_player, {})["life"] = int(player.group(2))
            continue
        zone = ZONE_RE.match(line)
        if zone and current_player:
            cards = split_cards(zone.group(2))
            key = zone.group(1).lower()
            pdata = players.setdefault(current_player, {})
            pdata[f"{key}_count"] = len(cards)
            if key in {"hand", "graveyard", "permanents"}:
                pdata[f"{key}_cards"] = [c.split(",", 1)[0] for c in cards[:24]]
    return {"players": players, "stack": stack}


def summarize_compact_state(line: str) -> Dict[str, object]:
    """Return a visible-state summary from a compact `STATE:` line."""
    text = line.strip()
    if text.startswith("STATE:"):
        text = text[len("STATE:"):].strip()
    if not text.startswith("stack="):
        return {}

    parts = [part.strip() for part in text.split(" || ") if part.strip()]
    stack: Dict[str, object] = {}
    players: Dict[str, Dict[str, object]] = {}
    if parts:
        stack_match = COMPACT_STACK_RE.match(parts[0])
        if stack_match:
            count, top_index, top_text = stack_match.groups()
            stack["count"] = int(count)
            if top_index is not None:
                stack["top_index"] = int(top_index)
            if top_text:
                stack["top_text"] = top_text

    zone_keys = {
        "H": "hand",
        "B": "permanents",
        "G": "graveyard",
        "X": "exile",
    }
    for raw_player in parts[1:]:
        player_match = COMPACT_PLAYER_RE.match(raw_player)
        if not player_match:
            continue
        name, life, zones = player_match.groups()
        pdata: Dict[str, object] = {"life": int(life)}
        for zone_match in COMPACT_ZONE_RE.finditer(zones):
            abbrev, count, cards = zone_match.groups()
            key = zone_keys[abbrev]
            pdata[f"{key}_count"] = int(count)
            if cards is not None:
                pdata[f"{key}_cards"] = [c.split(",", 1)[0] for c in split_cards(cards)[:24]]
        players[name] = pdata

    return {"players": players, "stack": stack}


def parse_game(path: Path, include_state: bool = False) -> Optional[Dict[str, object]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    game: Dict[str, object] = {
        "path": str(path),
        "game_id": path.stem,
        "opponent": None,
        "matchup": None,
        "winner": None,
        "loser": None,
        "turns": None,
        "reason": None,
        "decisions": [],
    }

    i = 0
    while i < len(lines):
        line = lines[i]
        match = GAME_RE.match(line)
        if match:
            game["game_log_id"] = match.group(1)
            i += 1
            continue
        match = OPPONENT_RE.match(line)
        if match:
            game["opponent"] = match.group(1)
            i += 1
            continue
        match = MATCHUP_RE.match(line)
        if match:
            agent, opponent = match.groups()
            game["matchup"] = {"agent": agent, "opponent": opponent}
            if not game.get("opponent"):
                label = opponent.strip()
                if label.lower().endswith(".dek"):
                    label = label[:-4]
                game["opponent"] = label
            i += 1
            continue
        match = WINNER_RE.match(line)
        if match:
            game["winner"] = match.group(1)
            i += 1
            continue
        match = LOSER_RE.match(line)
        if match:
            game["loser"] = match.group(1)
            i += 1
            continue
        match = RESULT_RE.match(line)
        if match:
            game["result"] = match.group(1)
            i += 1
            continue
        match = TURNS_RE.match(line)
        if match:
            game["turns"] = int(match.group(1))
            i += 1
            continue
        match = REASON_RE.match(line)
        if match:
            game["reason"] = match.group(1)
            i += 1
            continue

        match = MULLIGAN_RE.match(line)
        if match:
            player, mulligans, hand_size, lands, decision, p_keep, p_mull, hand = match.groups()
            selected = decision.lower()
            game["decisions"].append({
                "kind": "mulligan",
                "actor": player,
                "turn": 0,
                "phase": "Mulligan",
                "selected": selected,
                "options": [
                    {"index": 0, "prob": float(p_keep), "text": "keep"},
                    {"index": 1, "prob": float(p_mull), "text": "mulligan"},
                ],
                "mulligans_taken": int(mulligans),
                "hand_size": int(hand_size),
                "lands": int(lands),
                "hand": split_cards(hand or ""),
                "flags": action_flags(selected),
            })
            i += 1
            continue

        match = LONDON_RE.match(line)
        if match:
            player, mulligans, hand_size, bottom_n, kept, bottomed = match.groups()
            selected = "bottom " + "; ".join(split_cards(bottomed))
            game["decisions"].append({
                "kind": "london_bottom",
                "actor": player,
                "turn": 0,
                "phase": "London Bottom",
                "selected": selected,
                "options": [],
                "mulligans_taken": int(mulligans),
                "hand_size": int(hand_size),
                "bottom_n": int(bottom_n),
                "kept": split_cards(kept),
                "bottomed": split_cards(bottomed),
                "flags": action_flags(selected),
            })
            i += 1
            continue

        match = DECISION_RE.match(line)
        if not match:
            i += 1
            continue

        number, turn, turn_owner, phase, actor = match.groups()
        block: List[str] = []
        j = i + 1
        while j < len(lines):
            if j != i + 1 and DECISION_RE.match(lines[j]):
                break
            if lines[j].startswith("=") and j > i + 1:
                break
            block.append(lines[j])
            if SELECTED_RE.match(lines[j]):
                j += 1
                break
            j += 1

        state_lines: List[str] = []
        options: List[Dict[str, object]] = []
        selected = ""
        value_score: Optional[float] = None
        in_state = False
        for block_line in block:
            if block_line == "GAME STATE:":
                in_state = True
                continue
            if block_line == "OPTIONS & SCORES:":
                in_state = False
                continue
            if in_state:
                state_lines.append(block_line)
            if block_line.strip().startswith("STATE:"):
                state_lines = [block_line]
                continue
            opt = OPTION_RE.match(block_line)
            if opt:
                marker, idx, prob, text = opt.groups()
                options.append({
                    "index": int(idx),
                    "prob": float(prob),
                    "selected_marker": bool(marker),
                    "text": text,
                })
                continue
            compact_top = COMPACT_TOP_RE.match(block_line)
            if compact_top and not options:
                options.extend(parse_compact_top(compact_top.group(1)))
                continue
            val = VALUE_RE.match(block_line)
            if val:
                value_score = float(val.group(1))
                continue
            sel = SELECTED_RE.match(block_line)
            if sel:
                selected = sel.group(1)

        item: Dict[str, object] = {
            "kind": "decision",
            "number": int(number),
            "turn": int(turn),
            "turn_owner": turn_owner,
            "phase": phase,
            "actor": actor,
            "selected": selected,
            "options": options,
            "value_score": value_score,
            "flags": action_flags(selected, options),
            "state_summary": summarize_state(state_lines),
        }
        if include_state:
            item["state_text"] = "\n".join(state_lines)
        game["decisions"].append(item)
        i = j

    if not game["decisions"] and game["winner"] is None:
        return None
    if game.get("result") in {"WIN", "LOSS"}:
        game["rl_won"] = game.get("result") == "WIN"
    else:
        game["rl_won"] = game.get("winner") == "PlayerRL1"
    return game


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="Log file or directory. Can be repeated.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--include-state", action="store_true")
    args = parser.parse_args()

    roots = [Path(p) for p in args.root]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    decisions = 0
    with out.open("w", encoding="utf-8") as fh:
        for path in iter_logs(roots):
            parsed = parse_game(path, include_state=args.include_state)
            if parsed is None:
                continue
            fh.write(json.dumps(parsed, ensure_ascii=True, separators=(",", ":")) + "\n")
            written += 1
            decisions += len(parsed.get("decisions", []))
            if args.max_games and written >= args.max_games:
                break

    print(f"wrote games={written} decisions={decisions} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
