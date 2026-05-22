#!/usr/bin/env python3
"""Build the next-target manifest for Affinity pressure replay/search work.

This is an artifact-only ranking helper. It reads existing mined Affinity
failure-context CSVs and the v77/v83 current-family replay artifacts, then
emits a deterministic target-selection manifest. It does not run Maven,
replay gates, terminal search, training, or source collection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_FAILURE_CORPUS = Path("local-training/local_pbt/corpora/20260518_affinity_failure_context")
DEFAULT_CURRENT_SUMMARY = Path(
    "local-training/local_pbt/spy_line_replay/"
    "20260518_v77_current_family_provenance_gate/v77_decision_summary.csv"
)
DEFAULT_V83_DIAGNOSTIC = Path(
    "local-training/local_pbt/spy_line_replay/"
    "20260518_v83_current_family_d013_roost_randomutil_restore_cp7/"
    "v83_current_family_d013_roost_randomutil_restore_cp7_diagnostic.md"
)
DEFAULT_OUTPUT_DIR = Path("local-training/local_pbt/corpora/20260518_affinity_target_selection")
DEFAULT_DOC = Path("docs/mtgrl/plans/2026-05-18-affinity-target-selection-manifest.md")

TOP_PART_RE = re.compile(r"\s*(\*)?\[(\d+)]\s+([0-9.eE+-]+)\s+(.*)")
STATE_COUNT_RE = re.compile(r"PlayerRL1 L(?P<life>-?\d+) H(?P<hand>\d+)\[.*?\] B(?P<board>\d+)\[.*?\] G(?P<grave>\d+)")
OPP_STATE_RE = re.compile(r"EvalBot-[^ ]+ L(?P<life>-?\d+) H(?P<hand>\d+) B(?P<board>\d+)\[")
REPLAY_PREFIX = "REPLAY:"
REPLAY_RANDOM_PREFIX = "REPLAY_RANDOM:"


@dataclass
class Candidate:
    candidate_id: str
    source_family: str
    game_path: str
    scenario: str = ""
    seed: str = ""
    random_util_seed: str = ""
    action_counterfactual_compatible: bool = False
    decision_number: str = ""
    window_end_decision: str = ""
    turn: str = ""
    phase: str = ""
    selected: str = ""
    selected_class: str = ""
    selected_prob: Optional[float] = None
    top_best_prob: Optional[float] = None
    top_best_text: str = ""
    opp_permanents: int = 0
    own_graveyard_count: int = 0
    own_hand_count: int = 0
    own_life: int = 0
    replay_ready: bool = False
    replay_blocker: str = ""
    tags: List[str] = field(default_factory=list)
    top_alternatives: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    bounded_search_assessment: str = ""
    non_target_reason: str = ""
    thesis_score: float = 0.0
    actionability_score: float = 0.0
    rank_score: float = 0.0


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def basename(path_text: str) -> str:
    return Path(path_text.replace("\\", "/")).name


def parse_key_values(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for part in text.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_log_replay_metadata(log_path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {
        "action_counterfactual_compatible": "false",
        "replay_scenario": "",
        "replay_seed": "",
        "random_util_seed": "",
        "replay_random_scope": "",
    }
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return metadata
    for line in lines[:200]:
        if line.startswith(REPLAY_PREFIX):
            values = parse_key_values(line[len(REPLAY_PREFIX) :])
            metadata["replay_scenario"] = values.get("scenario", metadata["replay_scenario"])
            metadata["replay_seed"] = values.get("seed", metadata["replay_seed"])
            metadata["action_counterfactual_compatible"] = values.get(
                "action_counterfactual_compatible",
                metadata["action_counterfactual_compatible"],
            )
        elif line.startswith(REPLAY_RANDOM_PREFIX):
            values = parse_key_values(line[len(REPLAY_RANDOM_PREFIX) :])
            metadata["replay_scenario"] = values.get("scenario", metadata["replay_scenario"])
            metadata["replay_seed"] = values.get("seed", metadata["replay_seed"])
            metadata["random_util_seed"] = values.get("random_util_seed", metadata["random_util_seed"])
            metadata["replay_random_scope"] = values.get("scope", metadata["replay_random_scope"])
        if metadata["replay_seed"] and metadata["random_util_seed"]:
            break
    return metadata


def row_replay_metadata(row: Dict[str, str], cache: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    metadata = {
        "action_counterfactual_compatible": row.get("action_counterfactual_compatible", ""),
        "replay_scenario": row.get("replay_scenario", ""),
        "replay_seed": row.get("replay_seed", ""),
        "random_util_seed": row.get("random_util_seed", ""),
        "replay_random_scope": row.get("replay_random_scope", ""),
    }
    if metadata["replay_scenario"] and metadata["replay_seed"] and metadata["random_util_seed"]:
        return metadata
    game_path = row.get("game_path", "")
    if not game_path:
        return metadata
    if game_path not in cache:
        cache[game_path] = parse_log_replay_metadata(Path(game_path))
    parsed = cache[game_path]
    for key, value in parsed.items():
        if not metadata.get(key):
            metadata[key] = value
    return metadata


def replay_metadata_ready(metadata: Dict[str, str]) -> bool:
    return (
        truthy(metadata.get("action_counterfactual_compatible"))
        and bool(metadata.get("replay_scenario"))
        and bool(metadata.get("replay_seed"))
        and bool(metadata.get("random_util_seed"))
    )


def classify_selected(selected: str, phase: str = "") -> str:
    text = (selected or "").strip()
    lower = text.lower()
    phase_lower = (phase or "").lower()
    if "target_pick" in phase_lower or "choose_use" in phase_lower:
        return "target_or_choice"
    if text == "Pass":
        return "pass"
    if "flashback sacrifice" in lower:
        return "flashback"
    if lower.startswith("cast dread return"):
        return "cast_dread_return"
    if lower.startswith("cast balustrade spy"):
        return "cast_spy"
    if lower.startswith("cast "):
        return "cast_other"
    if lower.startswith("play "):
        return "play_land"
    if "cycling" in lower:
        return "cycling"
    if "add {" in lower or "add " in lower and "{t}" in lower:
        return "mana_ability"
    return "other"


def parse_top_line(top_line: str) -> Tuple[int, List[Tuple[int, float, str, bool]]]:
    if not top_line:
        return 0, []
    text = top_line.strip()
    if text.startswith("TOP:"):
        text = text[len("TOP:") :].strip()
    count = 0
    if text.startswith("n="):
        parts = text.split("|", 1)
        count_text = parts[0].strip().replace("n=", "")
        count = safe_int(count_text)
        text = parts[1] if len(parts) > 1 else ""
    entries: List[Tuple[int, float, str, bool]] = []
    for part in text.split("|"):
        match = TOP_PART_RE.match(part.strip())
        if not match:
            continue
        selected_marker, idx, prob, option_text = match.groups()
        parsed_prob = safe_float(prob)
        if parsed_prob is None:
            continue
        entries.append((safe_int(idx), parsed_prob, option_text.strip(), bool(selected_marker)))
    if not count:
        count = len(entries)
    return count, entries


def parse_options_json(options_json: str) -> List[Tuple[int, float, str, bool]]:
    if not options_json:
        return []
    try:
        raw = json.loads(options_json)
    except json.JSONDecodeError:
        return []
    entries: List[Tuple[int, float, str, bool]] = []
    for item in raw or []:
        prob = safe_float(item.get("prob"))
        if prob is None:
            continue
        entries.append(
            (
                safe_int(item.get("index")),
                prob,
                str(item.get("text", "")).strip(),
                bool(item.get("selected_marker")),
            )
        )
    return entries


def top_alternatives(entries: Sequence[Tuple[int, float, str, bool]], limit: int = 4) -> List[str]:
    alts = [entry for entry in entries if not entry[3]]
    alts.sort(key=lambda item: item[1], reverse=True)
    return [f"{idx}:{prob:.4f} {text}" for idx, prob, text, _ in alts[:limit]]


def parse_state_json(state_json: str) -> Dict[str, object]:
    if not state_json:
        return {}
    try:
        return json.loads(state_json)
    except json.JSONDecodeError:
        return {}


def player_state(state: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object]]:
    players = state.get("players") if isinstance(state, dict) else None
    if not isinstance(players, dict):
        return {}, {}
    own = players.get("PlayerRL1")
    if not isinstance(own, dict):
        own = {}
    opp = {}
    for name, payload in players.items():
        if name != "PlayerRL1" and isinstance(payload, dict):
            opp = payload
            break
    return own, opp


def text_has_card(cards: Iterable[object], needle: str) -> bool:
    needle_lower = needle.lower()
    return any(needle_lower in str(card).lower() for card in cards)


def parse_current_state_counts(state_text: str) -> Tuple[int, int, int, int]:
    own_life = own_hand = own_grave = opp_board = 0
    own_match = STATE_COUNT_RE.search(state_text or "")
    if own_match:
        own_life = safe_int(own_match.group("life"))
        own_hand = safe_int(own_match.group("hand"))
        own_grave = safe_int(own_match.group("grave"))
    opp_match = OPP_STATE_RE.search(state_text or "")
    if opp_match:
        opp_board = safe_int(opp_match.group("board"))
    return own_life, own_hand, own_grave, opp_board


def add_tag(tags: List[str], tag: str) -> None:
    if tag and tag not in tags:
        tags.append(tag)


def score_candidate(candidate: Candidate) -> None:
    pressure_score = 0.0
    if candidate.opp_permanents >= 10:
        pressure_score = 30.0 + min(8.0, float(candidate.opp_permanents - 10) * 1.5)
        add_tag(candidate.tags, "pressure_opp_permanents_ge_10")
    elif candidate.opp_permanents >= 8:
        pressure_score = 12.0

    low_prob_score = 0.0
    if candidate.selected_prob is not None:
        low_prob_score = max(0.0, 1.0 - candidate.selected_prob) * 12.0
        if candidate.selected_prob <= 0.35:
            low_prob_score += 8.0
            add_tag(candidate.tags, "accepted_low_probability")

    terminal_score = 0.0
    text = f"{candidate.selected} {candidate.phase} {' '.join(candidate.top_alternatives)}".lower()
    if "flashback sacrifice" in text or "dread return" in text:
        terminal_score += 20.0
        add_tag(candidate.tags, "terminal_combo_sequence")
    if "balustrade spy" in text or candidate.selected_class == "cast_spy":
        terminal_score += 14.0
        add_tag(candidate.tags, "spy_conversion_surface")
    if "target_pick" in candidate.phase.lower() or candidate.selected_class == "target_or_choice":
        terminal_score += 8.0
        add_tag(candidate.tags, "target_sequence")

    generic_score = 0.0
    if "source_zone_mistake" in candidate.tags:
        generic_score += 16.0
    if candidate.selected_class in {"flashback", "cast_dread_return", "cast_spy", "target_or_choice"}:
        generic_score += 8.0
    if candidate.selected_class == "pass" and candidate.opp_permanents >= 10 and candidate.top_alternatives:
        generic_score += 6.0
        add_tag(candidate.tags, "pass_under_pressure_with_siblings")
    if "current_family_window" in candidate.tags:
        generic_score += 6.0

    candidate.thesis_score = round(pressure_score + low_prob_score + terminal_score + generic_score, 3)

    actionability = 0.0
    if candidate.replay_ready:
        actionability += 30.0
        add_tag(candidate.tags, "replay_ready")
        if candidate.source_family == "v77_current_family":
            add_tag(candidate.tags, "replay_ready_current_family")
        else:
            add_tag(candidate.tags, "replay_ready_failure_corpus")
    if candidate.source_family == "v77_current_family":
        actionability += 10.0
    if "current_family_window" in candidate.tags:
        actionability += 10.0
    if candidate.top_alternatives and len(candidate.top_alternatives) <= 4:
        actionability += 4.0
    if not candidate.replay_ready and candidate.replay_blocker:
        actionability -= 10.0
        add_tag(candidate.tags, "needs_replay_metadata")

    candidate.actionability_score = round(actionability, 3)
    candidate.rank_score = round(candidate.thesis_score + candidate.actionability_score, 3)

    if not candidate.bounded_search_assessment:
        if candidate.replay_ready:
            candidate.bounded_search_assessment = (
                "High: replay metadata is present, so a bounded sibling/short-prefix "
                "checkpoint probe is actionable."
            )
        elif candidate.replay_blocker:
            candidate.bounded_search_assessment = (
                "Low immediate actionability: thesis-relevant state, but old compact logs "
                "lack replay seed/snapshot metadata."
            )
        else:
            candidate.bounded_search_assessment = "Medium: useful artifact row, but replay readiness is not proven."


def load_anchor_index(anchor_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    anchors: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in read_csv(anchor_path):
        key = (basename(row.get("game_path", "")), str(row.get("decision_number", "")))
        anchors[key] = row
    return anchors


def build_failure_candidates(corpus_dir: Path) -> List[Candidate]:
    rows = read_csv(corpus_dir / "loss_decisions.csv")
    anchors = load_anchor_index(corpus_dir / "replay_anchor_manifest.csv")
    metadata_cache: Dict[str, Dict[str, str]] = {}
    candidates: List[Candidate] = []
    for row in rows:
        decision_kind = row.get("decision_kind", "")
        if decision_kind in {"mulligan", "london_bottom"}:
            continue
        state = parse_state_json(row.get("state_json", ""))
        own, opp = player_state(state)
        selected = row.get("selected", "")
        phase = row.get("phase", "")
        selected_class = classify_selected(selected, phase)
        entries = parse_options_json(row.get("options_json", ""))
        selected_prob = safe_float(row.get("top_selected_prob"))
        if selected_prob is None:
            for _, prob, _, is_selected in entries:
                if is_selected:
                    selected_prob = prob
                    break

        game_path = row.get("game_path", "")
        decision = row.get("decision_number", "")
        anchor = anchors.get((basename(game_path), str(decision)), {})
        replay_metadata = row_replay_metadata(row, metadata_cache)
        replay_ready = anchor.get("replay_ready", "").lower() == "true" or replay_metadata_ready(replay_metadata)
        opp_permanents = safe_int(opp.get("permanents_count"))
        own_grave = safe_int(own.get("graveyard_count"))
        own_hand = safe_int(own.get("hand_count"))
        own_life = safe_int(own.get("life"))

        if opp_permanents < 10 and selected_prob is not None and selected_prob > 0.35 and selected_class not in {
            "flashback",
            "cast_dread_return",
            "cast_spy",
            "target_or_choice",
        }:
            continue
        alternatives = top_alternatives(entries)
        if selected_class == "pass":
            continue

        candidate = Candidate(
            candidate_id=anchor.get("anchor_id") or f"{basename(game_path).replace('.txt', '')}_D{int(decision):03d}",
            source_family="accepted_32_game_corpus",
            game_path=game_path,
            scenario=anchor.get("scenario") or replay_metadata.get("replay_scenario", ""),
            seed=anchor.get("seed") or replay_metadata.get("replay_seed", ""),
            random_util_seed=replay_metadata.get("random_util_seed", ""),
            action_counterfactual_compatible=truthy(replay_metadata.get("action_counterfactual_compatible")),
            decision_number=decision,
            turn=row.get("turn", ""),
            phase=phase,
            selected=selected,
            selected_class=anchor.get("selected_class") or selected_class,
            selected_prob=selected_prob,
            top_best_prob=safe_float(row.get("top_best_prob")),
            top_best_text=row.get("top_best_text", ""),
            opp_permanents=opp_permanents,
            own_graveyard_count=own_grave,
            own_hand_count=own_hand,
            own_life=own_life,
            replay_ready=replay_ready,
            replay_blocker="" if replay_ready else (anchor.get("replay_blocker", "") or "missing_replay_metadata"),
            top_alternatives=alternatives,
        )
        if anchor:
            candidate.evidence.append(f"anchor_category={anchor.get('category', '')}")
            candidate.evidence.append(f"anchor_rank={anchor.get('rank', '')}")
            if anchor.get("selected_in_hand", "").lower() == "true" and anchor.get("selected_in_graveyard", "").lower() != "true":
                add_tag(candidate.tags, "source_zone_mistake")
            if anchor.get("suffix"):
                candidate.evidence.append(f"suffix={anchor.get('suffix', '')[:220]}")
        if replay_ready:
            candidate.evidence.append("compact_replay_metadata_present=true")
            candidate.evidence.append(f"random_util_seed={candidate.random_util_seed}")
        if "dread return" in selected.lower():
            hand_cards = own.get("hand_cards") or []
            grave_cards = own.get("graveyard_cards") or []
            if text_has_card(hand_cards, "Dread Return") and not text_has_card(grave_cards, "Dread Return"):
                add_tag(candidate.tags, "source_zone_mistake")
        score_candidate(candidate)
        candidates.append(candidate)
    return candidates


def build_current_candidates(current_summary: Path) -> List[Candidate]:
    rows = read_csv(current_summary)
    candidates: List[Candidate] = []
    by_game: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_game.setdefault(row.get("game", ""), []).append(row)
        if row.get("result") != "LOSS":
            continue
        selected = row.get("selected", "")
        phase = row.get("phase", "")
        selected_class = classify_selected(selected, phase)
        top_count, entries = parse_top_line(row.get("top", ""))
        selected_prob = None
        for _, prob, _, is_selected in entries:
            if is_selected:
                selected_prob = prob
                break
        own_life, own_hand, own_grave, state_opp_board = parse_current_state_counts(row.get("state", ""))
        if own_life <= 0:
            continue
        opp_permanents = safe_int(row.get("opp_battlefield")) or state_opp_board
        if opp_permanents < 10 and selected_prob is not None and selected_prob > 0.35 and selected_class not in {
            "flashback",
            "cast_spy",
            "target_or_choice",
        }:
            continue
        alternatives = top_alternatives(entries)
        if selected_class == "pass":
            continue
        scenario = row.get("scenario", "")
        decision = row.get("decision", "")
        candidate = Candidate(
            candidate_id=f"v77_s{scenario}_D{safe_int(decision):03d}",
            source_family="v77_current_family",
            game_path=row.get("game", ""),
            scenario=scenario,
            seed=row.get("seed", ""),
            action_counterfactual_compatible=True,
            decision_number=decision,
            turn=row.get("turn", ""),
            phase=phase,
            selected=selected,
            selected_class=selected_class,
            selected_prob=selected_prob,
            top_best_prob=max((entry[1] for entry in entries), default=None),
            top_best_text=max(entries, key=lambda entry: entry[1])[2] if entries else "",
            opp_permanents=opp_permanents,
            own_graveyard_count=own_grave,
            own_hand_count=own_hand,
            own_life=own_life,
            replay_ready=True,
            top_alternatives=alternatives,
        )
        candidate.evidence.append(f"top_candidate_count={top_count}")
        candidate.evidence.append("current_family_has_replay_metadata=true")
        score_candidate(candidate)
        candidates.append(candidate)

    candidates.extend(build_current_windows(by_game))
    return candidates


def build_current_windows(by_game: Dict[str, List[Dict[str, str]]]) -> List[Candidate]:
    windows: List[Candidate] = []
    for game, rows in by_game.items():
        rows_sorted = sorted(rows, key=lambda item: safe_int(item.get("decision")))
        by_decision = {safe_int(row.get("decision")): row for row in rows_sorted}
        for row in rows_sorted:
            if row.get("result") != "LOSS":
                continue
            selected = row.get("selected", "")
            if "flashback sacrifice" not in selected.lower():
                continue
            decision = safe_int(row.get("decision"))
            following = [by_decision.get(decision + offset) for offset in range(1, 4)]
            following = [item for item in following if item and "target_pick" in item.get("phase", "").lower()]
            if not following:
                continue
            top_count, entries = parse_top_line(row.get("top", ""))
            selected_prob = next((prob for _, prob, _, selected_marker in entries if selected_marker), None)
            own_life, own_hand, own_grave, state_opp_board = parse_current_state_counts(row.get("state", ""))
            opp_permanents = safe_int(row.get("opp_battlefield")) or state_opp_board
            sequence = [f"D{row.get('decision')} {selected}"]
            for item in following:
                sequence.append(f"D{item.get('decision')} {item.get('selected')}")
            scenario = row.get("scenario", "")
            window_end = following[-1].get("decision", "")
            candidate = Candidate(
                candidate_id=f"v77_s{scenario}_D{decision:03d}_D{safe_int(window_end):03d}_flashback_target_window",
                source_family="v77_current_family",
                game_path=game,
                scenario=scenario,
                seed=row.get("seed", ""),
                action_counterfactual_compatible=True,
                decision_number=str(decision),
                window_end_decision=window_end,
                turn=row.get("turn", ""),
                phase=row.get("phase", ""),
                selected=" -> ".join(sequence),
                selected_class="flashback_target_window",
                selected_prob=selected_prob,
                top_best_prob=max((entry[1] for entry in entries), default=None),
                top_best_text=max(entries, key=lambda entry: entry[1])[2] if entries else "",
                opp_permanents=opp_permanents,
                own_graveyard_count=own_grave,
                own_hand_count=own_hand,
                own_life=own_life,
                replay_ready=True,
                top_alternatives=top_alternatives(entries),
                evidence=[
                    "current-family replay metadata present",
                    "v83 proves this source family can replay through D013 with per-search RNG provenance",
                    f"short target sequence={' | '.join(sequence)}",
                ],
                bounded_search_assessment=(
                    "Very high: build one forced-prefix bridge to D089, then enumerate the "
                    "D90-D92 target/sacrifice short window with terminal-win-only admission."
                ),
            )
            add_tag(candidate.tags, "current_family_window")
            add_tag(candidate.tags, "terminal_combo_sequence")
            add_tag(candidate.tags, "target_sequence")
            score_candidate(candidate)
            windows.append(candidate)
    return windows


def candidate_to_row(rank: int, candidate: Candidate) -> Dict[str, object]:
    return {
        "rank": rank,
        "candidate_id": candidate.candidate_id,
        "source_family": candidate.source_family,
        "rank_score": candidate.rank_score,
        "thesis_score": candidate.thesis_score,
        "actionability_score": candidate.actionability_score,
        "replay_ready": candidate.replay_ready,
        "game_path": candidate.game_path,
        "scenario": candidate.scenario,
        "seed": candidate.seed,
        "random_util_seed": candidate.random_util_seed,
        "action_counterfactual_compatible": candidate.action_counterfactual_compatible,
        "decision_number": candidate.decision_number,
        "window_end_decision": candidate.window_end_decision,
        "turn": candidate.turn,
        "phase": candidate.phase,
        "selected_class": candidate.selected_class,
        "selected": candidate.selected,
        "selected_prob": "" if candidate.selected_prob is None else round(candidate.selected_prob, 6),
        "top_best_prob": "" if candidate.top_best_prob is None else round(candidate.top_best_prob, 6),
        "top_best_text": candidate.top_best_text,
        "opp_permanents": candidate.opp_permanents,
        "own_graveyard_count": candidate.own_graveyard_count,
        "own_hand_count": candidate.own_hand_count,
        "own_life": candidate.own_life,
        "tags": ";".join(candidate.tags),
        "top_alternatives": " || ".join(candidate.top_alternatives),
        "bounded_search_assessment": candidate.bounded_search_assessment,
        "replay_blocker": candidate.replay_blocker,
        "evidence": " || ".join(candidate.evidence),
        "non_target_reason": candidate.non_target_reason,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["rank", "candidate_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for column in columns:
            value = str(row.get(column, ""))
            value = value.replace("|", "/").replace("\n", " ")
            if len(value) > 130:
                value = value[:127] + "..."
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    generated_at: str,
    ranked_rows: Sequence[Dict[str, object]],
    summary: Dict[str, object],
    recommendation: Dict[str, object],
    non_targets: Sequence[Dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top10 = list(ranked_rows[:10])
    old_high = [row for row in ranked_rows if row.get("source_family") == "accepted_32_game_corpus"][:5]
    replay_ready = [row for row in ranked_rows if row.get("replay_ready")][:5]

    lines = [
        "# Affinity Target-Selection Manifest - 2026-05-18",
        "",
        "## Scope",
        "",
        "Analysis-only manifest built from existing artifacts. No Maven, replay gate, league_bench, training, HPC, terminal search, or fresh source collection was run.",
        "",
        "The thesis filter is: choose targets that test whether the accepted policy has generic, pressure-state failures that a bounded sibling or short-prefix search can turn into terminal-winning corrections.",
        "",
        "## Inputs",
        "",
        f"- Generated at UTC: `{generated_at}`",
        f"- 32-game Affinity corpus: `{DEFAULT_FAILURE_CORPUS}`",
        f"- Current-family v77 decision summary: `{DEFAULT_CURRENT_SUMMARY}`",
        f"- v83 replay diagnostic: `{DEFAULT_V83_DIAGNOSTIC}`",
        "",
        "## Corpus Read",
        "",
        f"- 32-game corpus loss decisions ranked: `{summary.get('failure_candidates', 0)}`",
        f"- Current-family replay-ready candidates ranked: `{summary.get('current_candidates', 0)}`",
        f"- High-pressure candidates with opponent permanents >=10: `{summary.get('pressure_candidates', 0)}`",
        f"- Replay-ready candidates: `{summary.get('replay_ready_candidates', 0)}`",
        "",
        "Rows are immediate replay targets only when compact log metadata includes `action_counterfactual_compatible=true`, scenario/seed, and `REPLAY_RANDOM` random-util seed data. Older non-metadata logs remain useful for mining, but not for checkpoint probes.",
        "",
        "## Recommended Next Target",
        "",
        f"- Candidate: `{recommendation.get('candidate_id', '')}`",
        f"- Source: `{recommendation.get('source_family', '')}`",
        f"- Scenario/seed: `{recommendation.get('scenario', '')}` / `{recommendation.get('seed', '')}`",
        f"- Decision window: `D{recommendation.get('decision_number', '')}` to `D{recommendation.get('window_end_decision') or recommendation.get('decision_number', '')}`",
        f"- Selected sequence: `{recommendation.get('selected', '')}`",
        f"- Pressure: opponent permanents `{recommendation.get('opp_permanents', '')}`, own life `{recommendation.get('own_life', '')}`",
        f"- Assessment: {recommendation.get('bounded_search_assessment', '')}",
        "",
        "Why this advances the thesis: it is a live accepted-policy Affinity loss under public-board pressure with replay metadata. A bounded sibling/short-prefix checkpoint probe can admit only terminal-winning corrected suffixes, which directly tests whether the failure mode is policy-relevant rather than just replay-local.",
        "",
        "Next exact unit: build one forced-prefix bridge for the recommended replay-ready row, carrying forward the compact log replay metadata and any per-search RandomUtil provenance. Then run a bounded sibling/short-prefix checkpoint probe, admitting a case only if the baseline loses and the corrected suffix reaches terminal win. Do not train from this manifest alone.",
        "",
        "## Top 10 Ranked Targets",
        "",
        markdown_table(
            top10,
            [
                "rank",
                "candidate_id",
                "source_family",
                "rank_score",
                "replay_ready",
                "decision_number",
                "window_end_decision",
                "selected",
                "selected_prob",
                "opp_permanents",
                "own_life",
                "tags",
            ],
        ),
        "",
        "## Replay-Ready Shortlist",
        "",
        markdown_table(
            replay_ready,
            [
                "rank",
                "candidate_id",
                "source_family",
                "rank_score",
                "decision_number",
                "window_end_decision",
                "selected",
                "selected_prob",
                "opp_permanents",
                "bounded_search_assessment",
            ],
        ),
        "",
        "## High-Thesis Older Corpus Shortlist",
        "",
        markdown_table(
            old_high,
            [
                "rank",
                "candidate_id",
                "rank_score",
                "decision_number",
                "selected",
                "selected_prob",
                "opp_permanents",
                "replay_ready",
                "replay_blocker",
            ],
        ),
        "",
        "These older rows should guide future collection/recollection, not the immediate replay gate, unless a replay-ready source family is rebuilt for them.",
        "",
        "## Explicit Non-Targets",
        "",
    ]
    for item in non_targets:
        lines.append(f"- `{item['target']}`: {item['reason']}")
    lines.extend(
        [
            "",
            "## Machine Artifacts",
            "",
            f"- CSV: `{DEFAULT_OUTPUT_DIR / 'target_selection_manifest.csv'}`",
            f"- JSON: `{DEFAULT_OUTPUT_DIR / 'target_selection_manifest.json'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_non_targets() -> List[Dict[str, str]]:
    return [
        {
            "target": "v83 D013",
            "reason": "Already passed as a setup/parity gate; repeating it does not test pressure recovery.",
        },
        {
            "target": "v77 D025",
            "reason": "Documented as mana/pass-adjacent bridge work, not a policy-relevant pressure decision.",
        },
        {
            "target": "singleton pass rows under pressure",
            "reason": "They dominate the corpus but often have no meaningful sibling branch to test.",
        },
        {
            "target": "old compact D084/D103/D134 anchors as immediate replay gates",
            "reason": "High thesis value but not immediate gates because the old logs lack replay seed/snapshot metadata.",
        },
        {
            "target": "automatic D025 or D089 ordinal walking",
            "reason": "Replay work should be target-selected by pressure and terminal-recovery value, not by next ordinal.",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-corpus", type=Path, default=DEFAULT_FAILURE_CORPUS)
    parser.add_argument("--current-summary", type=Path, default=DEFAULT_CURRENT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    failure_candidates = build_failure_candidates(args.failure_corpus)
    current_candidates = build_current_candidates(args.current_summary)
    candidates = failure_candidates + current_candidates
    candidates.sort(key=lambda item: (-item.rank_score, -item.thesis_score, item.candidate_id))
    rows = [candidate_to_row(rank, candidate) for rank, candidate in enumerate(candidates, start=1)]

    recommendation = next((row for row in rows if row["replay_ready"]), rows[0] if rows else {})
    non_targets = build_non_targets()
    tag_counts = Counter(tag for candidate in candidates for tag in candidate.tags)
    summary = {
        "generated_at_utc": generated_at,
        "failure_candidates": len(failure_candidates),
        "current_candidates": len(current_candidates),
        "total_candidates": len(candidates),
        "pressure_candidates": sum(1 for candidate in candidates if candidate.opp_permanents >= 10),
        "replay_ready_candidates": sum(1 for candidate in candidates if candidate.replay_ready),
        "tag_counts": dict(sorted(tag_counts.items())),
        "inputs": {
            "failure_corpus": str(args.failure_corpus),
            "current_summary": str(args.current_summary),
            "v83_diagnostic": str(DEFAULT_V83_DIAGNOSTIC),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "target_selection_manifest.csv", rows)
    with (args.output_dir / "target_selection_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "recommended_next_target": recommendation,
                "top_10": rows[:10],
                "non_targets": non_targets,
                "all_candidates": rows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    write_markdown(args.doc, generated_at, rows, summary, recommendation, non_targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
