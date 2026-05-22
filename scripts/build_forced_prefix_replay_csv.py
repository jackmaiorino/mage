#!/usr/bin/env python3
"""Build an ActionCounterfactual forced-prefix replay CSV from one compact log.

`GameLogger` displays decisions as 1-based `DECISION #N` values. The replay
player stores `DecisionPoint` ordinals from 0. By default this converter keeps
the compact-log source ordinal stream for compatibility; use
`--ordinal-space acf` for ActionCounterfactual forced-prefix replay, where
singleton/no-op compact rows are skipped and emitted ordinals are remapped to
the trainer's eligible decision stream. The displayed source decision remains in
`source_decision_number` for inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_LOG = Path(
    "local-training/local_pbt/cp7_eval_sweeps/20260518_affinity_replay_metadata_smoke_g4/"
    "game_logs/Pauper-Spy-Combo-Value__Deck_-_Spy_Combo__vs__Deck_-_Grixis_Affinity__chunk_003/"
    "game_20260518_001722_0001.txt"
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
    "source_game_path",
    "source_decision_number",
    "source_anchor_id",
    "target_marker",
    "turn",
    "phase",
    "actor",
    "selected_prob",
    "selected_line",
    "top_line",
    "source_candidate_count",
    "source_candidate_indices",
    "source_candidate_texts",
    "source_selected_index",
    "source_selected_text",
    "source_stable_text_fallback",
    "source_hand",
    "source_hand_object_ids",
    "source_library_top",
    "source_library_top_object_ids",
    "source_library",
    "source_library_object_ids",
    "source_random_util_count_before_search",
    "source_stack_count",
    "source_stack_top",
    "source_candidate_object_ids",
    "source_selected_object_ids",
    "source_object_ids",
    "source_target_object_ids",
    "source_identity_required",
    "source_identity_status",
)

REPLAY_KV_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=")
DECISION_RE = re.compile(r"^DECISION #(\d+) - Turn (\d+) \((.*?) turn\), (.*?) - (.*?)$")
SELECTED_RE = re.compile(r"^\s*SELECTED\[(\d+)]\s+p=([0-9.eE+-]+)\s+value=[^:]*:\s*(.*)$")
TOP_PART_RE = re.compile(r"^\s*(\*)?\[(\d+)]\s+([0-9.eE+-]+)\s+(.*)$")
TOP_N_RE = re.compile(r"\bTOP:\s*n=(\d+)\b")
STACK_STATE_RE = re.compile(r"\bstack=(\d+) items(?:\s+top=\[\d+]\s*(.*?))?(?:\s*\|\||$)")
MULLIGAN_RE = re.compile(r"^MULLIGAN_DECISION: .*?\bdecision=(KEEP|MULLIGAN)\b.*?\bhand=\[(.*)]")
LONDON_BOTTOM_RE = re.compile(r"^LONDON_BOTTOM: .*?\bbottomN=(\d+)\s+kept=\[(.*?)]\s+bottomed=\[(.*)]")
LONDON_BOTTOM_PROBS_RE = re.compile(r"^LONDON_BOTTOM_PROBS: .*?\bcards=\[(.*)]")
UUID_LIKE_TEXT_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
UUID_LIKE_TOKEN_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
REPLAY_DECISION_JSON_PREFIX = "REPLAY_DECISION_JSON:"
REPLAY_OPPONENT_DECISION_JSON_PREFIX = "REPLAY_OPPONENT_DECISION_JSON:"
REPLAY_AGENT_SEARCH_JSON_PREFIX = "REPLAY_AGENT_SEARCH_JSON:"
REPLAY_SELECTED_LINE_ORDINAL_RE = re.compile(r"\bordinal=(\d+)\b")
COMBAT_TRACE_OBJECT_RE = re.compile(
    r"(?P<name>[^>|;{}]+?)\{id=(?P<object_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}),"
)
ORDINAL_SPACE_SOURCE = "source"
ORDINAL_SPACE_ACF = "acf"
PREFIX_CHOICE_TEXT = "text"
PREFIX_CHOICE_INDEX = "index"
PREFIX_CHOICE_OBJECT_TEXT = "object-text"
OBJECT_TEXT_ACTION_TYPES = {"SELECT_TARGETS", "SELECT_CARD"}
IDENTITY_REQUIRED_ACTION_TYPES = {"DECLARE_BLOCKS", "SELECT_TARGETS", "SELECT_CARD"}
OBJECT_TEXT_PHASE_MARKERS = (
    "DECLARE_BLOCKS",
    "DECLARE BLOCKERS",
    "TARGET_PICK",
    "CARD_PICK",
    "SELECT_CARD",
)
SINGLETON_COMBAT_PASS_ACTION_TYPES = {"DECLARE_ATTACKS", "DECLARE_BLOCKS"}
UNSTABLE_FALLBACK_TEXTS = {"", "PASS", "DONE", "STOP"}
NO_OBJECT_CHOICE_IDS = {
    "DONE": "__NO_OBJECT_CHOICE_DONE__",
}


class PrefixBuildError(ValueError):
    pass


def parse_replay_metadata(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    matches = list(REPLAY_KV_RE.finditer(line or ""))
    for idx, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line or "")
        out[match.group(1).strip()] = (line or "")[value_start:value_end].strip()
    return out


def replay_selected_line_ordinal(row: Dict[str, str]) -> str:
    match = REPLAY_SELECTED_LINE_ORDINAL_RE.search(str(row.get("selected_line", "") or ""))
    if not match:
        return ""
    return match.group(1)


def replay_selected_line_ordinal_int(row: Dict[str, str]) -> Optional[int]:
    raw = replay_selected_line_ordinal(row)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def split_joined_texts(value: str) -> List[str]:
    return [part.strip() for part in str(value or "").split("||") if part.strip()]


def split_joined_object_ids(value: str) -> List[str]:
    if value is None or value == "":
        return []
    return [part.strip() for part in str(value).split("||")]


def stable_no_object_choice_id(text: str) -> str:
    return NO_OBJECT_CHOICE_IDS.get((text or "").strip().upper(), "")


def normalize_trace_object_name(text: str) -> str:
    value = (text or "").strip()
    if "=" in value:
        value = value.rsplit("=", 1)[-1].strip()
    return re.sub(r"\s+", " ", value).strip().lower()


def combat_trace_object_id_map(trace: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for match in COMBAT_TRACE_OBJECT_RE.finditer(trace or ""):
        name = normalize_trace_object_name(match.group("name"))
        object_id = match.group("object_id").strip()
        if not name or not object_id:
            continue
        ids = out.setdefault(name, [])
        if object_id not in ids:
            ids.append(object_id)
    return out


def iter_replay_opponent_payloads(block: Sequence[str]) -> Iterable[Dict[str, object]]:
    for raw_line in block:
        line = (raw_line or "").strip()
        if not line.startswith(REPLAY_OPPONENT_DECISION_JSON_PREFIX):
            continue
        raw = line[len(REPLAY_OPPONENT_DECISION_JSON_PREFIX):].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


def declare_blocks_combat_object_ids(block: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for payload in iter_replay_opponent_payloads(block):
        phase = str(payload.get("phase") or "").upper()
        if "DECLARE BLOCK" not in phase:
            continue
        trace_ids = combat_trace_object_id_map(str(payload.get("combat_trace") or ""))
        for name, object_ids in trace_ids.items():
            ids = out.setdefault(name, [])
            for object_id in object_ids:
                if object_id not in ids:
                    ids.append(object_id)
    return out


def attach_declare_blocks_object_ids(row: Dict[str, str], block: Sequence[str]) -> None:
    if str(row.get("action_type", "")).upper() != "DECLARE_BLOCKS":
        return
    if row.get("source_candidate_object_ids") or row.get("source_selected_object_ids"):
        return
    candidate_texts = split_joined_texts(str(row.get("source_candidate_texts", "")))
    chosen_indices = parse_semicolon_ints(str(row.get("chosen_indices", "")))
    if not candidate_texts or not chosen_indices:
        return
    trace_ids = declare_blocks_combat_object_ids(block)
    candidate_ids: List[str] = []
    for candidate_text in candidate_texts:
        no_object_id = stable_no_object_choice_id(candidate_text)
        if no_object_id:
            candidate_ids.append(no_object_id)
            continue
        object_ids = trace_ids.get(normalize_trace_object_name(candidate_text), [])
        candidate_ids.append(object_ids[0] if len(object_ids) == 1 else "")
    selected_ids = [
        candidate_ids[idx]
        for idx in chosen_indices
        if 0 <= idx < len(candidate_ids) and candidate_ids[idx]
    ]
    row["source_candidate_object_ids"] = join_texts(candidate_ids)
    row["source_selected_object_ids"] = join_texts(selected_ids)


def row_source_hand_contains(row: Dict[str, str], source_name: str) -> bool:
    needle = (source_name or "").strip().lower()
    if not needle:
        return False
    return any(item.lower() == needle for item in split_joined_texts(str(row.get("source_hand", "") or "")))


def is_generic_search_or_cycling_text(text: str, source_name: str) -> bool:
    selected = (text or "").strip().lower()
    if not selected:
        return False
    if (source_name or "").strip().lower() in selected:
        return False
    return "cycling" in selected or "search your library" in selected


def iter_search_metadata(block: Sequence[str]) -> Iterable[Tuple[str, str, str, str, List[str]]]:
    for raw_line in block:
        line = (raw_line or "").strip()
        if not line.startswith(REPLAY_AGENT_SEARCH_JSON_PREFIX):
            continue
        raw = line[len(REPLAY_AGENT_SEARCH_JSON_PREFIX):].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if str(payload.get("actor") or "") != "PlayerRL1":
            continue
        source_name = str(payload.get("source_name") or "").strip()
        count = payload.get("random_util_count_before_search")
        if not source_name or count is None:
            continue
        source_selector = str(
            payload.get("source_decision_number")
            or payload.get("source_anchor_id")
            or ""
        ).strip()
        decision_ordinal_next = str(payload.get("decision_ordinal_next") or "").strip()
        chosen_names = str_list(payload.get("chosen_names"))
        target_name = str(payload.get("target_name") or "").strip()
        copy_pick_marker = str(payload.get("copy_pick_marker") or "").strip().lower()
        if (
            not chosen_names
            and not target_name
            and "without_search_context" in copy_pick_marker
        ):
            continue
        yield source_name, str(count), normalized_source_selector(source_selector), decision_ordinal_next, chosen_names


def attach_search_metadata(rows: Sequence[Dict[str, str]], block: Sequence[str]) -> None:
    for source_name, count, source_decision_number, decision_ordinal_next, chosen_names in iter_search_metadata(block):
        if source_decision_number:
            attached = False
            for row in rows:
                if row.get("source_random_util_count_before_search"):
                    continue
                if normalized_source_selector(row.get("source_decision_number", "")) == source_decision_number:
                    row["source_random_util_count_before_search"] = count
                    attached = True
                    break
            if attached:
                continue
        if decision_ordinal_next:
            try:
                trigger_ordinal = int(decision_ordinal_next) - 1
            except ValueError:
                trigger_ordinal = -1
            attached = False
            for row in reversed(rows):
                if row.get("source_random_util_count_before_search"):
                    continue
                if replay_selected_line_ordinal_int(row) != trigger_ordinal:
                    continue
                if not row_source_hand_contains(row, source_name):
                    continue
                if not is_generic_search_or_cycling_text(str(row.get("source_selected_text", "") or ""), source_name):
                    continue
                row["source_random_util_count_before_search"] = count
                attached = True
                break
            if attached:
                continue
        if decision_ordinal_next and chosen_names:
            chosen_set = {name.strip().lower() for name in chosen_names if name.strip()}
            attached = False
            for row in rows:
                if row.get("source_random_util_count_before_search"):
                    continue
                if str(row.get("action_type", "")).upper() not in OBJECT_TEXT_ACTION_TYPES:
                    continue
                if replay_selected_line_ordinal(row) != decision_ordinal_next:
                    continue
                selected_text = str(row.get("source_selected_text", "") or "").strip().lower()
                if selected_text and selected_text in chosen_set:
                    row["source_random_util_count_before_search"] = count
                    attached = True
                    break
            if attached:
                continue
        needle = source_name.lower()
        for row in reversed(rows):
            if row.get("source_random_util_count_before_search"):
                continue
            selected = " ".join(
                str(row.get(key, "") or "")
                for key in ("source_selected_text", "selected_line")
            ).lower()
            if needle in selected:
                row["source_random_util_count_before_search"] = count
                break


def join_texts(values: Iterable[str]) -> str:
    return "||".join((value or "").replace("||", " ").strip() for value in values)


def stable_text_fallback(action_type: str, text: str) -> str:
    value = (text or "").strip()
    if value.upper() in UNSTABLE_FALLBACK_TEXTS:
        return ""
    if UUID_LIKE_TOKEN_RE.search(value):
        return ""
    return value


def normalized_source_selector(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.upper().startswith("D") and text[1:].isdigit():
        return str(int(text[1:]))
    return text


def parse_selected(line: str) -> Tuple[int, str, str]:
    match = SELECTED_RE.match(line or "")
    if not match:
        raise PrefixBuildError(f"could not parse selected line: {line}")
    return int(match.group(1)), match.group(2), match.group(3).strip()


def parse_top(line: str) -> Tuple[List[int], List[str], Optional[int]]:
    indices: List[int] = []
    texts: List[str] = []
    starred: Optional[int] = None
    for raw_part in (line or "").split("|"):
        part = raw_part.strip()
        if part.startswith("TOP:"):
            continue
        match = TOP_PART_RE.match(part)
        if not match:
            continue
        idx = int(match.group(2))
        if match.group(1):
            starred = idx
        indices.append(idx)
        texts.append(match.group(4).strip())
    return indices, texts, starred


def parse_top_candidate_count(line: str, parsed_indices: Sequence[int]) -> int:
    match = TOP_N_RE.search(line or "")
    if match:
        return int(match.group(1))
    return len(parsed_indices)


def parse_state_stack(line: str) -> Tuple[str, str]:
    text = (line or "").strip()
    if text.startswith("STATE:"):
        text = text[len("STATE:"):].strip()
    match = STACK_STATE_RE.search(text)
    if not match:
        return "", ""
    return match.group(1), (match.group(2) or "").strip()


def acf_eligible_choice(candidate_count: int, chosen_count: int) -> bool:
    return candidate_count >= 2 and chosen_count >= 1


def is_singleton_combat_pass(
    action_type: str,
    selected_text: str,
    candidate_count: int,
    decision_payloads: Sequence[Dict[str, object]],
) -> bool:
    if decision_payloads:
        return False
    if (action_type or "").upper() not in SINGLETON_COMBAT_PASS_ACTION_TYPES:
        return False
    if candidate_count > 1:
        return False
    return (selected_text or "").strip().upper() in {"", "PASS", "DONE"}


def combat_pass_choice_text(action_type: str, selected_text: str) -> str:
    if (action_type or "").upper() in SINGLETON_COMBAT_PASS_ACTION_TYPES:
        value = (selected_text or "").strip().upper()
        if value in {"", "PASS", "DONE"}:
            return "DONE"
    return selected_text


def is_acf_pre_choose_use_priority_pass(
    decision_number: int,
    turn: str,
    actor: str,
    selected_idx: int,
    selected_text: str,
    action_type: str,
    state_line: str,
    next_decision_line: str,
) -> bool:
    if action_type != "ACTIVATE_ABILITY_OR_SPELL":
        return False
    if selected_idx != 0 or (selected_text or "").strip().lower() != "pass":
        return False
    if "stack ability" not in (state_line or "").lower():
        return False
    next_match = DECISION_RE.match(next_decision_line or "")
    if not next_match:
        return False
    if int(next_match.group(1)) != decision_number + 1:
        return False
    if next_match.group(2) != turn or next_match.group(5).strip() != actor:
        return False
    return "CHOOSE_USE" in next_match.group(4).upper()


def split_cards(value: str) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def parse_london_candidate_names(line: str) -> List[str]:
    match = LONDON_BOTTOM_PROBS_RE.match(line or "")
    if not match:
        return []
    names: List[str] = []
    for raw_card in split_cards(match.group(1)):
        names.append(re.sub(r"\s+\([0-9.]+(?:e[+-]?\d+)?\)$", "", raw_card, flags=re.IGNORECASE).strip())
    return names


def card_choice_indices(candidates: Sequence[str], chosen: Sequence[str]) -> List[int]:
    out: List[int] = []
    used: set[int] = set()
    for wanted in chosen:
        normalized = wanted.strip().lower()
        for idx, candidate in enumerate(candidates):
            if idx in used:
                continue
            if candidate.strip().lower() == normalized:
                out.append(idx)
                used.add(idx)
                break
    return out


def infer_action_type(phase: str, selected_text: str) -> str:
    phase_key = (phase or "").upper()
    combined = f"{phase or ''} {selected_text or ''}".upper()
    if (
        "(DECLARE_ATTACKS)" in phase_key
        or "(DECLARE_ATTACKERS)" in phase_key
        or "DECLARE ATTACKS" in phase_key
        or "DECLARE_ATTACKS" in phase_key
        or "DECLARE ATTACKERS" in phase_key
        or "DECLARE_ATTACKERS" in phase_key
    ):
        return "DECLARE_ATTACKS"
    if (
        "(DECLARE_BLOCKS)" in phase_key
        or "(DECLARE_BLOCKERS)" in phase_key
        or "DECLARE BLOCKS" in phase_key
        or "DECLARE_BLOCKS" in phase_key
        or "DECLARE BLOCKERS" in phase_key
        or "DECLARE_BLOCKERS" in phase_key
    ):
        return "DECLARE_BLOCKS"
    if "TARGET_PICK" in combined:
        return "SELECT_TARGETS"
    if "CARD_PICK" in combined or "SELECT_CARD" in combined:
        return "SELECT_CARD"
    if "CHOOSE_USE" in combined:
        return "CHOOSE_USE"
    if "CHOOSE_MODE" in combined:
        return "CHOOSE_MODE"
    if "ANNOUNCE_X" in combined:
        return "ANNOUNCE_X"
    return "ACTIVATE_ABILITY_OR_SPELL"


def row_for_choice(
    replay: Dict[str, str],
    log_path: Path,
    ordinal: int,
    action_type: str,
    chosen_indices: Sequence[int],
    chosen_texts: Sequence[str],
    source_decision_number: str,
    source_anchor_id: str,
    target_marker: str,
    turn: str,
    phase: str,
    actor: str,
    selected_prob: str,
    selected_line: str,
    top_line: str,
    source_candidate_count: int = -1,
    source_candidate_indices: Sequence[int] = (),
    source_candidate_texts: Sequence[str] = (),
    source_stack_count: str = "",
    source_stack_top: str = "",
) -> Dict[str, str]:
    first_idx = chosen_indices[0] if chosen_indices else -1
    first_text = chosen_texts[0] if chosen_texts else ""
    candidate_count = source_candidate_count if source_candidate_count >= 0 else len(source_candidate_texts)
    return {
        "scenario": replay["scenario"],
        "agent_deck": replay["agent_deck"],
        "opp_deck": replay["opp_deck"],
        "seed": replay["seed"],
        "ordinal": str(ordinal),
        "action_type": action_type,
        "chosen_indices": ";".join(str(idx) for idx in chosen_indices),
        "chosen_texts": join_texts(chosen_texts),
        "best_idx": str(first_idx),
        "best_text": first_text,
        "first_priority_hand": "",
        "first_mulligan_hand": "",
        "source_game_path": str(log_path.resolve()),
        "source_decision_number": source_decision_number,
        "source_anchor_id": source_anchor_id,
        "target_marker": target_marker,
        "turn": turn,
        "phase": phase,
        "actor": actor,
        "selected_prob": selected_prob,
        "selected_line": selected_line,
        "top_line": top_line,
        "source_candidate_count": str(candidate_count),
        "source_candidate_indices": ";".join(str(idx) for idx in source_candidate_indices),
        "source_candidate_texts": join_texts(source_candidate_texts),
        "source_selected_index": str(first_idx),
        "source_selected_text": first_text,
        "source_stable_text_fallback": stable_text_fallback(action_type, first_text),
        "source_hand": "",
        "source_hand_object_ids": "",
        "source_library_top": "",
        "source_library_top_object_ids": "",
        "source_library": "",
        "source_library_object_ids": "",
        "source_random_util_count_before_search": "",
        "source_stack_count": source_stack_count,
        "source_stack_top": source_stack_top,
        "source_candidate_object_ids": "",
        "source_selected_object_ids": "",
        "source_object_ids": "",
        "source_target_object_ids": "",
        "source_identity_required": "false",
        "source_identity_status": "not_required",
    }


def as_list(value: object) -> List[object]:
    return value if isinstance(value, list) else []


def str_list(value: object) -> List[str]:
    return [str(item) for item in as_list(value)]


def str_list_or_scalar(payload: Dict[str, object], array_key: str, scalar_key: str) -> List[str]:
    values = str_list(payload.get(array_key))
    if values:
        return values
    value = payload.get(scalar_key)
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def int_list(value: object) -> List[int]:
    out: List[int] = []
    for item in as_list(value):
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def float_at(values: object, idx: int) -> str:
    items = as_list(values)
    if idx < 0 or idx >= len(items):
        return ""
    try:
        return f"{float(items[idx]):.6f}"
    except (TypeError, ValueError):
        return ""


def row_for_replay_decision_json(
    replay: Dict[str, str],
    log_path: Path,
    payload: Dict[str, object],
    source_index: int,
    one_based_ordinals: bool,
    target_sources: Sequence[str],
    ordinal_override: Optional[int] = None,
    source_decision_number: Optional[str] = None,
    source_anchor_id: Optional[str] = None,
    target_marker: Optional[str] = None,
    turn_override: Optional[str] = None,
    phase_override: Optional[str] = None,
    actor_override: Optional[str] = None,
    source_stack_count: str = "",
    source_stack_top: str = "",
) -> Dict[str, str]:
    action_type = str(payload.get("action_type") or "")
    ordinal_raw = int(payload.get("ordinal", source_index - 1))
    ordinal = ordinal_override if ordinal_override is not None else ordinal_raw + (1 if one_based_ordinals else 0)
    candidates = str_list(payload.get("candidate_texts"))
    chosen_indices = int_list(payload.get("chosen_indices"))
    chosen_texts = str_list(payload.get("chosen_texts"))
    if not chosen_texts:
        chosen_texts = [
            candidates[idx]
            for idx in chosen_indices
            if 0 <= idx < len(candidates)
        ]
    first_idx = chosen_indices[0] if chosen_indices else -1
    first_text = chosen_texts[0] if chosen_texts else (
        candidates[first_idx] if 0 <= first_idx < len(candidates) else ""
    )
    source_id = source_decision_number if source_decision_number is not None else f"M{source_index}"
    phase = phase_override if phase_override is not None else str(payload.get("phase") or "")
    if not phase:
        phase = "London Mulligan" if action_type == "LONDON_MULLIGAN" else "Mulligan"
    selected_prob = float_at(payload.get("candidate_probs"), first_idx)
    selected_line = (
        f"REPLAY_DECISION_JSON ordinal={ordinal_raw} decision_number={payload.get('decision_number', '')} "
        f"action_type={action_type} chosen_indices={';'.join(str(idx) for idx in chosen_indices)}"
    )
    top_line = "REPLAY_DECISION_JSON candidates=" + join_texts(candidates)
    target_set = {value.strip() for value in target_sources if value.strip()}
    row = row_for_choice(
        replay,
        log_path,
        ordinal,
        action_type,
        chosen_indices,
        chosen_texts,
        source_id,
        source_anchor_id if source_anchor_id is not None else f"{log_path.stem}_M{source_index:03d}",
        target_marker if target_marker is not None else ("target" if source_id in target_set else "prefix"),
        turn_override if turn_override is not None else str(payload.get("turn") or ""),
        phase,
        actor_override if actor_override is not None else str(payload.get("actor") or payload.get("player") or "PlayerRL1"),
        selected_prob,
        selected_line,
        top_line,
        len(candidates),
        list(range(len(candidates))),
        candidates,
        source_stack_count,
        source_stack_top,
    )
    row["source_hand"] = join_texts(str_list(payload.get("hand")))
    row["source_library_top"] = join_texts(str_list(payload.get("library_top")))
    row["source_library"] = join_texts(str_list(payload.get("library")))
    row["source_hand_object_ids"] = join_texts(str_list(payload.get("hand_object_ids")))
    row["source_library_top_object_ids"] = join_texts(str_list(payload.get("library_top_object_ids")))
    row["source_library_object_ids"] = join_texts(str_list(payload.get("library_object_ids")))
    candidate_object_ids = str_list(payload.get("candidate_object_ids"))
    selected_object_ids = str_list(payload.get("chosen_object_ids"))
    if not selected_object_ids:
        selected_object_ids = str_list(payload.get("selected_object_ids"))
    if not selected_object_ids and candidate_object_ids:
        selected_object_ids = [
            candidate_object_ids[idx]
            for idx in chosen_indices
            if 0 <= idx < len(candidate_object_ids)
        ]
    row["source_candidate_object_ids"] = join_texts(candidate_object_ids)
    row["source_selected_object_ids"] = join_texts(selected_object_ids)
    row["source_object_ids"] = join_texts(str_list_or_scalar(payload, "source_object_ids", "source_id"))
    target_object_ids = str_list_or_scalar(payload, "target_object_ids", "target_object_id")
    if not target_object_ids and action_type.upper() == "SELECT_TARGETS":
        target_object_ids = selected_object_ids
    row["source_target_object_ids"] = join_texts(target_object_ids)
    return row


def read_replay_decision_pregame_rows(
    lines: Sequence[str],
    replay: Dict[str, str],
    log_path: Path,
    one_based_ordinals: bool,
    target_sources: Sequence[str],
    ordinal_space: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in lines:
        if line.startswith("TURN "):
            break
        if not line.startswith(REPLAY_DECISION_JSON_PREFIX):
            continue
        raw = line[len(REPLAY_DECISION_JSON_PREFIX):].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PrefixBuildError(f"could not parse replay decision JSON: {exc}: {line}") from exc
        action_type = str(payload.get("action_type") or "")
        if action_type not in {"MULLIGAN", "LONDON_MULLIGAN"}:
            continue
        if ordinal_space == ORDINAL_SPACE_ACF:
            candidate_count = len(str_list(payload.get("candidate_texts")))
            chosen_count = len(int_list(payload.get("chosen_indices"))) or len(str_list(payload.get("chosen_texts")))
            if not acf_eligible_choice(candidate_count, chosen_count):
                continue
        rows.append(
            row_for_replay_decision_json(
                replay,
                log_path,
                payload,
                len(rows) + 1,
                one_based_ordinals,
                target_sources,
            )
        )
    return rows


def read_replay_decision_payloads_by_decision(lines: Sequence[str]) -> Dict[int, List[Dict[str, object]]]:
    payloads: Dict[int, List[Dict[str, object]]] = {}
    for line in lines:
        if not line.startswith(REPLAY_DECISION_JSON_PREFIX):
            continue
        raw = line[len(REPLAY_DECISION_JSON_PREFIX):].strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PrefixBuildError(f"could not parse replay decision JSON: {exc}: {line}") from exc
        try:
            decision_number = int(payload.get("decision_number", -1))
        except (TypeError, ValueError):
            decision_number = -1
        if decision_number <= 0:
            continue
        payloads.setdefault(decision_number, []).append(payload)
    return payloads


def replay_payload_candidate_count(payload: Dict[str, object]) -> int:
    try:
        return int(payload.get("candidate_count") or len(str_list(payload.get("candidate_texts"))))
    except (TypeError, ValueError):
        return len(str_list(payload.get("candidate_texts")))


def replay_payload_chosen_count(payload: Dict[str, object]) -> int:
    return len(int_list(payload.get("chosen_indices"))) or len(str_list(payload.get("chosen_texts")))


def replay_payload_action_type(payload: Dict[str, object]) -> str:
    return str(payload.get("action_type") or "").strip().upper()


def ordered_replay_decision_payloads(payloads: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = list(payloads)
    has_activation = any(replay_payload_action_type(payload) == "ACTIVATE_ABILITY_OR_SPELL" for payload in out)
    if has_activation:
        out = [payload for payload in out if replay_payload_action_type(payload) != "SELECT_TARGETS"]
    if has_activation:
        order = {
            "ACTIVATE_ABILITY_OR_SPELL": 0,
            "CHOOSE_MODE": 1,
            "ANNOUNCE_X": 1,
            "SELECT_TARGETS": 2,
        }
        out = [
            payload
            for _, payload in sorted(
                enumerate(out),
                key=lambda item: (order.get(replay_payload_action_type(item[1]), 1), item[0]),
            )
        ]
    return out


def read_pregame_rows(
    lines: Sequence[str],
    replay: Dict[str, str],
    log_path: Path,
    one_based_ordinals: bool,
    target_sources: Sequence[str],
    ordinal_space: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    target_set = {value.strip() for value in target_sources if value.strip()}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("TURN "):
            break
        mulligan = MULLIGAN_RE.match(line)
        if mulligan:
            decision = mulligan.group(1)
            hand = join_texts(split_cards(mulligan.group(2)))
            choice_idx = 0 if decision == "KEEP" else 1
            if ordinal_space == ORDINAL_SPACE_ACF and not acf_eligible_choice(2, 1):
                idx += 1
                continue
            ordinal = len(rows) + (1 if one_based_ordinals else 0)
            source_id = f"M{len(rows) + 1}"
            rows.append(
                row_for_choice(
                    replay,
                    log_path,
                    ordinal,
                    "MULLIGAN",
                    [choice_idx],
                    [decision],
                    source_id,
                    f"{log_path.stem}_M{len(rows) + 1:03d}",
                    "target" if source_id in target_set else "prefix",
                    "",
                    "Mulligan",
                    "PlayerRL1",
                    "",
                    line,
                    "TOP: n=2 | [0] KEEP | [1] MULLIGAN",
                    2,
                    [0, 1],
                    ["KEEP", "MULLIGAN"],
                )
            )
            rows[-1]["source_hand"] = hand
            idx += 1
            continue
        bottom = LONDON_BOTTOM_RE.match(line)
        if bottom:
            ranked_cards = parse_london_candidate_names(lines[idx + 1] if idx + 1 < len(lines) else "")
            if not ranked_cards:
                raise PrefixBuildError(f"could not parse London ranked cards in line: {line}")
            if ordinal_space == ORDINAL_SPACE_ACF and not acf_eligible_choice(len(ranked_cards), len(ranked_cards)):
                idx += 1
                continue
            # ComputerPlayerRL records LONDON_MULLIGAN as the full best-to-worst
            # card ranking; the engine bottoms the last N cards afterward.
            chosen_indices = list(range(len(ranked_cards)))
            ordinal = len(rows) + (1 if one_based_ordinals else 0)
            source_id = f"M{len(rows) + 1}"
            rows.append(
                row_for_choice(
                    replay,
                    log_path,
                    ordinal,
                    "LONDON_MULLIGAN",
                    chosen_indices,
                    ranked_cards,
                    source_id,
                    f"{log_path.stem}_M{len(rows) + 1:03d}",
                    "target" if source_id in target_set else "prefix",
                    "",
                    f"London Mulligan bottomN={bottom.group(1)}",
                    "PlayerRL1",
                    "",
                    line,
                    lines[idx + 1].strip() if idx + 1 < len(lines) else "",
                    len(ranked_cards),
                    list(range(len(ranked_cards))),
                    ranked_cards,
                )
            )
            idx += 1
        idx += 1
    return rows


def opening_state_row(rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    for row in rows:
        if (
            row.get("action_type") == "MULLIGAN"
            and (row.get("chosen_texts") or "").strip().upper() == "KEEP"
            and row.get("source_hand")
        ):
            return row
    for row in rows:
        if row.get("source_hand"):
            return row
    return None


def apply_opening_state(rows: Sequence[Dict[str, str]], source: Dict[str, str]) -> bool:
    hand = source.get("source_hand", "")
    if not hand:
        return False
    for row in rows:
        if not str(row.get("source_decision_number", "")).isdigit():
            continue
        if not row.get("first_priority_hand"):
            row["first_priority_hand"] = hand
        if not row.get("source_hand"):
            row["source_hand"] = hand
        if source.get("source_library_top") and not row.get("source_library_top"):
            row["source_library_top"] = source.get("source_library_top", "")
        if source.get("source_library") and not row.get("source_library"):
            row["source_library"] = source.get("source_library", "")
        return True
    return False


def renumber_rows(rows: Sequence[Dict[str, str]], one_based_ordinals: bool) -> None:
    base = 1 if one_based_ordinals else 0
    for idx, row in enumerate(rows):
        row["ordinal"] = str(idx + base)


def object_text_prefix_row(row: Dict[str, str]) -> bool:
    action_type = str(row.get("action_type", "")).upper()
    if action_type in OBJECT_TEXT_ACTION_TYPES:
        return True
    if singleton_combat_pass_prefix_row(row):
        return True
    phase = str(row.get("phase", "")).upper()
    return any(marker in phase for marker in OBJECT_TEXT_PHASE_MARKERS)


def singleton_combat_pass_prefix_row(row: Dict[str, str]) -> bool:
    action_type = str(row.get("action_type", "")).upper()
    if action_type not in SINGLETON_COMBAT_PASS_ACTION_TYPES:
        return False
    try:
        candidate_count = int(str(row.get("source_candidate_count", "") or "0"))
    except ValueError:
        candidate_count = 0
    if candidate_count > 1:
        return False
    texts = split_joined_texts(str(row.get("chosen_texts", "")))
    if not texts:
        best_text = str(row.get("best_text", "")).strip()
        if best_text:
            texts.append(best_text)
    if not texts:
        selected_text = str(row.get("source_selected_text", "")).strip()
        if selected_text:
            texts.append(selected_text)
    return any(text.strip().upper() == "DONE" for text in texts)


def split_joined_texts(value: str) -> List[str]:
    return [part.strip() for part in (value or "").split("||") if part.strip()]


def parse_semicolon_ints(value: str) -> List[int]:
    out: List[int] = []
    for part in (value or "").split(";"):
        text = part.strip()
        if not text:
            continue
        try:
            out.append(int(text))
        except ValueError:
            continue
    return out


def uuid_like_select_card_prefix_row(row: Dict[str, str]) -> bool:
    if str(row.get("action_type", "")).upper() != "SELECT_CARD":
        return False
    texts = split_joined_texts(str(row.get("chosen_texts", "")))
    best_text = str(row.get("best_text", "")).strip()
    if best_text:
        texts.append(best_text)
    return any(UUID_LIKE_TEXT_RE.fullmatch(text) for text in texts)


def apply_prefix_choice_mode(
    rows: Sequence[Dict[str, str]],
    prefix_choice_mode: str,
    retain_prefix_text_sources: Sequence[str] = (),
    index_only_prefix_targets: bool = False,
) -> None:
    retain_sources = {normalized_source_selector(value) for value in retain_prefix_text_sources if value}
    for row in rows:
        if row.get("target_marker") != "prefix":
            continue
        if not str(row.get("source_decision_number", "")).isdigit():
            continue
        source_key = normalized_source_selector(str(row.get("source_decision_number", "")))
        fallback = str(row.get("source_stable_text_fallback", "")).strip()
        if source_key in retain_sources and fallback:
            row["chosen_texts"] = fallback
            row["best_text"] = fallback
            continue
        if index_only_prefix_targets and str(row.get("action_type", "")).upper() == "SELECT_TARGETS":
            row["chosen_texts"] = ""
            row["best_text"] = ""
            continue
        if uuid_like_select_card_prefix_row(row):
            row["chosen_texts"] = ""
            row["best_text"] = ""
            continue
        if prefix_choice_mode not in {PREFIX_CHOICE_INDEX, PREFIX_CHOICE_OBJECT_TEXT}:
            continue
        if prefix_choice_mode == PREFIX_CHOICE_OBJECT_TEXT and object_text_prefix_row(row):
            continue
        row["chosen_texts"] = ""
        row["best_text"] = ""


def object_identity_required(row: Dict[str, str]) -> bool:
    if row.get("target_marker") != "prefix":
        return False
    if not str(row.get("source_decision_number", "")).isdigit():
        return False
    return str(row.get("action_type", "")).upper() in IDENTITY_REQUIRED_ACTION_TYPES


def identity_status_for_row(row: Dict[str, str]) -> str:
    candidate_ids = split_joined_object_ids(str(row.get("source_candidate_object_ids", "")))
    selected_ids = [
        object_id
        for object_id in split_joined_object_ids(str(row.get("source_selected_object_ids", "")))
        if object_id
    ]
    chosen_indices = parse_semicolon_ints(str(row.get("chosen_indices", "")))
    if not candidate_ids or not selected_ids or not chosen_indices:
        return "identity_unverifiable_display_text_only"
    derived_selected_ids: List[str] = []
    for idx in chosen_indices:
        if idx < 0 or idx >= len(candidate_ids):
            return "identity_unverifiable_display_text_only"
        candidate_id = candidate_ids[idx].strip()
        if not candidate_id:
            return "identity_unverifiable_display_text_only"
        derived_selected_ids.append(candidate_id)
    if selected_ids == derived_selected_ids:
        return "stable_object_ids"
    return "identity_mismatch"


def annotate_identity_status(rows: Sequence[Dict[str, str]]) -> None:
    for row in rows:
        required = object_identity_required(row)
        row["source_identity_required"] = "true" if required else "false"
        if not required:
            row["source_identity_status"] = "not_required"
            continue
        row["source_identity_status"] = identity_status_for_row(row)


def read_log_rows(
    log_path: Path,
    through_decision: int,
    target_decisions: Sequence[int],
    one_based_ordinals: bool,
    include_mulligans: bool,
    target_sources: Sequence[str],
    ordinal_space: str,
    skip_pre_choose_use_priority_pass: bool,
    append_uuid_select_card_stop: bool,
    preserve_opening_state_from_pregame: bool,
    include_singleton_combat_passes: bool,
    singleton_combat_pass_sources: Sequence[str],
) -> List[Dict[str, str]]:
    if through_decision <= 0:
        raise PrefixBuildError("--through-decision must be positive")

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    replay_line = next((line for line in lines if line.startswith("REPLAY:")), "")
    replay = parse_replay_metadata(replay_line)
    replay_decision_by_number = read_replay_decision_payloads_by_decision(lines)
    required = ["scenario", "seed", "agent_deck", "opp_deck"]
    missing = [name for name in required if not replay.get(name)]
    if missing:
        raise PrefixBuildError("missing replay metadata fields: " + ", ".join(missing))

    target_set = {int(value) for value in target_decisions}
    rows: List[Dict[str, str]] = []
    pregame_rows: List[Dict[str, str]] = []
    if include_mulligans:
        replay_decision_rows = read_replay_decision_pregame_rows(
            lines,
            replay,
            log_path,
            one_based_ordinals,
            target_sources,
            ordinal_space,
        )
        pregame_rows = (
            replay_decision_rows
            if replay_decision_rows
            else read_pregame_rows(lines, replay, log_path, one_based_ordinals, target_sources, ordinal_space)
        )
        rows.extend(pregame_rows)
    elif preserve_opening_state_from_pregame:
        replay_decision_rows = read_replay_decision_pregame_rows(
            lines,
            replay,
            log_path,
            one_based_ordinals,
            target_sources,
            ordinal_space,
        )
        pregame_rows = (
            replay_decision_rows
            if replay_decision_rows
            else read_pregame_rows(lines, replay, log_path, one_based_ordinals, target_sources, ordinal_space)
        )
    parsed_decision_rows = 0
    seen_decisions: set[int] = set()
    target_source_set = {value.strip() for value in target_sources if value.strip()}
    singleton_combat_pass_source_set = {
        normalized_source_selector(value)
        for value in singleton_combat_pass_sources
        if value
    }
    idx = 0
    while idx < len(lines):
        decision_match = DECISION_RE.match(lines[idx])
        if not decision_match:
            idx += 1
            continue
        decision_number = int(decision_match.group(1))
        if decision_number > through_decision:
            break
        parsed_decision_rows += 1
        seen_decisions.add(decision_number)
        turn = decision_match.group(2)
        phase = decision_match.group(4).strip()
        actor = decision_match.group(5).strip()

        block: List[str] = []
        idx += 1
        while idx < len(lines) and not DECISION_RE.match(lines[idx]):
            block.append(lines[idx])
            idx += 1

        selected_line = next((line.strip() for line in block if line.strip().startswith("SELECTED[")), "")
        state_line = next((line.strip() for line in block if line.strip().startswith("STATE:")), "")
        top_line = next((line.strip() for line in block if line.strip().startswith("TOP:")), "")
        source_stack_count, source_stack_top = parse_state_stack(state_line)
        selected_idx, selected_prob, selected_text = parse_selected(selected_line)
        top_indices, top_texts, starred_idx = parse_top(top_line)
        top_candidate_count = parse_top_candidate_count(top_line, top_indices)
        action_type = infer_action_type(phase, selected_text)
        source_decision_key = normalized_source_selector(str(decision_number))
        decision_payloads = replay_decision_by_number.get(decision_number, [])
        if decision_payloads:
            json_candidate_count = max(replay_payload_candidate_count(payload) for payload in decision_payloads)
            json_chosen_count = max(replay_payload_chosen_count(payload) for payload in decision_payloads)
        else:
            json_candidate_count = top_candidate_count
            json_chosen_count = 1
        include_singleton_combat_pass = (
            (include_singleton_combat_passes or source_decision_key in singleton_combat_pass_source_set)
            and ordinal_space == ORDINAL_SPACE_ACF
            and is_singleton_combat_pass(action_type, selected_text, json_candidate_count, decision_payloads)
        )
        if (
            ordinal_space == ORDINAL_SPACE_ACF
            and not acf_eligible_choice(json_candidate_count, json_chosen_count)
            and not include_singleton_combat_pass
        ):
            continue
        if (
            skip_pre_choose_use_priority_pass
            and ordinal_space == ORDINAL_SPACE_ACF
            and is_acf_pre_choose_use_priority_pass(
            decision_number,
            turn,
            actor,
            selected_idx,
            selected_text,
            action_type,
            state_line,
            lines[idx] if idx < len(lines) else "",
            )
        ):
            continue
        if starred_idx is not None and starred_idx != selected_idx:
            raise PrefixBuildError(
                f"D{decision_number:03d}: selected index {selected_idx} differs from starred top index {starred_idx}"
            )
        top_text_by_index = dict(zip(top_indices, top_texts))
        if selected_idx in top_text_by_index:
            selected_text = top_text_by_index[selected_idx] or selected_text

        target_marker = "target" if decision_number in target_set or str(decision_number) in target_source_set else "prefix"
        if decision_payloads:
            emitted_payload = False
            for decision_payload in ordered_replay_decision_payloads(decision_payloads):
                payload_candidate_count = replay_payload_candidate_count(decision_payload)
                payload_chosen_count = replay_payload_chosen_count(decision_payload)
                if ordinal_space == ORDINAL_SPACE_ACF and not acf_eligible_choice(
                    payload_candidate_count,
                    payload_chosen_count,
                ):
                    continue
                ordinal = len(rows) + (1 if one_based_ordinals else 0)
                rows.append(
                    row_for_replay_decision_json(
                        replay,
                        log_path,
                        decision_payload,
                        decision_number,
                        one_based_ordinals,
                        target_sources,
                        ordinal_override=ordinal,
                        source_decision_number=str(decision_number),
                        source_anchor_id=f"{log_path.stem}_D{decision_number:03d}",
                        target_marker=target_marker,
                        turn_override=turn,
                        phase_override=str(decision_payload.get("phase") or phase),
                        actor_override=actor,
                        source_stack_count=source_stack_count,
                        source_stack_top=source_stack_top,
                    )
                )
                attach_declare_blocks_object_ids(rows[-1], block)
                emitted_payload = True
            if not emitted_payload:
                continue
        else:
            ordinal = len(rows) + (1 if one_based_ordinals else 0)
            emitted_selected_text = combat_pass_choice_text(action_type, selected_text)
            emitted_candidate_texts = (
                [emitted_selected_text]
                if include_singleton_combat_pass
                else top_texts
            )
            emitted_candidate_indices = (
                [0]
                if include_singleton_combat_pass
                else top_indices
            )
            rows.append(
                row_for_choice(
                    replay,
                    log_path,
                    ordinal,
                    action_type,
                    [selected_idx],
                    [emitted_selected_text],
                    str(decision_number),
                    f"{log_path.stem}_D{decision_number:03d}",
                    target_marker,
                    turn,
                    phase,
                    actor,
                    selected_prob,
                    selected_line,
                    top_line,
                    max(1, top_candidate_count) if include_singleton_combat_pass else top_candidate_count,
                    emitted_candidate_indices,
                    emitted_candidate_texts,
                    source_stack_count,
                    source_stack_top,
                )
            )
            attach_declare_blocks_object_ids(rows[-1], block)
        next_match = DECISION_RE.match(lines[idx] if idx < len(lines) else "")
        next_phase = next_match.group(4).strip() if next_match else ""
        if (
            append_uuid_select_card_stop
            and target_marker == "prefix"
            and action_type == "SELECT_CARD"
            and UUID_LIKE_TEXT_RE.fullmatch(selected_text or "")
            and "CARD_PICK" in (phase or "").upper()
            and "CARD_PICK" not in (next_phase or "").upper()
        ):
            stop_ordinal = len(rows) + (1 if one_based_ordinals else 0)
            rows.append(
                row_for_choice(
                    replay,
                    log_path,
                    stop_ordinal,
                    "SELECT_CARD",
                    [0],
                    ["STOP"],
                    f"{decision_number}_STOP",
                    f"{log_path.stem}_D{decision_number:03d}_STOP",
                    "prefix",
                    turn,
                    phase,
                    actor,
                    "0.000000",
                    f"SYNTHETIC_SELECT_CARD_STOP after D{decision_number:03d}",
                    "SYNTHETIC_SELECT_CARD_STOP",
                    1,
                    [0],
                    ["STOP"],
                    source_stack_count,
                    source_stack_top,
                )
            )
        attach_search_metadata(rows, block)

    if parsed_decision_rows != through_decision:
        missing = [str(i) for i in range(1, through_decision + 1) if i not in seen_decisions]
        raise PrefixBuildError(
            f"expected contiguous decisions 1..{through_decision}, found {parsed_decision_rows} rows; missing {', '.join(missing[:20])}"
        )
    if ordinal_space == ORDINAL_SPACE_ACF:
        renumber_rows(rows, one_based_ordinals)
    if preserve_opening_state_from_pregame and not include_mulligans:
        state = opening_state_row(pregame_rows)
        if state is None:
            raise PrefixBuildError(
                "--preserve-opening-state-from-pregame requested, but no pregame source hand metadata was found"
            )
        if not apply_opening_state(rows, state):
            raise PrefixBuildError(
                "--preserve-opening-state-from-pregame requested, but no decision row could receive the state"
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_requested_targets(
    rows: Sequence[Dict[str, str]],
    target_decisions: Sequence[int],
    target_sources: Sequence[str],
    allow_missing_targets: bool,
) -> None:
    requested_decisions = {str(value) for value in target_decisions}
    requested_sources = {value.strip() for value in target_sources if value.strip()}
    if not requested_decisions and not requested_sources:
        return
    emitted_targets = {
        str(row.get("source_decision_number", "")).strip()
        for row in rows
        if row.get("target_marker") == "target"
    }
    missing_decisions = sorted(
        requested_decisions - emitted_targets,
        key=lambda value: int(value) if value.isdigit() else value,
    )
    missing_sources = sorted(requested_sources - emitted_targets)
    if not missing_decisions and not missing_sources:
        return
    message = (
        "requested target rows were absent after filtering: "
        f"target_decisions={missing_decisions or []} "
        f"target_sources={missing_sources or []}; "
        "in --ordinal-space acf, singleton/no-op source decisions are skipped. "
        "Choose an emitted target row or pass --allow-missing-targets for legacy prefix-only output."
    )
    if allow_missing_targets:
        print("WARNING: " + message)
        return
    raise PrefixBuildError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Compact game log path.")
    parser.add_argument("--output", type=Path, required=True, help="Replay CSV output path.")
    parser.add_argument("--through-decision", type=int, default=138, help="Include displayed decisions up to this number.")
    parser.add_argument(
        "--target-decision",
        action="append",
        type=int,
        default=[],
        help="Displayed decision number to mark as target in metadata. Can be repeated.",
    )
    parser.add_argument(
        "--target-source",
        action="append",
        default=[],
        help="Raw source_decision_number to mark as target, such as M5 for pregame diagnostics.",
    )
    parser.add_argument(
        "--one-based-ordinals",
        action="store_true",
        help="Diagnostic mode: write ordinal=N instead of the default replay ordinal=N-1.",
    )
    parser.add_argument(
        "--include-mulligans",
        action="store_true",
        help="Prepend compact-log mulligan and London-bottom choices to the forced prefix path.",
    )
    parser.add_argument(
        "--ordinal-space",
        choices=[ORDINAL_SPACE_SOURCE, ORDINAL_SPACE_ACF],
        default=ORDINAL_SPACE_SOURCE,
        help=(
            "Ordinal stream to emit. 'source' preserves compact-log ordinals for compatibility; "
            "'acf' skips compact rows with candidate_count < 2 and remaps ordinals to the "
            "ActionCounterfactual eligible-decision stream."
        ),
    )
    parser.add_argument(
        "--prefix-choice-mode",
        choices=[PREFIX_CHOICE_TEXT, PREFIX_CHOICE_INDEX, PREFIX_CHOICE_OBJECT_TEXT],
        default=PREFIX_CHOICE_TEXT,
        help=(
            "How to encode non-target compact decision prefix rows. 'text' emits chosen_texts/best_text; "
            "'index' leaves those fields blank so forced-prefix replay matches prefix rows by index only; "
            "'object-text' keeps text for object/card/target prefix choices while leaving spell/ability rows "
            "index-only to tolerate source/candidate wording drift. Target rows keep their text for replay diagnostics."
        ),
    )
    parser.add_argument(
        "--retain-prefix-text-source",
        action="append",
        default=[],
        help=(
            "Retain the stable selected text for a specific non-target source decision even when "
            "--prefix-choice-mode would normally blank prefix text. Accepts values like 13 or D013; "
            "use for local diagnostics after an out-of-range prefix index exposes source/replay state drift."
        ),
    )
    parser.add_argument(
        "--skip-pre-choose-use-priority-pass",
        action="store_true",
        help=(
            "Diagnostic ACF mode: skip source ACTIVATE Pass rows immediately before same-turn CHOOSE_USE "
            "trigger prompts. Keep disabled for normal parity checks because replay can surface those priority rows."
        ),
    )
    parser.add_argument(
        "--append-uuid-select-card-stop",
        action="store_true",
        help=(
            "Diagnostic ACF mode: after the last UUID-like compact CARD_PICK row in a source sequence, append a "
            "synthetic SELECT_CARD STOP prefix row. Use only when source card-pick UUIDs are replay-unstable and "
            "the replay exposes an extra UUID-like card-pick prompt before the next source decision."
        ),
    )
    parser.add_argument(
        "--preserve-opening-state-from-pregame",
        action="store_true",
        help=(
            "When mulligan rows are omitted, copy the source KEEP opening hand/library metadata from the "
            "pregame replay row onto the first emitted decision row so non-pregame forced-prefix replay can "
            "stack that opening state before D001."
        ),
    )
    parser.add_argument(
        "--index-only-prefix-targets",
        action="store_true",
        help=(
            "Blank chosen_texts/best_text for non-target SELECT_TARGETS prefix rows so forced-prefix replay "
            "matches those target-selection prompts by index only. Target rows still keep text for diagnostics."
        ),
    )
    parser.add_argument(
        "--include-singleton-combat-passes",
        action="store_true",
        help=(
            "In --ordinal-space acf, retain compact singleton Declare Attackers/Blockers pass rows as DONE "
            "prefix choices. This is for logs where replay surfaces those compact pass rows as non-singleton "
            "combat prompts before the next ACF target."
        ),
    )
    parser.add_argument(
        "--include-singleton-combat-pass-source",
        action="append",
        default=[],
        help=(
            "Retain one compact singleton Declare Attackers/Blockers pass row by source decision number "
            "(for example D067 or 67). Can be repeated. This is safer than retaining all singleton combat passes."
        ),
    )
    parser.add_argument(
        "--allow-missing-targets",
        action="store_true",
        help=(
            "Legacy/diagnostic escape hatch: allow requested --target-decision or --target-source values to be "
            "absent after filtering. By default this is an error so ACF-space singleton targets do not silently "
            "produce prefix-only CSVs."
        ),
    )
    parser.add_argument(
        "--fail-on-unverifiable-object-identity",
        action="store_true",
        help=(
            "Fail closed when any object-sensitive prefix row lacks stable object ids. This prevents "
            "DECLARE_BLOCKS/SELECT_TARGETS/SELECT_CARD rows from being counted as clean by index/display text only."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_log_rows(
        args.log,
        args.through_decision,
        args.target_decision,
        args.one_based_ordinals,
        args.include_mulligans,
        args.target_source,
        args.ordinal_space,
        args.skip_pre_choose_use_priority_pass,
        args.append_uuid_select_card_stop,
        args.preserve_opening_state_from_pregame,
        args.include_singleton_combat_passes,
        args.include_singleton_combat_pass_source,
    )
    apply_prefix_choice_mode(
        rows,
        args.prefix_choice_mode,
        args.retain_prefix_text_source,
        args.index_only_prefix_targets,
    )
    annotate_identity_status(rows)
    validate_requested_targets(
        rows,
        args.target_decision,
        args.target_source,
        args.allow_missing_targets,
    )
    identity_unverifiable_prefix_rows = [
        row
        for row in rows
        if row.get("target_marker") == "prefix"
        and row.get("source_identity_status") in {"identity_unverifiable_display_text_only", "identity_mismatch"}
    ]
    if args.fail_on_unverifiable_object_identity and identity_unverifiable_prefix_rows:
        first = identity_unverifiable_prefix_rows[0]
        raise PrefixBuildError(
            "identity_unverifiable: "
            f"{len(identity_unverifiable_prefix_rows)} object-sensitive prefix row(s) lack matching stable object ids; "
            f"first source_decision_number={first.get('source_decision_number', '')} "
            f"action_type={first.get('action_type', '')} phase={first.get('phase', '')} "
            f"identity_status={first.get('source_identity_status', '')}"
        )
    emitted_decision_rows = sum(1 for row in rows if str(row["source_decision_number"]).isdigit())
    write_csv(args.output, rows)
    summary = {
        "output": str(args.output),
        "log": str(args.log),
        "rows": len(rows),
        "pregame_rows": sum(1 for row in rows if str(row["source_decision_number"]).startswith("M")),
        "decision_rows": emitted_decision_rows,
        "skipped_decision_rows": max(0, args.through_decision - emitted_decision_rows)
        if args.ordinal_space == ORDINAL_SPACE_ACF else 0,
        "through_decision": args.through_decision,
        "ordinal_base": "one_based" if args.one_based_ordinals else "zero_based",
        "ordinal_space": args.ordinal_space,
        "prefix_choice_mode": args.prefix_choice_mode,
        "retain_prefix_text_sources": args.retain_prefix_text_source,
        "skip_pre_choose_use_priority_pass": args.skip_pre_choose_use_priority_pass,
        "append_uuid_select_card_stop": args.append_uuid_select_card_stop,
        "preserve_opening_state_from_pregame": args.preserve_opening_state_from_pregame,
        "index_only_prefix_targets": args.index_only_prefix_targets,
        "include_singleton_combat_passes": args.include_singleton_combat_passes,
        "include_singleton_combat_pass_sources": args.include_singleton_combat_pass_source,
        "allow_missing_targets": args.allow_missing_targets,
        "first_priority_hand_rows": sum(1 for row in rows if row.get("first_priority_hand")),
        "source_hand_rows": sum(1 for row in rows if row.get("source_hand")),
        "source_hand_object_id_rows": sum(1 for row in rows if row.get("source_hand_object_ids")),
        "source_library_top_rows": sum(1 for row in rows if row.get("source_library_top")),
        "source_library_top_object_id_rows": sum(
            1
            for row in rows
            if row.get("source_library_top_object_ids")
        ),
        "source_library_rows": sum(1 for row in rows if row.get("source_library")),
        "source_library_object_id_rows": sum(1 for row in rows if row.get("source_library_object_ids")),
        "source_object_id_rows": sum(1 for row in rows if row.get("source_object_ids")),
        "source_target_object_id_rows": sum(1 for row in rows if row.get("source_target_object_ids")),
        "source_random_util_count_before_search_rows": sum(
            1
            for row in rows
            if row.get("source_random_util_count_before_search")
        ),
        "source_stack_known_rows": sum(
            1
            for row in rows
            if row.get("source_stack_count")
        ),
        "source_stack_nonempty_rows": sum(
            1
            for row in rows
            if row.get("source_stack_count") not in {"", "0"}
        ),
        "source_candidate_metadata_rows": sum(
            1
            for row in rows
            if row.get("source_candidate_texts")
        ),
        "source_identity_required_rows": sum(
            1
            for row in rows
            if row.get("source_identity_required") == "true"
        ),
        "source_identity_stable_rows": sum(
            1
            for row in rows
            if row.get("source_identity_status") == "stable_object_ids"
        ),
        "source_identity_unverifiable_rows": sum(
            1
            for row in rows
            if row.get("source_identity_status") == "identity_unverifiable_display_text_only"
        ),
        "source_identity_mismatch_rows": sum(
            1
            for row in rows
            if row.get("source_identity_status") == "identity_mismatch"
        ),
        "source_identity_unverifiable_prefix_rows": len(identity_unverifiable_prefix_rows),
        "replay_decision_json_rows": sum(
            1
            for row in rows
            if str(row.get("selected_line", "")).startswith("REPLAY_DECISION_JSON")
        ),
        "stable_text_fallback_rows": sum(
            1
            for row in rows
            if row.get("source_stable_text_fallback")
        ),
        "retained_prefix_text_source_rows": sum(
            1
            for row in rows
            if normalized_source_selector(row.get("source_decision_number", ""))
            in {normalized_source_selector(value) for value in args.retain_prefix_text_source}
            and bool(row.get("chosen_texts"))
        ),
        "synthetic_uuid_select_card_stop_rows": sum(
            1
            for row in rows
            if str(row.get("source_decision_number", "")).endswith("_STOP")
        ),
        "prefix_text_rows": sum(
            1
            for row in rows
            if row.get("target_marker") == "prefix"
            and str(row.get("source_decision_number", "")).isdigit()
            and bool(row.get("chosen_texts"))
        ),
        "uuid_select_card_prefix_text_blanked_rows": sum(
            1
            for row in rows
            if row.get("target_marker") == "prefix"
            and str(row.get("source_decision_number", "")).isdigit()
            and str(row.get("action_type", "")).upper() == "SELECT_CARD"
            and not row.get("chosen_texts")
            and UUID_LIKE_TOKEN_RE.search(row.get("selected_line", ""))
        ),
        "select_targets_prefix_text_blanked_rows": sum(
            1
            for row in rows
            if row.get("target_marker") == "prefix"
            and str(row.get("source_decision_number", "")).isdigit()
            and str(row.get("action_type", "")).upper() == "SELECT_TARGETS"
            and not row.get("chosen_texts")
        ),
        "min_ordinal": int(rows[0]["ordinal"]) if rows else None,
        "max_ordinal": int(rows[-1]["ordinal"]) if rows else None,
        "target_decisions": args.target_decision,
        "target_sources": args.target_source,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrefixBuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
