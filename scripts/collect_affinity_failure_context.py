#!/usr/bin/env python3
"""Collect an Affinity failure-context corpus from compact game logs.

This script is a diagnostics-only helper for MTGRL Experiment 1 (2026-05-18 plan):
given one or more `local-training/local_pbt/cp7_eval_sweeps/<run-id>/game_logs` roots
(or individual `.txt` logs), extract:

- per-game outcome metadata
- per-decision compact state summaries (`STATE:`) and top policy alternatives (`TOP:`)
- simple "policy-miss" heuristics from the visible TOP-k options

It does not attempt counterfactual rollouts / sibling-win verification; that
requires replay/search infrastructure outside these text logs.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from export_game_log_trajectories import iter_logs, parse_game


REPLAY_PREFIX = "REPLAY:"
REPLAY_RANDOM_PREFIX = "REPLAY_RANDOM:"


@dataclasses.dataclass(frozen=True)
class DecisionRow:
    game_path: str
    opponent: str
    rl_won: bool
    decision_kind: str
    decision_number: int
    turn: int
    phase: str
    selected: str
    value_score: Optional[float]
    top_selected_prob: Optional[float]
    top_best_prob: Optional[float]
    top_best_text: Optional[str]
    top_best_is_selected: Optional[bool]
    has_nonpass_option: bool
    has_play_land_option: bool
    has_cast_spy_option: bool
    has_dread_return_option: bool
    state_json: str
    options_json: str
    action_counterfactual_compatible: bool = False
    replay_scenario: str = ""
    replay_seed: str = ""
    random_util_seed: str = ""
    replay_random_scope: str = ""


def normalize_opponent(text: Optional[str]) -> str:
    return (text or "").strip()


def is_affinity(opponent: str, filter_token: str) -> bool:
    token = filter_token.strip().lower()
    if not token:
        return "affinity" in opponent.lower()
    return token in opponent.lower()


def parse_key_values(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for part in text.strip().split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_replay_metadata(log_path: Path) -> Dict[str, str]:
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


def topk_summary(options: List[Dict[str, object]], selected: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[bool]]:
    if not options:
        return None, None, None, None
    selected_prob: Optional[float] = None
    best_prob: Optional[float] = None
    best_text: Optional[str] = None
    best_is_selected: Optional[bool] = None
    for opt in options:
        text = str(opt.get("text", "")).strip()
        prob = opt.get("prob", None)
        try:
            p = float(prob) if prob is not None else None
        except (TypeError, ValueError):
            p = None
        if text == selected and p is not None:
            if selected_prob is None or p > selected_prob:
                selected_prob = p
        if p is not None:
            if best_prob is None or p > best_prob:
                best_prob = p
                best_text = text
    if best_text is not None:
        best_is_selected = best_text == selected
    return selected_prob, best_prob, best_text, best_is_selected


def decision_to_row(
    game: Dict[str, object],
    decision: Dict[str, object],
    replay_metadata: Dict[str, str],
) -> Optional[DecisionRow]:
    opponent = normalize_opponent(game.get("opponent"))
    if not opponent:
        return None
    kind = str(decision.get("kind", "decision"))
    number = int(decision.get("number", 0) or 0)
    turn = int(decision.get("turn", 0) or 0)
    phase = str(decision.get("phase", "") or "")
    selected = str(decision.get("selected", "") or "")
    value_score = decision.get("value_score", None)
    if value_score is not None:
        try:
            value_score = float(value_score)
        except (TypeError, ValueError):
            value_score = None
    options = decision.get("options") or []
    if not isinstance(options, list):
        options = []
    selected_prob, best_prob, best_text, best_is_selected = topk_summary(options, selected)
    flags = decision.get("flags") or {}
    if not isinstance(flags, dict):
        flags = {}
    state_summary = decision.get("state_summary") or {}
    try:
        state_json = json.dumps(state_summary, ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        state_json = "{}"
    try:
        options_json = json.dumps(options, ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        options_json = "[]"
    return DecisionRow(
        game_path=str(game.get("path", "")),
        opponent=opponent,
        rl_won=bool(game.get("rl_won", False)),
        decision_kind=kind,
        decision_number=number,
        turn=turn,
        phase=phase,
        selected=selected,
        value_score=value_score,
        top_selected_prob=selected_prob,
        top_best_prob=best_prob,
        top_best_text=best_text,
        top_best_is_selected=best_is_selected,
        has_nonpass_option=bool(flags.get("has_nonpass_option", False)),
        has_play_land_option=bool(flags.get("has_land_play_option", False)),
        has_cast_spy_option=bool(flags.get("has_cast_spy_option", False)),
        has_dread_return_option=bool(flags.get("has_dread_return_option", False)),
        state_json=state_json,
        options_json=options_json,
        action_counterfactual_compatible=replay_metadata.get("action_counterfactual_compatible", "").lower() == "true",
        replay_scenario=replay_metadata.get("replay_scenario", ""),
        replay_seed=replay_metadata.get("replay_seed", ""),
        random_util_seed=replay_metadata.get("random_util_seed", ""),
        replay_random_scope=replay_metadata.get("replay_random_scope", ""),
    )


def write_jsonl(path: Path, rows: Iterable[DecisionRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dataclasses.asdict(row), ensure_ascii=True) + "\n")
            written += 1
    return written


def write_csv(path: Path, rows: List[DecisionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [f.name for f in dataclasses.fields(DecisionRow)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="Log file, game_logs directory, or cp7_eval_sweeps/<run-id> root. Can be repeated.",
    )
    parser.add_argument(
        "--opponent-filter",
        default="grixis affinity",
        help="Case-insensitive substring filter for opponent label (default: grixis affinity).",
    )
    parser.add_argument("--only-losses", action="store_true", help="Emit only losing-game decisions.")
    parser.add_argument("--max-games", type=int, default=0, help="Stop after this many games (0 = no limit).")
    parser.add_argument("--output-jsonl", required=True, help="Output decision corpus JSONL path.")
    parser.add_argument("--output-csv", default="", help="Optional output CSV path.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path.")
    args = parser.parse_args()

    roots = [Path(p) for p in args.root]
    resolved: List[Path] = []
    for root in roots:
        if root.is_dir() and root.name != "game_logs":
            candidate = root / "game_logs"
            if candidate.exists():
                resolved.append(candidate)
            else:
                resolved.append(root)
        else:
            resolved.append(root)

    games_seen = 0
    games_included = 0
    decision_rows: List[DecisionRow] = []
    loss_games: set[str] = set()
    policy_miss_rows = 0
    for log_path in iter_logs(resolved):
        game = parse_game(log_path, include_state=False)
        if game is None:
            continue
        replay_metadata = parse_replay_metadata(log_path)
        games_seen += 1
        opponent = normalize_opponent(game.get("opponent"))
        if not is_affinity(opponent, args.opponent_filter):
            continue
        rl_won = bool(game.get("rl_won", False))
        if args.only_losses and rl_won:
            continue
        games_included += 1
        if not rl_won:
            loss_games.add(str(game.get("path", "")))
        for decision in game.get("decisions", []) or []:
            if not isinstance(decision, dict):
                continue
            row = decision_to_row(game, decision, replay_metadata)
            if row is None:
                continue
            decision_rows.append(row)
            if row.top_best_is_selected is False:
                policy_miss_rows += 1
        if args.max_games and games_included >= args.max_games:
            break

    out_jsonl = Path(args.output_jsonl)
    out_csv = Path(args.output_csv) if args.output_csv.strip() else None
    out_summary = Path(args.summary_json) if args.summary_json.strip() else None

    written = write_jsonl(out_jsonl, decision_rows)
    if out_csv is not None:
        write_csv(out_csv, decision_rows)

    summary = {
        "games_seen": games_seen,
        "games_included": games_included,
        "loss_games": len(loss_games),
        "decision_rows": len(decision_rows),
        "policy_miss_rows_visible_topk": policy_miss_rows,
        "opponent_filter": args.opponent_filter,
        "only_losses": bool(args.only_losses),
        "output_jsonl": str(out_jsonl),
        "output_csv": str(out_csv) if out_csv is not None else "",
    }
    if out_summary is not None:
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote rows={written} -> {out_jsonl}")
    if out_csv is not None:
        print(f"wrote csv rows={len(decision_rows)} -> {out_csv}")
    if out_summary is not None:
        print(f"wrote summary -> {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
