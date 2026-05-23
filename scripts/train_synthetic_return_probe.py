#!/usr/bin/env python3
"""Train a small terminal-outcome credit probe on exported MTGRL trajectories.

This is an offline diagnostic, not the online trainer. It answers two practical
questions before we wire synthetic returns into PPO:

1. Can decision/state features predict the eventual terminal outcome better
   than the game-level base rate?
2. Does adding a compact previous-decision history window improve prediction?

The output predictions can be inspected as synthetic-return candidates, but
they should not be used for training until attribution quality is checked.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


CARD_KEYWORD_FLAGS = [
    "selected_cast_spy",
    "selected_cast_dread_return",
    "selected_land_grant",
    "has_cast_spy_option",
    "has_dread_return_option",
]

GENERIC_BASE_FLAGS = [
    "selected_pass",
    "selected_play_land",
    "selected_mana_ability",
    "has_land_play_option",
    "has_nonpass_option",
]

BASE_FLAGS = GENERIC_BASE_FLAGS + CARD_KEYWORD_FLAGS

GENERIC_ACTION_KEYWORDS = [
    "Pass",
    "Play ",
    "Cast ",
    "Activate ",
    "Add {",
    "Attack",
    "Block",
    "DONE",
]

CARD_ACTION_KEYWORDS = [
    "Cast Balustrade Spy",
    "Cast Dread Return",
    "Cast Land Grant",
    "Cleansing Wildfire",
    "Deadly Dispute",
    "Ichor Wellspring",
    "Writhing Chrysalis",
]

ACTION_KEYWORDS = GENERIC_ACTION_KEYWORDS + CARD_ACTION_KEYWORDS

STATE_CARD_KEYWORDS = [
    "Balustrade Spy",
    "Dread Return",
    "Lotleth Giant",
    "Land Grant",
    "Lotus Petal",
    "Tinder Wall",
    "Overgrown Battlement",
    "Wall of Roots",
    "Quirion Ranger",
    "Saruli Caretaker",
    "Forest",
    "Swamp",
    "Cleansing Wildfire",
    "Ichor Wellspring",
    "Writhing Chrysalis",
    "Deadly Dispute",
]


def iter_games(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def selected_prob(decision: Dict[str, object]) -> float:
    selected = str(decision.get("selected") or "")
    best = 0.0
    for option in decision.get("options") or []:
        prob = safe_float(option.get("prob"), 0.0)
        if option.get("selected_marker") or str(option.get("text") or "") == selected:
            return prob
        best = max(best, prob)
    return best


def max_prob(decision: Dict[str, object]) -> float:
    probs = [safe_float(o.get("prob"), 0.0) for o in decision.get("options") or []]
    return max(probs) if probs else 0.0


def selected_is_top(decision: Dict[str, object]) -> float:
    sp = selected_prob(decision)
    mp = max_prob(decision)
    return 1.0 if sp >= mp - 1e-6 and mp > 0 else 0.0


def phase_buckets(phase: str) -> Dict[str, float]:
    text = phase.lower()
    return {
        "phase_mulligan": 1.0 if "mulligan" in text else 0.0,
        "phase_precombat": 1.0 if "precombat" in text else 0.0,
        "phase_postcombat": 1.0 if "postcombat" in text else 0.0,
        "phase_attack": 1.0 if "attack" in text else 0.0,
        "phase_block": 1.0 if "block" in text else 0.0,
        "phase_target": 1.0 if "target" in text else 0.0,
    }


def add_player_state_features(out: Dict[str, float], prefix: str, pdata: Dict[str, object], feature_mode: str) -> None:
    out[f"{prefix}_life_norm"] = max(-10.0, min(40.0, safe_float(pdata.get("life"), 0.0))) / 40.0
    for zone, denom in (
        ("hand", 20.0),
        ("permanents", 40.0),
        ("graveyard", 80.0),
        ("exile", 80.0),
    ):
        out[f"{prefix}_{zone}_count_norm"] = min(denom, safe_float(pdata.get(f"{zone}_count"), 0.0)) / denom
    if feature_mode == "generic":
        return
    for zone in ("hand", "permanents", "graveyard"):
        cards = [str(card) for card in (pdata.get(f"{zone}_cards") or [])]
        joined = " | ".join(cards)
        for keyword in STATE_CARD_KEYWORDS:
            key = keyword.replace(" ", "_").replace("'", "").lower()
            out[f"{prefix}_{zone}_has:{key}"] = 1.0 if keyword in joined else 0.0


def state_feature_map(decision: Dict[str, object], feature_mode: str) -> Dict[str, float]:
    summary = decision.get("state_summary") or {}
    players = summary.get("players") or {}
    if not isinstance(players, dict) or not players:
        return {}
    actor_name = str(decision.get("actor") or "")
    out: Dict[str, float] = {}
    actor_state = players.get(actor_name)
    if isinstance(actor_state, dict):
        add_player_state_features(out, "actor", actor_state, feature_mode)
    for name, pdata in players.items():
        if name == actor_name:
            continue
        if isinstance(pdata, dict):
            add_player_state_features(out, "opp", pdata, feature_mode)
            break
    return out


def feature_map(decision: Dict[str, object], feature_mode: str) -> Dict[str, float]:
    selected = str(decision.get("selected") or "")
    options = decision.get("options") or []
    flags = decision.get("flags") or {}
    out: Dict[str, float] = {
        "bias": 1.0,
        "turn_norm": min(20.0, safe_float(decision.get("turn"), 0.0)) / 20.0,
        "value_score": max(-1.0, min(1.0, safe_float(decision.get("value_score"), 0.0))),
        "options_count_norm": min(64.0, len(options)) / 64.0,
        "selected_prob": selected_prob(decision),
        "max_prob": max_prob(decision),
        "selected_is_top": selected_is_top(decision),
        "actor_player_rl1": 1.0 if str(decision.get("actor") or "") == "PlayerRL1" else 0.0,
    }
    base_flags = BASE_FLAGS if feature_mode == "card_keywords" else GENERIC_BASE_FLAGS
    action_keywords = ACTION_KEYWORDS if feature_mode == "card_keywords" else GENERIC_ACTION_KEYWORDS
    for key in base_flags:
        out[key] = 1.0 if flags.get(key) else 0.0
    for key, value in phase_buckets(str(decision.get("phase") or "")).items():
        out[key] = value
    for keyword in action_keywords:
        out[f"selected_contains:{keyword}"] = 1.0 if keyword in selected else 0.0
    out.update(state_feature_map(decision, feature_mode))
    if str(decision.get("kind") or "") == "mulligan":
        out["mulligan_lands_norm"] = min(7.0, safe_float(decision.get("lands"), 0.0)) / 7.0
        out["mulligan_hand_size_norm"] = min(7.0, safe_float(decision.get("hand_size"), 0.0)) / 7.0
        out["mulligans_taken_norm"] = min(6.0, safe_float(decision.get("mulligans_taken"), 0.0)) / 6.0
    return out


def build_rows(games: Sequence[Dict[str, object]], actor: str, history_window: int, feature_mode: str):
    rows = []
    feature_names = set()
    for game_idx, game in enumerate(games):
        label = 1 if game.get("rl_won") else 0
        history: List[Dict[str, float]] = []
        for decision_idx, decision in enumerate(game.get("decisions") or []):
            fmap = feature_map(decision, feature_mode)
            if actor != "all" and str(decision.get("actor") or "") != actor:
                history.append(fmap)
                continue
            merged = dict(fmap)
            if history_window > 0:
                recent = history[-history_window:]
                history_flags = BASE_FLAGS if feature_mode == "card_keywords" else GENERIC_BASE_FLAGS
                for key in history_flags:
                    merged[f"hist{history_window}_count:{key}"] = sum(h.get(key, 0.0) for h in recent) / float(history_window)
                merged[f"hist{history_window}_selected_pass_rate"] = sum(h.get("selected_pass", 0.0) for h in recent) / float(history_window)
                merged[f"hist{history_window}_nonpass_option_rate"] = sum(h.get("has_nonpass_option", 0.0) for h in recent) / float(history_window)
            feature_names.update(merged.keys())
            rows.append({
                "game_idx": game_idx,
                "decision_idx": decision_idx,
                "label": label,
                "features": merged,
                "selected": str(decision.get("selected") or ""),
                "phase": str(decision.get("phase") or ""),
                "turn": decision.get("turn"),
                "kind": str(decision.get("kind") or ""),
                "actor": str(decision.get("actor") or ""),
                "path": game.get("path"),
            })
            history.append(fmap)
    return rows, sorted(feature_names)


def tensorize(rows: Sequence[Dict[str, object]], feature_names: Sequence[str]):
    x = torch.zeros((len(rows), len(feature_names)), dtype=torch.float32)
    y = torch.zeros((len(rows),), dtype=torch.float32)
    index = {name: i for i, name in enumerate(feature_names)}
    for r, row in enumerate(rows):
        y[r] = float(row["label"])
        for key, value in row["features"].items():
            i = index.get(key)
            if i is not None:
                x[r, i] = float(value)
    return x, y


def train_linear(x_train, y_train, x_test, y_test, epochs: int, lr: float, seed: int):
    torch.manual_seed(seed)
    weight = torch.zeros((x_train.shape[1],), dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=lr)
    pos = float(y_train.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)
    for _ in range(epochs):
        logits = x_train.mv(weight) + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        loss = loss + 1e-4 * weight.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_logits = x_train.mv(weight) + bias
        test_logits = x_test.mv(weight) + bias if len(x_test) else torch.empty(0)
        train_prob = torch.sigmoid(train_logits)
        test_prob = torch.sigmoid(test_logits)
        train_loss = F.binary_cross_entropy(train_prob, y_train).item()
        test_loss = F.binary_cross_entropy(test_prob, y_test).item() if len(y_test) else 0.0
    return weight.detach(), bias.detach(), train_prob, test_prob, train_loss, test_loss


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--history-window", type=int, default=0)
    parser.add_argument("--actor", default="PlayerRL1", help="PlayerRL1 or all")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--feature-mode",
        choices=("generic", "card_keywords"),
        default="generic",
        help="generic excludes card-name indicators; card_keywords preserves the older diagnostic feature set.",
    )
    parser.add_argument("--predictions-csv", default="")
    parser.add_argument("--weights-csv", default="")
    args = parser.parse_args()

    paths = [Path(p) for p in args.input]
    games = list(iter_games(paths))
    if len(games) < 4:
        raise SystemExit("need at least four games for a train/test split")

    rng = random.Random(args.seed)
    game_indices = list(range(len(games)))
    rng.shuffle(game_indices)
    test_count = max(1, int(round(len(games) * args.test_frac)))
    test_games = set(game_indices[:test_count])

    rows, feature_names = build_rows(games, args.actor, args.history_window, args.feature_mode)
    train_rows = [r for r in rows if r["game_idx"] not in test_games]
    test_rows = [r for r in rows if r["game_idx"] in test_games]
    x_train, y_train = tensorize(train_rows, feature_names)
    x_test, y_test = tensorize(test_rows, feature_names)

    w, b, train_prob, test_prob, train_loss, test_loss = train_linear(
        x_train, y_train, x_test, y_test, args.epochs, args.lr, args.seed)

    train_auc = auc_score([int(v) for v in y_train.tolist()], train_prob.tolist())
    test_auc = auc_score([int(v) for v in y_test.tolist()], test_prob.tolist()) if len(test_rows) else 0.5
    base = float(y_train.mean().item()) if len(y_train) else 0.0
    test_base_loss = F.binary_cross_entropy(
        torch.full_like(y_test, max(1e-4, min(0.9999, base))), y_test).item() if len(y_test) else 0.0

    print(
        f"games={len(games)} train_decisions={len(train_rows)} test_decisions={len(test_rows)} "
        f"features={len(feature_names)} actor={args.actor} history={args.history_window} "
        f"feature_mode={args.feature_mode}"
    )
    print(
        f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
        f"test_base_loss={test_base_loss:.4f} train_auc={train_auc:.4f} test_auc={test_auc:.4f}"
    )

    weighted = sorted(
        ((feature_names[i], float(w[i].item())) for i in range(len(feature_names))),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    print("top_weights:")
    for name, value in weighted[:20]:
        print(f"{name},{value:+.4f}")

    if args.weights_csv:
        out = Path(args.weights_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["feature", "weight"])
            writer.writerows(weighted)

    if args.predictions_csv:
        out = Path(args.predictions_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        all_probs_by_row = {}
        for row, prob in zip(train_rows, train_prob.tolist()):
            all_probs_by_row[(row["game_idx"], row["decision_idx"])] = (prob, "train")
        for row, prob in zip(test_rows, test_prob.tolist()):
            all_probs_by_row[(row["game_idx"], row["decision_idx"])] = (prob, "test")
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "split", "game_idx", "decision_idx", "label", "pred_win_prob",
                "synthetic_return", "turn", "phase", "kind", "actor", "selected", "path"
            ])
            for row in rows:
                prob, split = all_probs_by_row[(row["game_idx"], row["decision_idx"])]
                writer.writerow([
                    split,
                    row["game_idx"],
                    row["decision_idx"],
                    row["label"],
                    f"{prob:.6f}",
                    f"{(2.0 * prob - 1.0):.6f}",
                    row["turn"],
                    row["phase"],
                    row["kind"],
                    row["actor"],
                    row["selected"],
                    row["path"],
                ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
