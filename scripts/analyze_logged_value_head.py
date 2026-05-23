#!/usr/bin/env python3
"""Analyze logged MTGRL value-head scores against terminal outcomes.

This diagnostic consumes JSONL emitted by export_game_log_trajectories.py. It
does not train a probe; it measures whether the model's own logged VALUE SCORE
has usable terminal-outcome signal.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
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
    return (rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)


def pearson(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    if len(labels) < 2:
        return None
    mean_y = sum(labels) / float(len(labels))
    mean_s = sum(scores) / float(len(scores))
    cov = sum((y - mean_y) * (s - mean_s) for y, s in zip(labels, scores))
    var_y = sum((y - mean_y) ** 2 for y in labels)
    var_s = sum((s - mean_s) ** 2 for s in scores)
    if var_y <= 0.0 or var_s <= 0.0:
        return None
    return cov / math.sqrt(var_y * var_s)


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = max(0.0, min(1.0, q)) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / float(len(values)) if values else None


def affine_prob(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def brier_affine(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    if not labels:
        return None
    return mean([(affine_prob(s) - y) ** 2 for y, s in zip(labels, scores)])


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
    return 1.0 if mp > 0.0 and sp >= mp - 1e-6 else 0.0


def matchup_from_path(path: str) -> str:
    match = re.search(r"__vs__(.+?)__chunk", path)
    if not match:
        return "unknown"
    text = match.group(1)
    text = text.replace("Deck_-_", "Deck - ")
    text = text.replace("_", " ")
    return text.strip()


def top_options(decision: Dict[str, object], limit: int = 6) -> str:
    opts = list(decision.get("options") or [])
    opts.sort(key=lambda o: safe_float(o.get("prob"), 0.0), reverse=True)
    parts = []
    for opt in opts[:limit]:
        marker = "*" if opt.get("selected_marker") else ""
        parts.append(f"{marker}{safe_float(opt.get('prob'), 0.0):.3f}:{str(opt.get('text') or '')}")
    return " | ".join(parts)


def iter_games(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def build_decision_rows(games: Sequence[Dict[str, object]], actor: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for game_idx, game in enumerate(games):
        label = 1 if game.get("rl_won") else 0
        path = str(game.get("path") or "")
        matchup = matchup_from_path(path)
        for decision_idx, decision in enumerate(game.get("decisions") or []):
            if actor != "all" and str(decision.get("actor") or "") != actor:
                continue
            value = decision.get("value_score")
            if value is None:
                continue
            options = decision.get("options") or []
            flags = decision.get("flags") or {}
            selected = str(decision.get("selected") or "")
            phase = str(decision.get("phase") or "")
            row = {
                "game_idx": game_idx,
                "decision_idx": decision_idx,
                "label": label,
                "value_score": safe_float(value, 0.0),
                "kind": str(decision.get("kind") or ""),
                "turn": int(safe_float(decision.get("turn"), 0.0)),
                "phase": phase,
                "actor": str(decision.get("actor") or ""),
                "selected": selected,
                "path": path,
                "matchup": matchup,
                "options_count": len(options),
                "selected_prob": selected_prob(decision),
                "max_prob": max_prob(decision),
                "selected_is_top": selected_is_top(decision),
                "selected_pass": bool(flags.get("selected_pass")),
                "selected_play_land": bool(flags.get("selected_play_land")),
                "selected_cast_spy": bool(flags.get("selected_cast_spy")),
                "selected_cast_dread_return": bool(flags.get("selected_cast_dread_return")),
                "has_land_play_option": bool(flags.get("has_land_play_option")),
                "has_cast_spy_option": bool(flags.get("has_cast_spy_option")),
                "has_dread_return_option": bool(flags.get("has_dread_return_option")),
                "has_nonpass_option": bool(flags.get("has_nonpass_option")),
                "selected_lotleth": selected.startswith("Cast Lotleth Giant"),
                "has_lotleth_option": any(str(o.get("text") or "").startswith("Cast Lotleth Giant") for o in options),
                "top_options": top_options(decision),
            }
            phase_upper = phase.upper()
            row["target_pick"] = "TARGET_PICK" in phase_upper
            row["card_select"] = "CARD_SELECT" in phase_upper or "CARD_PICK" in phase_upper
            row["combat_pick"] = "ATTACK_PICK" in phase_upper or "BLOCK_PICK" in phase_upper
            row["action_head_like"] = not row["target_pick"] and not row["card_select"] and not row["combat_pick"]
            row["value_gt_1"] = row["value_score"] > 1.0
            row["critical_combo"] = (
                row["selected_cast_spy"]
                or row["selected_cast_dread_return"]
                or row["selected_lotleth"]
                or row["has_cast_spy_option"]
                or row["has_dread_return_option"]
                or row["has_lotleth_option"]
            )
            row["pass_with_nonpass"] = row["selected_pass"] and row["has_nonpass_option"]
            row["critical_action_option"] = row["has_cast_spy_option"] or row["has_dread_return_option"]
            rows.append(row)
    return rows


def fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def metric_row(name: str, rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    labels = [int(r["label"]) for r in rows]
    scores = [float(r["value_score"]) for r in rows]
    wins = sum(labels)
    losses = len(labels) - wins
    win_scores = [s for y, s in zip(labels, scores) if y == 1]
    loss_scores = [s for y, s in zip(labels, scores) if y == 0]
    selected_top = [float(r["selected_is_top"]) for r in rows]
    selected_probs = [float(r["selected_prob"]) for r in rows]
    max_probs = [float(r["max_prob"]) for r in rows]
    acc_zero = None
    if rows:
        acc_zero = sum(1 for y, s in zip(labels, scores) if (s >= 0.0) == (y == 1)) / float(len(rows))
    return {
        "subset": name,
        "n": len(rows),
        "wins": wins,
        "losses": losses,
        "base_winrate": fmt(wins / float(len(rows)) if rows else None),
        "auc": fmt(auc_score(labels, scores)),
        "pearson": fmt(pearson(labels, scores)),
        "acc_value_ge_0": fmt(acc_zero),
        "brier_affine": fmt(brier_affine(labels, scores)),
        "mean_score": fmt(mean(scores)),
        "mean_win_score": fmt(mean(win_scores)),
        "mean_loss_score": fmt(mean(loss_scores)),
        "score_gap_win_minus_loss": fmt((mean(win_scores) - mean(loss_scores)) if win_scores and loss_scores else None),
        "min_score": fmt(min(scores) if scores else None),
        "p10_score": fmt(quantile(scores, 0.10)),
        "p50_score": fmt(quantile(scores, 0.50)),
        "p90_score": fmt(quantile(scores, 0.90)),
        "max_score": fmt(max(scores) if scores else None),
        "mean_selected_prob": fmt(mean(selected_probs)),
        "mean_max_prob": fmt(mean(max_probs)),
        "selected_top_rate": fmt(mean(selected_top)),
    }


def grouped_metrics(rows: Sequence[Dict[str, object]], key: str) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "unknown")].append(row)
    return [metric_row(name, items) for name, items in sorted(groups.items())]


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def game_aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_game: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_game[int(row["game_idx"])].append(row)
    out = []
    for game_idx, items in sorted(by_game.items()):
        items = sorted(items, key=lambda r: int(r["decision_idx"]))
        values = [float(r["value_score"]) for r in items]
        non_mull = [r for r in items if r["kind"] != "mulligan"]
        critical = [r for r in items if r["critical_combo"]]
        row = {
            "game_idx": game_idx,
            "label": int(items[0]["label"]),
            "matchup": str(items[0]["matchup"]),
            "path": str(items[0]["path"]),
            "decisions": len(items),
            "mean_value_score": mean(values),
            "first_value_score": values[0],
            "last_value_score": values[-1],
            "max_value_score": max(values),
            "min_value_score": min(values),
            "first_non_mulligan_value_score": float(non_mull[0]["value_score"]) if non_mull else values[0],
            "mean_critical_value_score": mean([float(r["value_score"]) for r in critical]),
            "critical_decisions": len(critical),
        }
        out.append(row)
    return out


def aggregate_metric_row(name: str, rows: Sequence[Dict[str, object]], score_key: str) -> Dict[str, object]:
    valid = [r for r in rows if r.get(score_key) is not None]
    labels = [int(r["label"]) for r in valid]
    scores = [float(r[score_key]) for r in valid]
    wins = sum(labels)
    win_scores = [s for y, s in zip(labels, scores) if y == 1]
    loss_scores = [s for y, s in zip(labels, scores) if y == 0]
    return {
        "score_key": score_key,
        "subset": name,
        "n": len(valid),
        "wins": wins,
        "losses": len(valid) - wins,
        "base_winrate": fmt(wins / float(len(valid)) if valid else None),
        "auc": fmt(auc_score(labels, scores)),
        "pearson": fmt(pearson(labels, scores)),
        "mean_score": fmt(mean(scores)),
        "mean_win_score": fmt(mean(win_scores)),
        "mean_loss_score": fmt(mean(loss_scores)),
        "score_gap_win_minus_loss": fmt((mean(win_scores) - mean(loss_scores)) if win_scores and loss_scores else None),
        "p10_score": fmt(quantile(scores, 0.10)),
        "p50_score": fmt(quantile(scores, 0.50)),
        "p90_score": fmt(quantile(scores, 0.90)),
    }


def write_decision_samples(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    high_loss = sorted((r for r in rows if int(r["label"]) == 0), key=lambda r: float(r["value_score"]), reverse=True)[:30]
    low_win = sorted((r for r in rows if int(r["label"]) == 1), key=lambda r: float(r["value_score"]))[:30]
    sample_rows = []
    for bucket, items in (("high_score_loss", high_loss), ("low_score_win", low_win)):
        for row in items:
            item = dict(row)
            item["bucket"] = bucket
            sample_rows.append(item)
    write_csv(path, sample_rows)


def write_readme(path: Path, summary: Sequence[Dict[str, object]], aggregate: Sequence[Dict[str, object]]) -> None:
    def pick(rows: Sequence[Dict[str, object]], key: str, value: str) -> Optional[Dict[str, object]]:
        for row in rows:
            if row.get(key) == value:
                return row
        return None

    all_row = pick(summary, "subset", "all")
    nontrivial_row = pick(summary, "subset", "nontrivial_options")
    critical_row = pick(summary, "subset", "critical_combo")
    mean_game = None
    for row in aggregate:
        if row.get("score_key") == "mean_value_score":
            mean_game = row
            break

    lines = [
        "# Logged Value-Head Diagnostic",
        "",
        "This analyzes the model's logged `VALUE SCORE` against the terminal game result. No probe is trained here.",
        "",
        "## Headline",
        "",
    ]
    if all_row:
        lines.append(
            f"- Decision rows: n={all_row['n']}, win base rate={all_row['base_winrate']}, "
            f"AUC={all_row['auc']}, mean win score={all_row['mean_win_score']}, "
            f"mean loss score={all_row['mean_loss_score']}."
        )
    if nontrivial_row:
        lines.append(
            f"- Nontrivial decision rows: n={nontrivial_row['n']}, AUC={nontrivial_row['auc']}, "
            f"score gap={nontrivial_row['score_gap_win_minus_loss']}."
        )
    if critical_row:
        lines.append(
            f"- Critical combo rows: n={critical_row['n']}, AUC={critical_row['auc']}, "
            f"score gap={critical_row['score_gap_win_minus_loss']}."
        )
    if mean_game:
        lines.append(
            f"- Per-game mean value score: n={mean_game['n']}, AUC={mean_game['auc']}, "
            f"score gap={mean_game['score_gap_win_minus_loss']}."
        )

    lines.extend([
        "",
        "## Files",
        "",
        "- `summary.csv`: decision-level subsets.",
        "- `by_matchup.csv`: decision-level matchup splits.",
        "- `game_aggregate_metrics.csv`: one-row-per-game value aggregations.",
        "- `game_values.csv`: raw per-game aggregate values.",
        "- `decision_samples.csv`: high-score losses and low-score wins for log inspection.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Trajectory JSONL path. Can be repeated.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--actor", default="PlayerRL1", help="Actor to analyze, or 'all'.")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    games = list(iter_games(inputs))
    rows = build_decision_rows(games, actor=args.actor)

    subsets: List[Tuple[str, Callable[[Dict[str, object]], bool]]] = [
        ("all", lambda r: True),
        ("nontrivial_options", lambda r: int(r["options_count"]) > 1),
        ("mulligan", lambda r: r["kind"] == "mulligan"),
        ("non_mulligan", lambda r: r["kind"] != "mulligan"),
        ("action_head_like", lambda r: bool(r["action_head_like"])),
        ("target_pick", lambda r: bool(r["target_pick"])),
        ("value_gt_1", lambda r: bool(r["value_gt_1"])),
        ("turn_le_3", lambda r: int(r["turn"]) <= 3),
        ("turn_ge_4", lambda r: int(r["turn"]) >= 4),
        ("has_land_play_option", lambda r: bool(r["has_land_play_option"])),
        ("pass_with_nonpass", lambda r: bool(r["pass_with_nonpass"])),
        ("selected_play_land", lambda r: bool(r["selected_play_land"])),
        ("has_cast_spy_option", lambda r: bool(r["has_cast_spy_option"])),
        ("selected_cast_spy", lambda r: bool(r["selected_cast_spy"])),
        ("has_dread_return_option", lambda r: bool(r["has_dread_return_option"])),
        ("selected_cast_dread_return", lambda r: bool(r["selected_cast_dread_return"])),
        ("critical_action_option", lambda r: bool(r["critical_action_option"])),
        ("critical_combo", lambda r: bool(r["critical_combo"])),
    ]

    summary_rows = [metric_row(name, [r for r in rows if pred(r)]) for name, pred in subsets]
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "by_matchup.csv", grouped_metrics(rows, "matchup"))

    game_rows = game_aggregate_rows(rows)
    write_csv(out_dir / "game_values.csv", game_rows)
    aggregate_rows = []
    for key in (
        "mean_value_score",
        "first_value_score",
        "first_non_mulligan_value_score",
        "last_value_score",
        "max_value_score",
        "min_value_score",
        "mean_critical_value_score",
    ):
        aggregate_rows.append(aggregate_metric_row("all_games", game_rows, key))
    write_csv(out_dir / "game_aggregate_metrics.csv", aggregate_rows)
    write_decision_samples(out_dir / "decision_samples.csv", rows)
    write_readme(out_dir / "README.md", summary_rows, aggregate_rows)

    print(f"games={len(games)} decision_rows={len(rows)} output={out_dir}")
    for row in summary_rows[:4]:
        print(
            f"{row['subset']}: n={row['n']} base={row['base_winrate']} "
            f"auc={row['auc']} gap={row['score_gap_win_minus_loss']}"
        )
    for row in aggregate_rows[:4]:
        print(
            f"game {row['score_key']}: n={row['n']} auc={row['auc']} "
            f"gap={row['score_gap_win_minus_loss']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
