#!/usr/bin/env python3
"""Export per-profile MTGRL training CSV metrics for Prometheus.

This intentionally runs out-of-process so dashboarding cannot perturb the
trainer. It reads the append-only CSV files under each profile's logs/stats
directory on every scrape and emits gauges that Grafana can group by profile,
deck, opponent deck, and opponent type.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_ROOT = (
    REPO_ROOT
    / "Mage.Server.Plugins"
    / "Mage.Player.AIRL"
    / "src"
    / "mage"
    / "player"
    / "ai"
    / "rl"
    / "profiles"
)


def _float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def _won(row: Dict[str, str]) -> float:
    return 1.0 if _float(row.get("final_reward"), 0.0) > 0.0 else 0.0


def _label_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace('"', '\\"')
    )


def _labels(**labels: object) -> str:
    if not labels:
        return ""
    parts = [f'{key}="{_label_escape(value)}"' for key, value in labels.items()]
    return "{" + ",".join(parts) + "}"


def _metric(name: str, value: object, **labels: object) -> str:
    return f"{name}{_labels(**labels)} {_float(value):.10g}"


def _read_csv(path: Path, max_lines: int) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            rows: List[Dict[str, str]] = []
            for row in reader:
                rows.append(row)
                if max_lines > 0 and len(rows) > max_lines:
                    rows.pop(0)
            return rows
    except Exception:
        return []


def _mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _avg(values: Iterable[float]) -> Optional[float]:
    data = list(values)
    if not data:
        return None
    return sum(data) / float(len(data))


class ProfileMetricsExporter:
    def __init__(self, profiles_root: Path, profiles: Optional[List[str]], window: int, max_lines: int) -> None:
        self.profiles_root = profiles_root
        self.profiles = profiles
        self.window = max(1, int(window))
        self.max_lines = int(max_lines)

    def _discover_profiles(self) -> List[str]:
        if self.profiles:
            return sorted(self.profiles)
        if not self.profiles_root.exists():
            return []
        found = []
        for child in self.profiles_root.iterdir():
            if not child.is_dir():
                continue
            stats_dir = child / "logs" / "stats"
            if (stats_dir / "training_stats.csv").exists() or (stats_dir / "training_cells.csv").exists():
                found.append(child.name)
        return sorted(found)

    def render(self) -> str:
        now = time.time()
        lines = [
            "# HELP mage_profile_exporter_up Profile CSV exporter health.",
            "# TYPE mage_profile_exporter_up gauge",
            "mage_profile_exporter_up 1",
            "# HELP mage_profile_exporter_scrape_time_seconds Unix timestamp of exporter scrape.",
            "# TYPE mage_profile_exporter_scrape_time_seconds gauge",
            f"mage_profile_exporter_scrape_time_seconds {now:.3f}",
            "# HELP mage_profile_training_episodes Training rows recorded per profile.",
            "# TYPE mage_profile_training_episodes gauge",
            "# HELP mage_profile_training_episode_index Max episode index observed per profile.",
            "# TYPE mage_profile_training_episode_index gauge",
            "# HELP mage_profile_winrate_rolling Rolling winrate from final_reward over the last N profile rows.",
            "# TYPE mage_profile_winrate_rolling gauge",
            "# HELP mage_profile_winrate_csv_latest Latest winrate column value in training_stats.csv.",
            "# TYPE mage_profile_winrate_csv_latest gauge",
            "# HELP mage_profile_episode_seconds_rolling_avg Rolling average episode duration in seconds.",
            "# TYPE mage_profile_episode_seconds_rolling_avg gauge",
            "# HELP mage_profile_turns_rolling_avg Rolling average turns per game.",
            "# TYPE mage_profile_turns_rolling_avg gauge",
            "# HELP mage_profile_training_file_mtime_seconds Last modification time for profile stats files.",
            "# TYPE mage_profile_training_file_mtime_seconds gauge",
            "# HELP mage_profile_training_deck_games Total profile rows by RL deck.",
            "# TYPE mage_profile_training_deck_games gauge",
            "# HELP mage_profile_training_deck_winrate_rolling Rolling winrate by RL deck.",
            "# TYPE mage_profile_training_deck_winrate_rolling gauge",
            "# HELP mage_profile_training_matchup_games Total profile rows by RL deck, opponent deck, and opponent type.",
            "# TYPE mage_profile_training_matchup_games gauge",
            "# HELP mage_profile_training_matchup_winrate_rolling Rolling winrate by matchup.",
            "# TYPE mage_profile_training_matchup_winrate_rolling gauge",
            "# HELP mage_profile_value_accuracy Latest value prediction sign accuracy by profile.",
            "# TYPE mage_profile_value_accuracy gauge",
            "# HELP mage_profile_value_avg_wins Latest average value prediction for wins by profile.",
            "# TYPE mage_profile_value_avg_wins gauge",
            "# HELP mage_profile_value_avg_losses Latest average value prediction for losses by profile.",
            "# TYPE mage_profile_value_avg_losses gauge",
            "# HELP mage_profile_training_loss Latest training loss components by profile.",
            "# TYPE mage_profile_training_loss gauge",
            "# HELP mage_profile_head_usage_rolling_avg Rolling average decisions per game by neural head.",
            "# TYPE mage_profile_head_usage_rolling_avg gauge",
            "# HELP mage_profile_head_usage_decision_share Rolling decision share by neural head.",
            "# TYPE mage_profile_head_usage_decision_share gauge",
            "# HELP mage_profile_head_usage_by_deck_rolling_avg Rolling average RL-side decisions per game by RL deck and neural head.",
            "# TYPE mage_profile_head_usage_by_deck_rolling_avg gauge",
            "# HELP mage_profile_health_latest Latest training health counters from training_health.csv.",
            "# TYPE mage_profile_health_latest gauge",
            "# HELP mage_profile_eval_winrate Latest checkpoint eval winrate by profile, if eval_history.csv exists.",
            "# TYPE mage_profile_eval_winrate gauge",
        ]

        for profile in self._discover_profiles():
            self._append_profile(lines, profile)
        return "\n".join(lines) + "\n"

    def _append_profile(self, lines: List[str], profile: str) -> None:
        stats_dir = self.profiles_root / profile / "logs" / "stats"
        stats_path = stats_dir / "training_stats.csv"
        cells_path = stats_dir / "training_cells.csv"
        value_path = stats_dir / "value_accuracy.csv"
        losses_path = stats_dir / "training_losses.csv"
        head_path = stats_dir / "head_usage.csv"
        health_path = self.profiles_root / profile / "logs" / "health" / "training_health.csv"
        eval_path = stats_dir / "eval_history.csv"

        for file_name, path in (
            ("training_stats", stats_path),
            ("training_cells", cells_path),
            ("value_accuracy", value_path),
            ("training_losses", losses_path),
            ("head_usage", head_path),
            ("training_health", health_path),
            ("eval_history", eval_path),
        ):
            lines.append(_metric("mage_profile_training_file_mtime_seconds", _mtime(path), profile=profile, file=file_name))

        stats_rows = _read_csv(stats_path, self.max_lines)
        if stats_rows:
            latest = stats_rows[-1]
            recent = stats_rows[-self.window :]
            lines.append(_metric("mage_profile_training_episodes", len(stats_rows), profile=profile))
            lines.append(_metric("mage_profile_training_episode_index", max(_int(row.get("episode")) for row in stats_rows), profile=profile))
            lines.append(_metric("mage_profile_winrate_rolling", _avg(_won(row) for row in recent) or 0.0, profile=profile, window=self.window))
            lines.append(_metric("mage_profile_winrate_csv_latest", latest.get("winrate", 0.0), profile=profile))
            ep_seconds = [_float(row.get("episode_seconds")) for row in recent if str(row.get("episode_seconds", "")).strip()]
            turns = [_float(row.get("turns")) for row in recent if str(row.get("turns", "")).strip()]
            if ep_seconds:
                lines.append(_metric("mage_profile_episode_seconds_rolling_avg", _avg(ep_seconds) or 0.0, profile=profile, window=self.window))
            if turns:
                lines.append(_metric("mage_profile_turns_rolling_avg", _avg(turns) or 0.0, profile=profile, window=self.window))
        else:
            lines.append(_metric("mage_profile_training_episodes", 0, profile=profile))

        self._append_cell_metrics(lines, profile, cells_path)
        self._append_value_metrics(lines, profile, value_path)
        self._append_loss_metrics(lines, profile, losses_path)
        self._append_head_usage_metrics(lines, profile, head_path)
        self._append_health_metrics(lines, profile, health_path)
        self._append_eval_metrics(lines, profile, eval_path)

    def _append_cell_metrics(self, lines: List[str], profile: str, path: Path) -> None:
        rows = _read_csv(path, self.max_lines)
        if not rows:
            return
        deck_counts: Dict[str, int] = defaultdict(int)
        deck_recent: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))
        matchup_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        matchup_recent: Dict[Tuple[str, str, str], Deque[float]] = defaultdict(lambda: deque(maxlen=self.window))

        for row in rows:
            rl_deck = row.get("rl_deck", "unknown") or "unknown"
            opp_deck = row.get("opp_deck", "unknown") or "unknown"
            opponent_type = row.get("opponent_type", "unknown") or "unknown"
            won = _won(row)
            deck_counts[rl_deck] += 1
            deck_recent[rl_deck].append(won)
            key = (rl_deck, opp_deck, opponent_type)
            matchup_counts[key] += 1
            matchup_recent[key].append(won)

        for rl_deck, count in sorted(deck_counts.items()):
            lines.append(_metric("mage_profile_training_deck_games", count, profile=profile, rl_deck=rl_deck))
            lines.append(_metric(
                "mage_profile_training_deck_winrate_rolling",
                _avg(deck_recent[rl_deck]) or 0.0,
                profile=profile,
                rl_deck=rl_deck,
                window=self.window,
            ))

        for (rl_deck, opp_deck, opponent_type), count in sorted(matchup_counts.items()):
            lines.append(_metric(
                "mage_profile_training_matchup_games",
                count,
                profile=profile,
                rl_deck=rl_deck,
                opp_deck=opp_deck,
                opponent_type=opponent_type,
            ))
            lines.append(_metric(
                "mage_profile_training_matchup_winrate_rolling",
                _avg(matchup_recent[(rl_deck, opp_deck, opponent_type)]) or 0.0,
                profile=profile,
                rl_deck=rl_deck,
                opp_deck=opp_deck,
                opponent_type=opponent_type,
                window=self.window,
            ))

    def _append_value_metrics(self, lines: List[str], profile: str, path: Path) -> None:
        rows = _read_csv(path, self.max_lines)
        if not rows:
            return
        row = rows[-1]
        lines.append(_metric("mage_profile_value_accuracy", row.get("value_accuracy", 0.0), profile=profile))
        lines.append(_metric("mage_profile_value_avg_wins", row.get("avg_win_value", 0.0), profile=profile))
        lines.append(_metric("mage_profile_value_avg_losses", row.get("avg_loss_value", 0.0), profile=profile))

    def _append_loss_metrics(self, lines: List[str], profile: str, path: Path) -> None:
        rows = _read_csv(path, self.max_lines)
        if not rows:
            return
        row = rows[-1]
        for column, component in (
            ("total_loss", "total"),
            ("policy_loss", "policy"),
            ("value_loss", "value"),
            ("entropy", "entropy"),
            ("entropy_coef", "entropy_coef"),
            ("clip_frac", "clip_frac"),
            ("approx_kl", "approx_kl"),
            ("advantage_mean", "advantage_mean"),
        ):
            if column in row:
                lines.append(_metric("mage_profile_training_loss", row.get(column, 0.0), profile=profile, component=component))

    def _append_head_usage_metrics(self, lines: List[str], profile: str, path: Path) -> None:
        rows = _read_csv(path, self.max_lines)
        if not rows:
            return
        recent = rows[-self.window :]
        heads = (
            ("rl", "action", "rl_action_head"),
            ("rl", "target", "rl_target_head"),
            ("rl", "card_select", "rl_card_select_head"),
            ("opp", "action", "opp_action_head"),
            ("opp", "target", "opp_target_head"),
            ("opp", "card_select", "opp_card_select_head"),
        )
        for side, head, column in heads:
            avg = _avg(_float(row.get(column), 0.0) for row in recent) or 0.0
            total_column = "rl_total" if side == "rl" else "opp_total"
            total_avg = _avg(_float(row.get(total_column), 0.0) for row in recent) or 0.0
            share = avg / total_avg if total_avg > 0.0 else 0.0
            lines.append(_metric(
                "mage_profile_head_usage_rolling_avg",
                avg,
                profile=profile,
                side=side,
                head=head,
                window=self.window,
            ))
            lines.append(_metric(
                "mage_profile_head_usage_decision_share",
                share,
                profile=profile,
                side=side,
                head=head,
                window=self.window,
            ))

        by_deck: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for row in recent:
            rl_deck = row.get("rl_deck", "unknown") or "unknown"
            by_deck[(rl_deck, "action")].append(_float(row.get("rl_action_head"), 0.0))
            by_deck[(rl_deck, "target")].append(_float(row.get("rl_target_head"), 0.0))
            by_deck[(rl_deck, "card_select")].append(_float(row.get("rl_card_select_head"), 0.0))
        for (rl_deck, head), values in sorted(by_deck.items()):
            lines.append(_metric(
                "mage_profile_head_usage_by_deck_rolling_avg",
                _avg(values) or 0.0,
                profile=profile,
                rl_deck=rl_deck,
                head=head,
                window=self.window,
            ))

    def _append_health_metrics(self, lines: List[str], profile: str, path: Path) -> None:
        rows = _read_csv(path, self.max_lines)
        if not rows:
            return
        row = rows[-1]
        for column in (
            "uptime_min",
            "games_killed",
            "rl_activation_failures",
            "gpu_ooms",
            "python_errors",
            "model_nans",
        ):
            if column in row:
                lines.append(_metric(
                    "mage_profile_health_latest",
                    row.get(column, 0.0),
                    profile=profile,
                    component=column,
                ))

    def _append_eval_metrics(self, lines: List[str], profile: str, path: Path) -> None:
        rows = _read_csv(path, self.max_lines)
        if not rows:
            return
        row = rows[-1]
        if "winrate" in row:
            lines.append(_metric("mage_profile_eval_winrate", row.get("winrate", 0.0), profile=profile))


class Handler(BaseHTTPRequestHandler):
    exporter: ProfileMetricsExporter

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return
        body = self.exporter.render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default=os.getenv("PROFILE_METRICS_BIND", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PROFILE_METRICS_PORT", "9102")))
    parser.add_argument("--profiles-root", default=os.getenv("PROFILE_METRICS_PROFILES_ROOT", str(DEFAULT_PROFILES_ROOT)))
    parser.add_argument("--profiles", default=os.getenv("PROFILE_METRICS_PROFILES", os.getenv("TRAIN_PROFILES_LIST", "")))
    parser.add_argument("--window", type=int, default=int(os.getenv("PROFILE_METRICS_WINDOW", "200")))
    parser.add_argument("--max-lines", type=int, default=int(os.getenv("PROFILE_METRICS_MAX_LINES", "200000")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = [item.strip() for item in str(args.profiles).split(",") if item.strip()] or None
    Handler.exporter = ProfileMetricsExporter(
        profiles_root=Path(args.profiles_root),
        profiles=profiles,
        window=args.window,
        max_lines=args.max_lines,
    )
    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"profile metrics exporter listening on {args.bind}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
