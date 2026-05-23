#!/usr/bin/env python3
"""Summarize basic action-health signals from MTGRL game logs.

This intentionally stays generic: it only looks for legal action text already
written by GameLogger, such as a Pass selected while a Play <land> option exists.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DECISION_RE = re.compile(r"^DECISION #\d+ - Turn (\d+) \((.*?) turn\), (.*?) - (.*)$")
OPTION_RE = re.compile(r"^\s*(?:>>>\s*)?\[\d+\]\s+[-0-9.]+\s+-\s+(.+?)\s*$")
OPTION_SCORE_RE = re.compile(r"^\s*(>>>\s*)?\[\d+\]\s+([-0-9.]+)\s+-\s+(.+?)\s*$")
SELECTED_RE = re.compile(r"^SELECTED:\s+(.+?)\s*$")
PLAYER_RE = re.compile(r"^\[(PlayerRL\d+)\], life = (-?\d+)$")
ZONE_RE = re.compile(r"^-> (Hand|Permanents|Graveyard|Exile): \[(.*)\]$")
RESULT_RE = re.compile(r"^RESULT:\s+(WIN|LOSS|DRAW)\s*$")
MULLIGAN_RE = re.compile(
    r"^MULLIGAN_DECISION: .*?mulligansTaken=(\d+) handSize=(\d+) lands=(\d+) "
    r"decision=(KEEP|MULLIGAN)\b.*?(?:hand=\[(.*)\])?$"
)


def iter_logs(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.txt"))


def classify_land_play(text: str) -> bool:
    return text.startswith("Play ") and not text.startswith("Play with")


def classify_cleansing_target(text: str) -> str:
    if "(you)" in text:
        if "Bridge" in text:
            return "self_bridge"
        return "self_other"
    if "(EvalBot" in text or "(Opponent" in text or "(opponent" in text:
        return "opponent"
    if "(" in text:
        return "opponent"
    return "unknown"


def split_cards(text: str) -> List[str]:
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def contains_card(cards: List[str], name: str) -> bool:
    return any(card == name or card.startswith(f"{name},") for card in cards)


def count_creatures(cards: List[str]) -> int:
    # This is Spy-deck specific and intentionally narrow. It is used only for
    # combo-health diagnostics, not for training labels or rewards.
    creature_names = {
        "Balustrade Spy",
        "Elves of Deep Shadow",
        "Gatecreeper Vine",
        "Generous Ent",
        "Lotleth Giant",
        "Masked Vandal",
        "Mesmeric Fiend",
        "Overgrown Battlement",
        "Quirion Ranger",
        "Sagu Wildling",
        "Saruli Caretaker",
        "Tinder Wall",
        "Troll of Khazad-dum",
        "Wall of Roots",
    }
    return sum(1 for card in cards if card.split(",", 1)[0] in creature_names)


def count_true_lands(cards: List[str]) -> int:
    return sum(1 for card in cards if card.split(",", 1)[0] in {"Forest", "Swamp"})


def visible_true_lands(player_zones: Dict[str, List[str]]) -> int:
    return sum(
        count_true_lands(player_zones.get(zone, []))
        for zone in ("hand", "permanents", "graveyard", "exile")
    )


def scan_log(path: Path) -> Counter:
    counts: Counter = Counter()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    game: Counter = Counter()
    player_zones: Dict[str, List[str]] = {}
    current_player: Optional[str] = None

    def mark_game_flag(name: str) -> None:
        game[name] = 1

    def observe_state_line(line: str) -> bool:
        nonlocal current_player
        result = RESULT_RE.match(line)
        if result:
            mark_game_flag(f"result_{result.group(1).lower()}")
            return True

        player = PLAYER_RE.match(line)
        if player:
            current_player = player.group(1)
            return True

        zone = ZONE_RE.match(line)
        if zone and current_player == "PlayerRL1":
            zone_name = zone.group(1).lower()
            cards = split_cards(zone.group(2))
            player_zones[zone_name] = cards
            if zone_name == "hand":
                if contains_card(cards, "Balustrade Spy"):
                    mark_game_flag("spy_seen_in_hand_games")
                if contains_card(cards, "Land Grant"):
                    mark_game_flag("land_grant_seen_in_hand_games")
            elif zone_name == "permanents":
                if contains_card(cards, "Balustrade Spy"):
                    mark_game_flag("spy_seen_on_battlefield_games")
                if contains_card(cards, "Lotleth Giant"):
                    mark_game_flag("lotleth_seen_on_battlefield_games")
            elif zone_name == "graveyard":
                if contains_card(cards, "Balustrade Spy"):
                    mark_game_flag("spy_seen_in_graveyard_games")
                if contains_card(cards, "Dread Return"):
                    mark_game_flag("dread_return_seen_in_graveyard_games")
                if contains_card(cards, "Lotleth Giant"):
                    mark_game_flag("lotleth_seen_in_graveyard_games")
                if contains_card(cards, "Dread Return") and contains_card(cards, "Lotleth Giant"):
                    mark_game_flag("combo_payload_in_graveyard_games")
                if count_creatures(cards) >= 3:
                    mark_game_flag("three_creatures_in_graveyard_games")
                if contains_card(cards, "Dread Return") and contains_card(cards, "Lotleth Giant") and count_creatures(cards) >= 3:
                    mark_game_flag("combo_ready_graveyard_games")
            elif zone_name == "exile":
                if contains_card(cards, "Dread Return"):
                    mark_game_flag("dread_return_seen_in_exile_games")
                if contains_card(cards, "Lotleth Giant"):
                    mark_game_flag("lotleth_seen_in_exile_games")
            return True
        return False

    i = 0
    while i < len(lines):
        if observe_state_line(lines[i]):
            i += 1
            continue

        mull = MULLIGAN_RE.match(lines[i])
        if mull:
            counts["mulligan_decisions"] += 1
            _, hand_size, lands, decision, hand = mull.groups()
            lands_i = int(lands)
            counts[f"mulligan_{decision.lower()}"] += 1
            if decision == "KEEP" and lands_i <= 1:
                counts["keep_0_1_land"] += 1
            if decision == "KEEP" and lands_i >= 5:
                counts["keep_5_plus_land"] += 1
            if decision == "KEEP":
                kept_cards = split_cards(hand or "")
                if contains_card(kept_cards, "Balustrade Spy"):
                    counts["keep_with_balustrade_spy"] += 1
                    if lands_i <= 1:
                        counts["keep_0_1_land_with_balustrade_spy"] += 1
                if contains_card(kept_cards, "Land Grant"):
                    counts["keep_with_land_grant"] += 1
            i += 1
            continue

        decision = DECISION_RE.match(lines[i])
        if not decision:
            i += 1
            continue

        turn = int(decision.group(1))
        phase = decision.group(3)
        actor = decision.group(4)
        block_lines: List[str] = []
        options: List[str] = []
        option_entries: List[tuple[bool, float, str]] = []
        selected: Optional[str] = None
        j = i + 1
        while j < len(lines):
            if j != i + 1 and DECISION_RE.match(lines[j]):
                break
            block_lines.append(lines[j])
            scored = OPTION_SCORE_RE.match(lines[j])
            if scored:
                marker, score_raw, text = scored.groups()
                try:
                    score = float(score_raw)
                except ValueError:
                    score = 0.0
                option_entries.append((bool(marker), score, text))
            opt = OPTION_RE.match(lines[j])
            if opt:
                options.append(opt.group(1))
            sel = SELECTED_RE.match(lines[j])
            if sel:
                selected = sel.group(1)
                break
            j += 1

        land_options = [opt for opt in options if classify_land_play(opt)]
        for block_line in block_lines:
            observe_state_line(block_line)
        if land_options:
            counts["land_play_opportunities"] += 1
            if selected == "Pass":
                counts["pass_over_land"] += 1
                if turn <= 2:
                    counts["early_pass_over_land"] += 1
            elif selected and classify_land_play(selected):
                counts["land_play_selected"] += 1
            if turn == 1 and "Precombat Main" in phase:
                counts["turn1_land_opportunities"] += 1
                if selected == "Pass":
                    counts["turn1_pass_over_land"] += 1
        if selected == "Pass":
            counts["selected_pass"] += 1
        elif selected:
            counts["selected_nonpass"] += 1
        if selected and "TARGET_PICK" in phase and any("Cleansing Wildfire (controller=" in line for line in block_lines):
            kind = classify_cleansing_target(selected)
            counts["cleansing_target_decisions"] += 1
            counts[f"cleansing_target_{kind}"] += 1
        spy_cast_options = [opt for opt in options if opt.startswith("Cast Balustrade Spy")]
        if spy_cast_options:
            counts["spy_cast_opportunities"] += 1
            mark_game_flag("spy_cast_opportunity_games")
            lands_visible = min(4, visible_true_lands(player_zones))
            hidden_lands_est = max(0, 4 - lands_visible)
            counts["spy_cast_hidden_lands_est_sum"] += hidden_lands_est
            counts[f"spy_cast_hidden_lands_est_{hidden_lands_est}"] += 1
            if hidden_lands_est == 0:
                counts["spy_cast_no_hidden_lands_opportunities"] += 1
            else:
                counts["spy_cast_hidden_land_opportunities"] += 1
            spy_scores = [
                score for _, score, opt in option_entries
                if opt.startswith("Cast Balustrade Spy")
            ]
            max_score = max((score for _, score, _ in option_entries), default=0.0)
            best_spy_score = max(spy_scores, default=0.0)
            counts["spy_cast_policy_score_sum"] += best_spy_score
            counts["spy_cast_policy_score_milli_sum"] += int(round(best_spy_score * 1000.0))
            if best_spy_score <= 0.001:
                counts["spy_cast_policy_near_zero"] += 1
            if best_spy_score >= max_score - 1e-6:
                counts["spy_cast_policy_top"] += 1
            if selected and selected.startswith("Cast Balustrade Spy"):
                counts["spy_casts"] += 1
                mark_game_flag("spy_cast_games")
                if hidden_lands_est == 0:
                    counts["spy_casts_no_hidden_lands"] += 1
                    mark_game_flag("spy_cast_no_hidden_lands_games")
                else:
                    counts["spy_casts_with_hidden_lands"] += 1
                    mark_game_flag("spy_cast_with_hidden_lands_games")
                if best_spy_score >= max_score - 1e-6:
                    counts["spy_cast_selected_when_policy_top"] += 1
            else:
                counts["spy_cast_available_not_selected"] += 1
                mark_game_flag("spy_cast_available_not_selected_games")
                if best_spy_score >= max_score - 1e-6:
                    counts["spy_cast_skipped_when_policy_top"] += 1
                if best_spy_score <= 0.001:
                    counts["spy_cast_skipped_when_policy_near_zero"] += 1
        if selected and selected.startswith("Cast Balustrade Spy"):
            counts["spy_casts"] += 1 if not spy_cast_options else 0
            mark_game_flag("spy_cast_games")

        dread_ready = any("Dread Return" in line for line in block_lines)
        if selected == "Flashback sacrifice three creatures" and dread_ready:
            counts["dread_return_flashback_selected"] += 1
            mark_game_flag("dread_return_flashback_games")
            gy = player_zones.get("graveyard", [])
            combo_ready_now = (
                contains_card(gy, "Dread Return")
                and contains_card(gy, "Lotleth Giant")
                and count_creatures(gy) >= 3
            )
            if not contains_card(gy, "Lotleth Giant"):
                counts["premature_dread_flashback_no_lotleth_graveyard"] += 1
                mark_game_flag("premature_dread_flashback_no_lotleth_graveyard_games")
            if not combo_ready_now:
                counts["premature_dread_flashback_not_combo_ready"] += 1
                mark_game_flag("premature_dread_flashback_not_combo_ready_games")
        if "TARGET_PICK" in phase and any("Dread Return (controller=PlayerRL1)" in line for line in block_lines):
            if "TARGET_PICK 0 min=1 max=1" in phase:
                counts["dread_return_target_decisions"] += 1
                mark_game_flag("dread_return_target_decision_games")
                lotleth_available = any(opt == "Lotleth Giant" for opt in options)
                if not lotleth_available:
                    counts["dread_return_target_no_lotleth_available"] += 1
                    mark_game_flag("dread_return_target_no_lotleth_available_games")
                if lotleth_available:
                    counts["dread_return_lotleth_target_opportunities"] += 1
                    mark_game_flag("dread_return_lotleth_target_opportunity_games")
                if selected == "Lotleth Giant":
                    counts["dread_return_target_lotleth"] += 1
                    mark_game_flag("dread_return_target_lotleth_games")
                    if lotleth_available:
                        counts["dread_return_target_lotleth_when_available"] += 1
                elif selected:
                    counts["dread_return_target_other"] += 1
                    if lotleth_available:
                        counts["dread_return_target_other_over_lotleth"] += 1
            elif "min=3 max=3" in phase:
                counts["dread_return_sacrifice_decisions"] += 1
        if selected == "Lotleth Giant" and any("Lotleth Giant (controller=PlayerRL1)" in line for line in block_lines):
            counts["lotleth_trigger_target_selected"] += 1
            mark_game_flag("lotleth_trigger_target_selected_games")
        counts[f"actor:{actor}"] += 1
        i = max(j + 1, i + 1)
    counts["games"] += 1
    counts.update(game)
    return counts


def pct(num: int, den: int) -> str:
    return f"{(num / den * 100.0):.1f}%" if den else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    total: Counter = Counter()
    files = 0
    for root in args.paths:
        for path in iter_logs(root):
            files += 1
            total.update(scan_log(path))

    land = total["land_play_opportunities"]
    t1_land = total["turn1_land_opportunities"]
    mull = total["mulligan_decisions"]
    games = total["games"]
    spy_opp = total["spy_cast_opportunities"]
    dread_targets = total["dread_return_target_decisions"]
    dread_lotleth_opp = total["dread_return_lotleth_target_opportunities"]
    print(f"files,{files}")
    print(f"games,{games}")
    print(f"result_win,{total['result_win']},{pct(total['result_win'], games)}")
    print(f"result_loss,{total['result_loss']},{pct(total['result_loss'], games)}")
    print(f"land_play_opportunities,{land}")
    print(f"pass_over_land,{total['pass_over_land']},{pct(total['pass_over_land'], land)}")
    print(f"early_pass_over_land,{total['early_pass_over_land']},{pct(total['early_pass_over_land'], land)}")
    print(f"turn1_pass_over_land,{total['turn1_pass_over_land']},{pct(total['turn1_pass_over_land'], t1_land)}")
    print(f"land_play_selected,{total['land_play_selected']},{pct(total['land_play_selected'], land)}")
    print(f"mulligan_decisions,{mull}")
    print(f"mulligan_keep,{total['mulligan_keep']},{pct(total['mulligan_keep'], mull)}")
    print(f"mulligan_mulligan,{total['mulligan_mulligan']},{pct(total['mulligan_mulligan'], mull)}")
    print(f"keep_0_1_land,{total['keep_0_1_land']},{pct(total['keep_0_1_land'], total['mulligan_keep'])}")
    print(f"keep_5_plus_land,{total['keep_5_plus_land']},{pct(total['keep_5_plus_land'], total['mulligan_keep'])}")
    print(f"keep_with_balustrade_spy,{total['keep_with_balustrade_spy']},{pct(total['keep_with_balustrade_spy'], total['mulligan_keep'])}")
    print(f"keep_0_1_land_with_balustrade_spy,{total['keep_0_1_land_with_balustrade_spy']},{pct(total['keep_0_1_land_with_balustrade_spy'], total['keep_with_balustrade_spy'])}")
    print(f"keep_with_land_grant,{total['keep_with_land_grant']},{pct(total['keep_with_land_grant'], total['mulligan_keep'])}")
    print(f"spy_seen_in_hand_games,{total['spy_seen_in_hand_games']},{pct(total['spy_seen_in_hand_games'], games)}")
    print(f"land_grant_seen_in_hand_games,{total['land_grant_seen_in_hand_games']},{pct(total['land_grant_seen_in_hand_games'], games)}")
    print(f"spy_cast_opportunities,{spy_opp}")
    print(f"spy_cast_opportunity_games,{total['spy_cast_opportunity_games']},{pct(total['spy_cast_opportunity_games'], games)}")
    print(f"spy_casts,{total['spy_casts']},{pct(total['spy_casts'], spy_opp)}")
    print(f"spy_cast_games,{total['spy_cast_games']},{pct(total['spy_cast_games'], games)}")
    print(f"spy_cast_available_not_selected,{total['spy_cast_available_not_selected']},{pct(total['spy_cast_available_not_selected'], spy_opp)}")
    avg_spy_score = (total["spy_cast_policy_score_sum"] / spy_opp) if spy_opp else 0.0
    print(f"spy_cast_policy_score_avg,{avg_spy_score:.4f}")
    print(f"spy_cast_policy_top,{total['spy_cast_policy_top']},{pct(total['spy_cast_policy_top'], spy_opp)}")
    print(f"spy_cast_policy_near_zero,{total['spy_cast_policy_near_zero']},{pct(total['spy_cast_policy_near_zero'], spy_opp)}")
    print(f"spy_cast_skipped_when_policy_top,{total['spy_cast_skipped_when_policy_top']},{pct(total['spy_cast_skipped_when_policy_top'], spy_opp)}")
    print(f"spy_cast_skipped_when_policy_near_zero,{total['spy_cast_skipped_when_policy_near_zero']},{pct(total['spy_cast_skipped_when_policy_near_zero'], spy_opp)}")
    avg_hidden_lands = (total["spy_cast_hidden_lands_est_sum"] / spy_opp) if spy_opp else 0.0
    print(f"spy_cast_hidden_lands_est_avg,{avg_hidden_lands:.2f}")
    print(f"spy_cast_no_hidden_lands_opportunities,{total['spy_cast_no_hidden_lands_opportunities']},{pct(total['spy_cast_no_hidden_lands_opportunities'], spy_opp)}")
    print(f"spy_cast_hidden_land_opportunities,{total['spy_cast_hidden_land_opportunities']},{pct(total['spy_cast_hidden_land_opportunities'], spy_opp)}")
    print(f"spy_cast_hidden_lands_est_0,{total['spy_cast_hidden_lands_est_0']},{pct(total['spy_cast_hidden_lands_est_0'], spy_opp)}")
    print(f"spy_cast_hidden_lands_est_1,{total['spy_cast_hidden_lands_est_1']},{pct(total['spy_cast_hidden_lands_est_1'], spy_opp)}")
    print(f"spy_cast_hidden_lands_est_2,{total['spy_cast_hidden_lands_est_2']},{pct(total['spy_cast_hidden_lands_est_2'], spy_opp)}")
    print(f"spy_cast_hidden_lands_est_3,{total['spy_cast_hidden_lands_est_3']},{pct(total['spy_cast_hidden_lands_est_3'], spy_opp)}")
    print(f"spy_cast_hidden_lands_est_4,{total['spy_cast_hidden_lands_est_4']},{pct(total['spy_cast_hidden_lands_est_4'], spy_opp)}")
    print(f"spy_casts_no_hidden_lands,{total['spy_casts_no_hidden_lands']},{pct(total['spy_casts_no_hidden_lands'], total['spy_casts'])}")
    print(f"spy_casts_with_hidden_lands,{total['spy_casts_with_hidden_lands']},{pct(total['spy_casts_with_hidden_lands'], total['spy_casts'])}")
    print(f"spy_seen_on_battlefield_games,{total['spy_seen_on_battlefield_games']},{pct(total['spy_seen_on_battlefield_games'], games)}")
    print(f"dread_return_seen_in_graveyard_games,{total['dread_return_seen_in_graveyard_games']},{pct(total['dread_return_seen_in_graveyard_games'], games)}")
    print(f"lotleth_seen_in_graveyard_games,{total['lotleth_seen_in_graveyard_games']},{pct(total['lotleth_seen_in_graveyard_games'], games)}")
    print(f"three_creatures_in_graveyard_games,{total['three_creatures_in_graveyard_games']},{pct(total['three_creatures_in_graveyard_games'], games)}")
    print(f"combo_payload_in_graveyard_games,{total['combo_payload_in_graveyard_games']},{pct(total['combo_payload_in_graveyard_games'], games)}")
    print(f"combo_ready_graveyard_games,{total['combo_ready_graveyard_games']},{pct(total['combo_ready_graveyard_games'], games)}")
    print(f"dread_return_flashback_selected,{total['dread_return_flashback_selected']}")
    print(f"dread_return_flashback_games,{total['dread_return_flashback_games']},{pct(total['dread_return_flashback_games'], games)}")
    print(f"premature_dread_flashback_no_lotleth_graveyard,{total['premature_dread_flashback_no_lotleth_graveyard']},{pct(total['premature_dread_flashback_no_lotleth_graveyard'], total['dread_return_flashback_selected'])}")
    print(f"premature_dread_flashback_no_lotleth_graveyard_games,{total['premature_dread_flashback_no_lotleth_graveyard_games']},{pct(total['premature_dread_flashback_no_lotleth_graveyard_games'], games)}")
    print(f"premature_dread_flashback_not_combo_ready,{total['premature_dread_flashback_not_combo_ready']},{pct(total['premature_dread_flashback_not_combo_ready'], total['dread_return_flashback_selected'])}")
    print(f"premature_dread_flashback_not_combo_ready_games,{total['premature_dread_flashback_not_combo_ready_games']},{pct(total['premature_dread_flashback_not_combo_ready_games'], games)}")
    print(f"dread_return_target_decisions,{dread_targets}")
    print(f"dread_return_target_no_lotleth_available,{total['dread_return_target_no_lotleth_available']},{pct(total['dread_return_target_no_lotleth_available'], dread_targets)}")
    print(f"dread_return_target_no_lotleth_available_games,{total['dread_return_target_no_lotleth_available_games']},{pct(total['dread_return_target_no_lotleth_available_games'], games)}")
    print(f"dread_return_lotleth_target_opportunities,{dread_lotleth_opp},{pct(dread_lotleth_opp, dread_targets)}")
    print(f"dread_return_lotleth_target_opportunity_games,{total['dread_return_lotleth_target_opportunity_games']},{pct(total['dread_return_lotleth_target_opportunity_games'], games)}")
    print(f"dread_return_target_lotleth,{total['dread_return_target_lotleth']},{pct(total['dread_return_target_lotleth'], dread_targets)}")
    print(f"dread_return_target_lotleth_when_available,{total['dread_return_target_lotleth_when_available']},{pct(total['dread_return_target_lotleth_when_available'], dread_lotleth_opp)}")
    print(f"dread_return_target_lotleth_games,{total['dread_return_target_lotleth_games']},{pct(total['dread_return_target_lotleth_games'], games)}")
    print(f"dread_return_target_other,{total['dread_return_target_other']},{pct(total['dread_return_target_other'], dread_targets)}")
    print(f"dread_return_target_other_over_lotleth,{total['dread_return_target_other_over_lotleth']},{pct(total['dread_return_target_other_over_lotleth'], dread_lotleth_opp)}")
    print(f"dread_return_sacrifice_decisions,{total['dread_return_sacrifice_decisions']}")
    print(f"lotleth_seen_on_battlefield_games,{total['lotleth_seen_on_battlefield_games']},{pct(total['lotleth_seen_on_battlefield_games'], games)}")
    print(f"lotleth_trigger_target_selected,{total['lotleth_trigger_target_selected']}")
    print(f"lotleth_trigger_target_selected_games,{total['lotleth_trigger_target_selected_games']},{pct(total['lotleth_trigger_target_selected_games'], games)}")
    print(f"dread_return_seen_in_exile_games,{total['dread_return_seen_in_exile_games']},{pct(total['dread_return_seen_in_exile_games'], games)}")
    print(f"lotleth_seen_in_exile_games,{total['lotleth_seen_in_exile_games']},{pct(total['lotleth_seen_in_exile_games'], games)}")
    cleansing = total["cleansing_target_decisions"]
    print(f"cleansing_target_decisions,{cleansing}")
    print(f"cleansing_target_self_bridge,{total['cleansing_target_self_bridge']},{pct(total['cleansing_target_self_bridge'], cleansing)}")
    print(f"cleansing_target_self_other,{total['cleansing_target_self_other']},{pct(total['cleansing_target_self_other'], cleansing)}")
    print(f"cleansing_target_opponent,{total['cleansing_target_opponent']},{pct(total['cleansing_target_opponent'], cleansing)}")
    print(f"cleansing_target_unknown,{total['cleansing_target_unknown']},{pct(total['cleansing_target_unknown'], cleansing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
