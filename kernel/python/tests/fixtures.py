from __future__ import annotations

import copy
import os
import stat
import sys
from pathlib import Path
from typing import Any

PROVENANCE = {
    "protocol": "kernel_rl_jsonl",
    "protocol_version": 2,
    "schema_version": 2,
    "kernel_version": "0.0.1-spike",
    "surface_version": 2,
    "card_db_hash": 13755609902749199750,
}


def stable_ref(arena_id: int, card_db_id: int, owner: str = "p0", zone: str = "Hand") -> dict[str, Any]:
    return {
        "arena_id": arena_id,
        "card_db_id": card_db_id,
        "owner": owner,
        "controller": owner,
        "zone": zone,
        "zone_change_count": 0,
    }


def public_card(arena_id: int, card_db_id: int, owner: str, zone: str = "Battlefield") -> dict[str, Any]:
    return {
        "stable": stable_ref(arena_id, card_db_id, owner, zone),
        "card_name": f"Name {card_db_id}",
        "tapped": False,
        "summoning_sick": False,
        "damage": 0,
        "counters": {"plus1_plus1": 0},
        "attachments": [],
        "plotted_turn": None,
        "characteristics": {
            "type_flags": {
                "land": zone == "Battlefield" and card_db_id in (10, 11),
                "creature": card_db_id in (20, 21),
                "instant": card_db_id in (30, 31),
                "sorcery": False,
                "artifact": False,
                "enchantment": False,
            },
            "base_power": 2 if card_db_id in (20, 21) else None,
            "base_toughness": 2 if card_db_id in (20, 21) else None,
            "effective_power": 2 if card_db_id in (20, 21) else None,
            "effective_toughness": 2 if card_db_id in (20, 21) else None,
            "effective_keywords": {
                "flying": False,
                "reach": False,
                "haste": card_db_id == 20,
                "vigilance": False,
                "trample": False,
                "first_strike": False,
                "double_strike": False,
                "deathtouch": False,
                "menace": False,
                "defender": False,
            },
        },
    }


def complete_observation() -> dict[str, Any]:
    obs = observation()
    p = obs["projection"]
    p0_creature = p["battlefield"][0][0]
    p0_land = p["battlefield"][0][1]
    p1_creature = p["battlefield"][1][0]
    p1_land = p["battlefield"][1][1]
    attachment = public_card(10, 41, "p0")
    p["battlefield"][0].append(attachment)
    p0_creature["attachments"] = [attachment["stable"]["arena_id"]]
    p["stack"][0]["controller"] = "p1"
    p["stack"][0]["targets"] = [
        {"target_kind": "player", "player": "p1"},
        {"target_kind": "object", "object": p1_creature["stable"]},
    ]
    p["combat"]["ordered_attackers"] = [p0_creature["stable"], p0_land["stable"]]
    p["combat"]["attacker_to_ordered_blockers"] = [
        [p0_creature["stable"], [p1_creature["stable"], p1_land["stable"]]],
        [p0_land["stable"], []],
    ]
    p["continuous_effects"] = [
        {
            "affected_objects": [p0_creature["stable"], p1_creature["stable"]],
            "layers": 7,
            "timestamp": 42,
            "duration": "end_of_turn",
            "power_delta": 1,
            "toughness_delta": -1,
            "grants_haste": True,
        }
    ]
    p["exile_play_permissions"] = [
        {
            "object": stable_ref(9, 40, "p0", "Exile"),
            "holder": "p0",
            "play_or_cast": "cast",
            "zone_change_generation": 2,
            "expiry": {"expiry_kind": "until_holders_next_turn", "holder_turn_started": False},
        }
    ]
    p["engine_context"].update(
        {
            "pending_cast": {
                "source": obs["own_hand"][0]["stable"],
                "controller": "p0",
                "chosen_targets": [{"target_kind": "object", "object": p1_creature["stable"]}],
                "is_flashback": False,
                "cast_mode": "Alternative",
                "additional_cost_discarded": [obs["own_hand"][0]["stable"]],
                "mode_chosen": 1,
                "origin_zone": "Hand",
                "sacrifice_chosen": [p0_land["stable"]],
                "kicked": True,
            },
            "pending_activation": {
                "source": p0_land["stable"],
                "controller": "p0",
                "ability_index": 2,
                "chosen_targets": [{"target_kind": "player", "player": "p1"}],
                "cost_discard_paid": [obs["own_hand"][0]["stable"]],
            },
            "pending_discard": {
                "player": "p0",
                "count": 1,
                "resume_stage": "finish_cast",
                "resume_source": obs["own_hand"][0]["stable"],
            },
            "pending_optional_cost": {
                "player": "p0",
                "source": obs["own_hand"][0]["stable"],
                "discard_cards": 1,
                "sacrifice_lands": 1,
                "discard_payable": True,
                "sacrifice_payable": True,
                "spell_resume_source": obs["own_hand"][0]["stable"],
                "spell_resume_zone": "Hand",
            },
            "pending_optional_cost_sacrifice": {
                "player": "p0",
                "source": obs["own_hand"][0]["stable"],
                "remaining": 1,
                "chosen": [p0_land["stable"]],
                "spell_resume_source": obs["own_hand"][0]["stable"],
                "spell_resume_zone": "Hand",
            },
            "pending_triggers": [
                {"source": p0_creature["stable"], "controller": "p0", "trigger_kind": "triggered_ability", "kicked": False},
                {"source": p1_creature["stable"], "controller": "p1", "trigger_kind": "madness_offer", "kicked": True},
            ],
        }
    )
    p["surface_context"].update(
        {
            "current_stage": "declare_blockers_for_attacker",
            "stack_length_changed_since_observed": True,
            "madness_cast_reprompt_source": obs["own_hand"][0]["stable"],
            "private_blockers": {
                "current_attacker": p0_creature["stable"],
                "accumulated": [[p0_creature["stable"], p1_creature["stable"]]],
                "remaining": [[p0_land["stable"], [p1_land["stable"]]]],
            },
            "private_discard": {
                "chosen": [obs["own_hand"][0]["stable"]],
                "remaining_choices": [stable_ref(11, 31, "p0", "Hand")],
                "remaining_needed": 1,
            },
            "private_optional_cost": {
                "discard_payable": True,
                "sacrifice_payable": True,
                "stage": "optional_cost_which",
            },
        }
    )
    return obs


def observation() -> dict[str, Any]:
    self_hand = {"stable": stable_ref(1, 30, "p0", "Hand"), "card_name": "Lightning Bolt"}
    p0_creature = public_card(2, 20, "p0")
    p0_land = public_card(3, 10, "p0")
    p1_creature = public_card(4, 21, "p1")
    p1_land = public_card(5, 11, "p1")
    return {
        "schema_version": 2,
        "kernel_version": PROVENANCE["kernel_version"],
        "surface_version": PROVENANCE["surface_version"],
        "card_db_hash": PROVENANCE["card_db_hash"],
        "acting_player": "p0",
        "step_index": 0,
        "projection": {
            "turn": 1,
            "phase": "main1",
            "active_player": "p0",
            "priority_player": "p0",
            "life_totals": [20, 18],
            "mana_pools": [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
            "hand_counts": [1, 2],
            "library_counts": [53, 53],
            "player_status": [
                {"has_lost": False, "lands_played_this_turn": 0, "drew_from_empty": False, "draws_this_turn": 1},
                {"has_lost": False, "lands_played_this_turn": 1, "drew_from_empty": False, "draws_this_turn": 1},
            ],
            "battlefield": [[p0_creature, p0_land], [p1_creature, p1_land]],
            "graveyards": [[public_card(6, 31, "p0", "Graveyard")], [public_card(7, 32, "p1", "Graveyard")]],
            "exile": [],
            "stack": [
                {
                    "stack_index": 0,
                    "source": stable_ref(8, 30, "p0", "Stack"),
                    "controller": "p0",
                    "targets": [{"target_kind": "player", "player": "p1"}],
                    "stack_item_kind": "spell",
                    "is_flashback": False,
                    "mode_chosen": 0,
                    "madness_offer": False,
                    "kicked": False,
                }
            ],
            "combat": {
                "attackers_declared": True,
                "blockers_declared": False,
                "ordered_attackers": [p0_creature["stable"]],
                "attacker_to_ordered_blockers": [[p0_creature["stable"], [p1_creature["stable"]]]],
            },
            "continuous_effects": [
                {
                    "affected_objects": [p0_creature["stable"]],
                    "layers": 1,
                    "timestamp": 42,
                    "duration": "end_of_turn",
                    "power_delta": 1,
                    "toughness_delta": 0,
                    "grants_haste": False,
                }
            ],
            "exile_play_permissions": [
                {
                    "object": stable_ref(9, 40, "p0", "Exile"),
                    "holder": "p0",
                    "play_or_cast": "cast",
                    "zone_change_generation": 0,
                    "expiry": {"expiry_kind": "end_of_turn"},
                }
            ],
            "engine_context": {
                "priority_passes": [False, False],
                "stack_nonempty": True,
                "stack_activity_since_priority_boundary": True,
                "mana_activity_since_priority_boundary": False,
                "last_mana_ability_activator_since_priority_boundary": None,
                "current_stage": "priority",
                "pending_cast": None,
                "pending_activation": None,
                "pending_discard": None,
                "pending_optional_cost": None,
                "pending_optional_cost_sacrifice": None,
                "pending_triggers": [],
            },
            "surface_context": {
                "current_stage": "priority",
                "combat_priority_spent": [False, False],
                "combat_priority_rearmed_by_stack_activity": False,
                "combat_priority_rearmed_by_mana_activity": False,
                "stack_grew_since_round_open": True,
                "mana_activity_since_round_open": False,
                "stack_length_changed_since_observed": None,
                "mana_activity_since_last_stack_change": False,
                "madness_cast_reprompt_source": None,
                "private_blockers": None,
                "private_discard": None,
                "private_optional_cost": None,
            },
        },
        "own_hand": [self_hand],
        "visible_projection_hash": 123456,
    }


def legal_actions() -> list[dict[str, Any]]:
    src = stable_ref(1, 30, "p0", "Hand")
    return [
        {"schema_version": 2, "selected_index": 0, "stable_id": "legal-action-v2:a", "semantic": {"action_kind": "pass", "actor": "p0"}, "display_text": "Pass"},
        {"schema_version": 2, "selected_index": 1, "stable_id": "legal-action-v2:b", "semantic": {"action_kind": "cast_spell", "actor": "p0", "source": src}, "display_text": "Cast Lightning Bolt"},
        {"schema_version": 2, "selected_index": 2, "stable_id": "legal-action-v2:c", "semantic": {"action_kind": "choose_target", "actor": "p0", "source": src, "remaining": 1, "target": {"target_kind": "player", "player": "p1"}}, "display_text": "Target opponent"},
    ]


def complete_legal_actions() -> list[dict[str, Any]]:
    base = stable_ref(1, 30, "p0", "Hand")
    second = stable_ref(12, 31, "p0", "Hand")
    blocker = stable_ref(4, 21, "p1", "Battlefield")
    return [
        {"schema_version": 2, "selected_index": 0, "stable_id": "a0", "semantic": {"action_kind": "pass", "actor": "p0"}, "display_text": "Pass"},
        {"schema_version": 2, "selected_index": 1, "stable_id": "a1", "semantic": {"action_kind": "choose_target", "actor": "p0", "source": base, "remaining": 1, "target": {"target_kind": "object", "object": blocker}}, "display_text": "Target creature"},
        {"schema_version": 2, "selected_index": 2, "stable_id": "a2", "semantic": {"action_kind": "declare_blockers_for_attacker", "actor": "p0", "attacker": base, "blockers": [blocker]}, "display_text": "Block"},
        {"schema_version": 2, "selected_index": 3, "stable_id": "a3", "semantic": {"action_kind": "discard", "actor": "p0", "cards": [base, second]}, "display_text": "Discard two"},
        {"schema_version": 2, "selected_index": 4, "stable_id": "a4", "semantic": {"action_kind": "order_triggers", "actor": "p0", "pending_sources": [base, second], "order": [1, 0]}, "display_text": "Order triggers"},
    ]


def decision_response(request_id: str = "r0", episode_id: int = 0, step: int = 0) -> dict[str, Any]:
    obs = observation()
    obs["step_index"] = step
    return {
        "response_type": "decision",
        "schema_version": 2,
        "request_id": request_id,
        "provenance": copy.deepcopy(PROVENANCE),
        "episode_id": episode_id,
        "step": step,
        "acting_player": obs["acting_player"],
        "observation": obs,
        "legal_actions": legal_actions(),
        "reward": [0, 0],
    }


def terminal_response(request_id: str = "r1", episode_id: int = 0, decisions: int = 1) -> dict[str, Any]:
    return {
        "response_type": "terminal",
        "schema_version": 2,
        "request_id": request_id,
        "provenance": copy.deepcopy(PROVENANCE),
        "episode_id": episode_id,
        "terminal_outcome": "p0_win",
        "terminal_classification": "natural",
        "terminal_code": "natural_game_over",
        "winner": "p0",
        "terminal_reward": [1, -1],
        "terminal_reason": "game_over",
        "decision_count": decisions,
    }


def fake_launcher(tmp: Path, scenario: str) -> Path:
    script = Path(__file__).with_name("fake_env.py")
    if os.name == "nt":
        launcher = tmp / f"fake_{scenario}.cmd"
        launcher.write_text(f"@echo off\nset FAKE_SCENARIO={scenario}\n\"{sys.executable}\" \"{script}\"\n", encoding="utf-8")
    else:
        launcher = tmp / f"fake_{scenario}.sh"
        launcher.write_text(f"#!/usr/bin/env sh\nFAKE_SCENARIO={scenario} exec \"{sys.executable}\" \"{script}\"\n", encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return launcher


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)
