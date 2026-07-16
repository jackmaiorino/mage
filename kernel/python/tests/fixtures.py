from __future__ import annotations

import copy
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

PROVENANCE = {
    "protocol": "kernel_rl_jsonl",
    "protocol_version": 4,
    "schema_version": 4,
    "kernel_version": "0.0.3-spike",
    "surface_version": 2,
    "card_db_hash": 13755609902749199750,
}
DECK_IDS = ("Burn", "Burn")
DECK_HASHES = (0x4255_524E_0000_0001, 0x4255_524E_0000_0001)


def stable_ref(arena_id: int, card_db_id: int, owner: str = "p0", zone: str = "Hand", zone_change_count: int = 0, controller: str | None = None) -> dict[str, Any]:
    return {
        "arena_id": arena_id,
        "card_db_id": card_db_id,
        "owner": owner,
        "controller": owner if controller is None else controller,
        "zone": zone,
        "zone_change_count": zone_change_count,
    }


def public_card(arena_id: int, card_db_id: int, owner: str, zone: str = "Battlefield") -> dict[str, Any]:
    return {
        "stable": stable_ref(arena_id, card_db_id, owner, zone),
        "card_name": f"Name {card_db_id}",
        "tapped": False,
        "summoning_sick": False,
        "damage": 0,
        "counters": {
            "plus1_plus1": 0,
            "minus1_minus1": 0,
            "minus0_minus1": 0,
            "stun": 0,
            "lore": 0,
        },
        "attachments": [],
        "plotted_turn": None,
        "is_token": False,
        "face_index": 0,
        "chosen_color": None,
        "entered_battlefield_turn": None,
        "ability_uses_this_turn": [],
        "skip_next_untap": False,
        "goaded_by": [],
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
            "effective_color_mask": 8 if card_db_id in (10, 20, 30, 31) else 0,
            "effective_subtype_ids": [1] if card_db_id in (20, 21) else [],
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
                "lifelink": False,
                "hexproof": False,
                "indestructible": False,
                "protection_from_monocolored": False,
                "ward_generic": 0,
                "minimum_blockers": 1 if card_db_id in (20, 21) else 0,
                "landwalk_mask": 0,
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
    # The live v4 contract intentionally preserves payment-time incarnation
    # provenance even after the object has changed zones/incarnations.
    p0_land["stable"]["zone_change_count"] = 1
    historical_paid_land = stable_ref(
        p0_land["stable"]["arena_id"],
        p0_land["stable"]["card_db_id"],
        "p0",
        "Battlefield",
        zone_change_count=0,
    )
    attachment = public_card(10, 41, "p0")
    p["battlefield"][0].append(attachment)
    p0_creature["attachments"] = [attachment["stable"]["arena_id"]]
    p0_creature.update(
        {
            "counters": {"plus1_plus1": 2, "minus1_minus1": -1, "minus0_minus1": 1, "stun": 1, "lore": 3},
            "is_token": True,
            "face_index": 1,
            "chosen_color": "R",
            "entered_battlefield_turn": 0,
            "ability_uses_this_turn": [
                {"ability_kind": "mana", "ability_index": 1, "uses": 2},
                {"ability_kind": "activated", "ability_index": 4, "uses": 1},
            ],
            "skip_next_untap": True,
            "goaded_by": [{"player": "p1", "expires_at_turn": 2}],
        }
    )
    p0_creature["characteristics"].update(
        {
            "effective_color_mask": 9,
            "effective_subtype_ids": [1, 17],
        }
    )
    p0_creature["characteristics"]["effective_keywords"].update(
        {
            "lifelink": True,
            "hexproof": True,
            "indestructible": True,
            "protection_from_monocolored": True,
            "ward_generic": 2,
            "minimum_blockers": 2,
            "landwalk_mask": 8,
        }
    )
    p["exile"][0]["stable"]["zone_change_count"] = 2
    p["stack"][0]["controller"] = "p1"
    p["stack"][0]["targets"] = [
        {"target_kind": "player", "player": "p1"},
        {"target_kind": "object", "object": p1_creature["stable"]},
    ]
    p["stack"][0].update(
        {
            "cast_method": "alternative",
            "face_index": 1,
            "x_value": 3,
            "paid_cost_refs": [historical_paid_land],
        }
    )
    p["combat"]["ordered_attackers"] = [p0_creature["stable"], p0_land["stable"]]
    p["combat"]["attacker_to_ordered_blockers"] = [
        [p0_creature["stable"], [p1_creature["stable"], p1_land["stable"]]],
        [p0_land["stable"], []],
    ]
    p["continuous_effects"] = [
        {
            "source": p["stack"][0]["source"],
            "controller": "p1",
            "affected_objects": [p0_creature["stable"], p1_creature["stable"]],
            "affected_players": ["p0", "p1"],
            "global": True,
            "layers": 7,
            "timestamp": 42,
            "duration": "end_of_turn",
            "power_delta": 1,
            "toughness_delta": -1,
            "grants_haste": True,
            "set_power": 4,
            "set_toughness": 5,
            "add_color_mask": 8,
            "remove_color_mask": 2,
            "add_subtype_ids": [2, 9],
            "remove_subtype_ids": [1],
            "add_keyword_mask": 17,
            "remove_keyword_mask": 2,
            "ward_generic_delta": 2,
            "minimum_blockers": 3,
            "add_landwalk_mask": 8,
            "remove_landwalk_mask": 1,
            "prevent_damage_from_color_mask": 4,
            "damage_cannot_be_prevented": True,
        }
    ]
    p["object_relations"] = [
        {"relation_kind": "attached_to", "object": attachment["stable"], "attached_to": p0_creature["stable"]},
        {"relation_kind": "exiled_by", "object": p["exile"][0]["stable"], "exiled_by": p0_creature["stable"]},
    ]
    p["exile_play_permissions"] = [
        {
            "object": stable_ref(9, 40, "p0", "Exile", zone_change_count=2),
            "holder": "p0",
            "play_or_cast": "cast",
            "zone_change_generation": 2,
            "expiry": {"expiry_kind": "until_holders_next_turn", "holder_turn_started": False},
        }
    ]
    p["surface_context"].update(
        {
            "stack_length_changed_since_observed": True,
            "madness_cast_reprompt_source": obs["own_hand"][0]["stable"],
        }
    )
    p["initiative"] = "p1"
    p["player_status"][0].update(
        {
            "spells_cast_this_turn": 2,
            "dungeon": {"dungeon_id": 3, "room_id": 2, "completed_dungeons": [1, 4]},
        }
    )
    obs["known_library_cards"] = [
        [{"position": 0, "card": {"stable": stable_ref(113, 30, "p0", "Library"), "card_name": "Known own top"}}],
        [{"position": 1, "card": {"stable": stable_ref(114, 31, "p1", "Library"), "card_name": "Known opponent"}}],
    ]
    obs["known_hand_cards"] = [
        [],
        [{"stable": stable_ref(115, 32, "p1", "Hand"), "card_name": "Revealed opponent card"}],
    ]
    return obs


def observation() -> dict[str, Any]:
    self_hand = {"stable": stable_ref(1, 30, "p0", "Hand"), "card_name": "Lightning Bolt"}
    second_hand = {"stable": stable_ref(12, 31, "p0", "Hand"), "card_name": "Lava Dart"}
    p0_creature = public_card(2, 20, "p0")
    p0_land = public_card(3, 10, "p0")
    p1_creature = public_card(4, 21, "p1")
    p1_land = public_card(5, 11, "p1")
    exile_card = public_card(9, 40, "p0", "Exile")
    return {
        "schema_version": 4,
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
            "initiative": None,
            "life_totals": [20, 18],
            "mana_pools": [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
            "hand_counts": [2, 2],
            "library_counts": [53, 53],
            "player_status": [
                {"has_lost": False, "lands_played_this_turn": 0, "drew_from_empty": False, "draws_this_turn": 1, "spells_cast_this_turn": 0, "dungeon": {"dungeon_id": None, "room_id": None, "completed_dungeons": []}},
                {"has_lost": False, "lands_played_this_turn": 1, "drew_from_empty": False, "draws_this_turn": 1, "spells_cast_this_turn": 1, "dungeon": {"dungeon_id": None, "room_id": None, "completed_dungeons": []}},
            ],
            "battlefield": [[p0_creature, p0_land], [p1_creature, p1_land]],
            "graveyards": [[public_card(6, 31, "p0", "Graveyard")], [public_card(7, 32, "p1", "Graveyard")]],
            "exile": [exile_card],
            "stack": [
                {
                    "stack_index": 0,
                    "source": stable_ref(8, 30, "p0", "Stack"),
                    "controller": "p0",
                    "targets": [{"target_kind": "player", "player": "p1"}],
                    "stack_item_kind": "spell",
                    "is_copy": False,
                    "is_flashback": False,
                    "mode_chosen": 0,
                    "madness_offer": False,
                    "kicked": False,
                    "cast_method": "normal",
                    "face_index": 0,
                    "x_value": 0,
                    "paid_cost_refs": [],
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
                    "source": None,
                    "controller": None,
                    "affected_objects": [p0_creature["stable"]],
                    "affected_players": [],
                    "global": False,
                    "layers": 1,
                    "timestamp": 42,
                    "duration": "end_of_turn",
                    "power_delta": 1,
                    "toughness_delta": 0,
                    "grants_haste": False,
                    "set_power": None,
                    "set_toughness": None,
                    "add_color_mask": 0,
                    "remove_color_mask": 0,
                    "add_subtype_ids": [],
                    "remove_subtype_ids": [],
                    "add_keyword_mask": 0,
                    "remove_keyword_mask": 0,
                    "ward_generic_delta": 0,
                    "minimum_blockers": None,
                    "add_landwalk_mask": 0,
                    "remove_landwalk_mask": 0,
                    "prevent_damage_from_color_mask": 0,
                    "damage_cannot_be_prevented": False,
                }
            ],
            "object_relations": [],
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
                "pending_spell_copy": None,
                "pending_effect": None,
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
        "own_hand": [self_hand, second_hand],
        "known_library_cards": [[], []],
        "known_hand_cards": [[], []],
        "visible_projection_hash": 123456,
    }


def legal_actions(actor: str = "p0") -> list[dict[str, Any]]:
    src = stable_ref(1 if actor == "p0" else 101, 30, actor, "Hand")
    opponent = "p1" if actor == "p0" else "p0"
    return [
        {"schema_version": 4, "selected_index": 0, "stable_id": f"legal-action-v4:{actor}:a", "semantic": {"action_kind": "pass", "actor": actor}, "display_text": "Pass"},
        {"schema_version": 4, "selected_index": 1, "stable_id": f"legal-action-v4:{actor}:b", "semantic": {"action_kind": "cast_spell", "actor": actor, "source": src}, "display_text": "Cast Lightning Bolt"},
        {"schema_version": 4, "selected_index": 2, "stable_id": f"legal-action-v4:{actor}:c", "semantic": {"action_kind": "choose_target", "actor": actor, "source": src, "remaining": 1, "target": {"target_kind": "player", "player": opponent}}, "display_text": "Target opponent"},
    ]


def complete_legal_actions() -> list[dict[str, Any]]:
    base = stable_ref(1, 30, "p0", "Hand")
    second = stable_ref(12, 31, "p0", "Hand")
    blocker = stable_ref(4, 21, "p1", "Battlefield")
    return [
        {"schema_version": 4, "selected_index": 0, "stable_id": "legal-action-v4:a0", "semantic": {"action_kind": "pass", "actor": "p0"}, "display_text": "Pass"},
        {"schema_version": 4, "selected_index": 1, "stable_id": "legal-action-v4:a1", "semantic": {"action_kind": "choose_target", "actor": "p0", "source": base, "remaining": 1, "target": {"target_kind": "object", "object": blocker}}, "display_text": "Target creature"},
        {"schema_version": 4, "selected_index": 2, "stable_id": "legal-action-v4:a2", "semantic": {"action_kind": "declare_blockers_for_attacker", "actor": "p0", "attacker": base, "blockers": [blocker]}, "display_text": "Block"},
        {"schema_version": 4, "selected_index": 3, "stable_id": "legal-action-v4:a3", "semantic": {"action_kind": "discard", "actor": "p0", "cards": [base, second]}, "display_text": "Discard two"},
        {"schema_version": 4, "selected_index": 4, "stable_id": "legal-action-v4:a4", "semantic": {"action_kind": "order_triggers", "actor": "p0", "pending_sources": [base, second], "order": [1, 0]}, "display_text": "Order triggers"},
    ]


def actor_observation(actor: str, step: int = 0) -> dict[str, Any]:
    obs = observation()
    obs["acting_player"] = actor
    obs["step_index"] = step
    obs["projection"]["priority_player"] = actor
    obs["own_hand"] = [
        {"stable": stable_ref(1 if actor == "p0" else 101, 30, actor, "Hand"), "card_name": "Lightning Bolt"},
        {"stable": stable_ref(12 if actor == "p0" else 112, 31, actor, "Hand"), "card_name": "Lava Dart"},
    ]
    return obs


def decision_response(request_id: str = "r0", episode_id: int = 0, step: int = 0, actor: str = "p0") -> dict[str, Any]:
    obs = actor_observation(actor, step)
    return {
        "response_type": "decision",
        "schema_version": 4,
        "request_id": request_id,
        "provenance": copy.deepcopy(PROVENANCE),
        "deck_ids": list(DECK_IDS),
        "deck_hashes": list(DECK_HASHES),
        "episode_id": episode_id,
        "step": step,
        "acting_player": actor,
        "observation": obs,
        "legal_actions": legal_actions(actor),
        "reward": [0, 0],
    }


def terminal_response(request_id: str = "r1", episode_id: int = 0, decisions: int = 1, outcome: str = "p0_win") -> dict[str, Any]:
    if outcome == "p0_win":
        winner: str | None = "p0"
        reward = [1, -1]
    elif outcome == "p1_win":
        winner = "p1"
        reward = [-1, 1]
    elif outcome == "draw":
        winner = None
        reward = [0, 0]
    else:
        raise ValueError(outcome)
    return {
        "response_type": "terminal",
        "schema_version": 4,
        "request_id": request_id,
        "provenance": copy.deepcopy(PROVENANCE),
        "deck_ids": list(DECK_IDS),
        "deck_hashes": list(DECK_HASHES),
        "episode_id": episode_id,
        "terminal_outcome": outcome,
        "terminal_classification": "natural",
        "terminal_code": "natural_game_over",
        "winner": winner,
        "terminal_reward": reward,
        "terminal_reason": "game_over",
        "decision_count": decisions,
    }


def fake_launcher(tmp: Path, scenario: str, extra_env: dict[str, str] | None = None) -> Path:
    script = Path(__file__).with_name("fake_env.py")
    extra_env = extra_env or {}
    if os.name == "nt":
        launcher = tmp / f"fake_{scenario}.cmd"
        lines = ["@echo off", f"set FAKE_SCENARIO={scenario}"]
        for key, value in extra_env.items():
            lines.append(f"set {key}={value}")
        lines.append(f"\"{sys.executable}\" \"{script}\"")
        launcher.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        launcher = tmp / f"fake_{scenario}.sh"
        exports = " ".join(f"{key}={json.dumps(value)}" for key, value in {"FAKE_SCENARIO": scenario, **extra_env}.items())
        launcher.write_text(f"#!/usr/bin/env sh\n{exports} exec \"{sys.executable}\" \"{script}\"\n", encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    return launcher


def deep_copy(value: Any) -> Any:
    return copy.deepcopy(value)
