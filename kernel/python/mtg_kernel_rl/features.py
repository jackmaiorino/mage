"""Actor-relative feature encoding with explicit leaf classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

MODEL_INPUT = "model_input"
OPERATIONAL_ONLY = "operational_only"
FORBIDDEN = "forbidden"

PHASES = [
    "untap",
    "upkeep",
    "draw",
    "main1",
    "begin_combat",
    "declare_attackers",
    "declare_blockers",
    "combat_damage",
    "end_combat",
    "main2",
    "end",
    "cleanup",
]
ZONES = ["Library", "Hand", "Battlefield", "Graveyard", "Stack", "Exile", "Command"]
ENGINE_STAGES = [
    "priority",
    "pending_cast",
    "pending_activation",
    "pending_discard",
    "pending_optional_cost",
    "pending_optional_cost_sacrifice",
    "pending_triggers",
    "halted",
]
SURFACE_STAGES = ["priority", "declare_blockers_for_attacker", "discard_pick", "optional_cost_use", "optional_cost_which"]
STACK_KINDS = ["spell", "activated_ability", "triggered_ability", "madness_offer"]
ACTION_KINDS = [
    "pass",
    "play_land",
    "cast_spell",
    "activate_mana_ability",
    "activate_ability",
    "plot_spell",
    "choose_target",
    "choose_cost_target",
    "choose_cast_mode",
    "choose_kicker",
    "choose_spell_mode",
    "choose_optional_cost_use",
    "choose_optional_cost_which",
    "choose_madness_cast",
    "discard",
    "declare_attackers",
    "declare_blockers_for_attacker",
    "order_triggers",
]
CAST_MODES = ["Normal", "Alternative"]
COST_KINDS = ["SacrificeLands"]
OPTIONAL_COST_CHOICES = ["Decline", "Discard", "SacrificeLand"]
TARGET_KINDS = ["player", "object"]
OBJECT_GROUPS = [
    "self_hand",
    "self_battlefield",
    "opponent_battlefield",
    "self_graveyard",
    "opponent_graveyard",
    "exile",
    "stack",
    "combat",
    "effect",
    "permission",
]
MAX_ACTION_CARD_REFS = 4

FORBIDDEN_FIELDS = {
    "card_name",
    "display_text",
    "stable_id",
    "selected_index",
    "visible_projection_hash",
    "kernel_version",
    "surface_version",
    "card_db_hash",
    "schema_version",
    "request_id",
    "episode_key",
    "game_id",
    "env_seed",
    "policy_seed",
    "diagnostic_state_hash",
    "state_hash",
    "terminal_reason",
    "reason",
}
OPERATIONAL_FIELDS = {
    "arena_id",
    "zone_change_count",
    "timestamp",
    "step_index",
    "episode_id",
    "record_type",
    "stream_safety",
    "matchup",
    "deck_identifiers",
    "observation_projection_hash",
    "reward",
    "terminal_outcome",
    "terminal_classification",
    "terminal_code",
    "terminal_reward",
    "decision_count",
    "winner",
    "protocol",
    "protocol_version",
}
MODEL_FIELDS = {
    "acting_player",
    "projection",
    "own_hand",
    "stable",
    "card_db_id",
    "owner",
    "controller",
    "zone",
    "turn",
    "phase",
    "active_player",
    "priority_player",
    "life_totals",
    "mana_pools",
    "hand_counts",
    "library_counts",
    "player_status",
    "has_lost",
    "lands_played_this_turn",
    "drew_from_empty",
    "draws_this_turn",
    "battlefield",
    "graveyards",
    "exile",
    "stack",
    "combat",
    "continuous_effects",
    "exile_play_permissions",
    "type_flags",
    "land",
    "creature",
    "instant",
    "sorcery",
    "artifact",
    "enchantment",
    "base_power",
    "base_toughness",
    "effective_power",
    "effective_toughness",
    "effective_keywords",
    "flying",
    "reach",
    "haste",
    "vigilance",
    "trample",
    "first_strike",
    "double_strike",
    "deathtouch",
    "menace",
    "defender",
    "tapped",
    "summoning_sick",
    "damage",
    "counters",
    "plus1_plus1",
    "attachments",
    "plotted_turn",
    "characteristics",
    "stack_index",
    "source",
    "targets",
    "target_kind",
    "player",
    "object",
    "stack_item_kind",
    "is_flashback",
    "mode_chosen",
    "madness_offer",
    "kicked",
    "attackers_declared",
    "blockers_declared",
    "ordered_attackers",
    "attacker_to_ordered_blockers",
    "affected_objects",
    "layers",
    "duration",
    "power_delta",
    "toughness_delta",
    "grants_haste",
    "holder",
    "play_or_cast",
    "zone_change_generation",
    "expiry",
    "expiry_kind",
    "holder_turn_started",
    "engine_context",
    "priority_passes",
    "stack_nonempty",
    "stack_activity_since_priority_boundary",
    "mana_activity_since_priority_boundary",
    "last_mana_ability_activator_since_priority_boundary",
    "current_stage",
    "pending_cast",
    "pending_activation",
    "pending_discard",
    "pending_optional_cost",
    "pending_optional_cost_sacrifice",
    "pending_triggers",
    "chosen_targets",
    "cast_mode",
    "additional_cost_discarded",
    "origin_zone",
    "sacrifice_chosen",
    "ability_index",
    "cost_discard_paid",
    "count",
    "resume_stage",
    "resume_source",
    "discard_cards",
    "sacrifice_lands",
    "discard_payable",
    "sacrifice_payable",
    "spell_resume_source",
    "spell_resume_zone",
    "remaining",
    "chosen",
    "trigger_kind",
    "surface_context",
    "combat_priority_spent",
    "combat_priority_rearmed_by_stack_activity",
    "combat_priority_rearmed_by_mana_activity",
    "stack_grew_since_round_open",
    "mana_activity_since_round_open",
    "stack_length_changed_since_observed",
    "mana_activity_since_last_stack_change",
    "madness_cast_reprompt_source",
    "private_blockers",
    "private_discard",
    "private_optional_cost",
    "current_attacker",
    "accumulated",
    "remaining_choices",
    "remaining_needed",
    "stage",
    "legal_actions",
    "semantic",
    "action_kind",
    "actor",
    "candidate",
    "attacker",
    "cost_kind",
    "target",
    "mode",
    "pay",
    "mode_index",
    "mode_count",
    "use_cost",
    "choice",
    "card",
    "cast_it",
    "cards",
    "attackers",
    "blockers",
    "pending_sources",
    "order",
}
KNOWN_FIELDS = FORBIDDEN_FIELDS | OPERATIONAL_FIELDS | MODEL_FIELDS

ACTION_FIELDS = {
    "pass": {"action_kind", "actor"},
    "play_land": {"action_kind", "actor", "source"},
    "cast_spell": {"action_kind", "actor", "source"},
    "activate_mana_ability": {"action_kind", "actor", "source"},
    "activate_ability": {"action_kind", "actor", "source", "ability_index"},
    "plot_spell": {"action_kind", "actor", "source"},
    "choose_target": {"action_kind", "actor", "source", "remaining", "target"},
    "choose_cost_target": {"action_kind", "actor", "source", "cost_kind", "remaining", "candidate"},
    "choose_cast_mode": {"action_kind", "actor", "source", "mode"},
    "choose_kicker": {"action_kind", "actor", "source", "pay"},
    "choose_spell_mode": {"action_kind", "actor", "source", "mode_index", "mode_count"},
    "choose_optional_cost_use": {"action_kind", "actor", "use_cost"},
    "choose_optional_cost_which": {"action_kind", "actor", "choice"},
    "choose_madness_cast": {"action_kind", "actor", "card", "cast_it"},
    "discard": {"action_kind", "actor", "cards"},
    "declare_attackers": {"action_kind", "actor", "attackers"},
    "declare_blockers_for_attacker": {"action_kind", "actor", "attacker", "blockers"},
    "order_triggers": {"action_kind", "actor", "pending_sources", "order"},
    "ambiguous": {"action_kind", "reason"},
}


class FeatureSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    state_dim: int
    object_feature_dim: int
    action_feature_dim: int
    object_group_count: int
    max_action_card_refs: int


@dataclass(frozen=True)
class EncodedDecision:
    schema: FeatureSchema
    state: torch.Tensor
    object_features: torch.Tensor
    object_card_ids: torch.Tensor
    object_groups: torch.Tensor
    action_features: torch.Tensor
    action_card_ids: torch.Tensor


def field_classification(field_name: str) -> str:
    if field_name in FORBIDDEN_FIELDS:
        return FORBIDDEN
    if field_name in OPERATIONAL_FIELDS:
        return OPERATIONAL_ONLY
    if field_name in MODEL_FIELDS:
        return MODEL_INPUT
    raise FeatureSchemaError(f"unknown ObservationV2/action leaf or container field: {field_name}")


def _walk_known(value: Any, path: tuple[str, ...]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            field_classification(key)
            _walk_known(child, path + (key,))
    elif isinstance(value, list):
        for child in value:
            _walk_known(child, path + ("[]",))


def assert_observation_classified(observation: dict[str, Any]) -> None:
    _walk_known(observation, ())


def assert_action_classified(action: dict[str, Any]) -> None:
    _walk_known(action, ())
    semantic = action.get("semantic", action)
    if not isinstance(semantic, dict):
        raise FeatureSchemaError("action semantic must be an object")
    kind = semantic.get("action_kind")
    if kind == "ambiguous":
        raise FeatureSchemaError("ambiguous action semantics are fail-closed")
    expected = ACTION_FIELDS.get(kind)
    if expected is None:
        raise FeatureSchemaError(f"unknown action_kind {kind!r}")
    actual = set(semantic)
    if actual != expected:
        raise FeatureSchemaError(f"action {kind} fields mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")


def _one_hot(value: str, choices: list[str]) -> list[float]:
    if value not in choices:
        raise FeatureSchemaError(f"unknown categorical value {value!r}; choices={choices}")
    return [1.0 if value == choice else 0.0 for choice in choices]


def _seat(value: str, actor: str) -> int:
    if value not in ("p0", "p1") or actor not in ("p0", "p1"):
        raise FeatureSchemaError("seat must be p0 or p1")
    return 0 if value == actor else 1


def _seat_features(value: str | None, actor: str) -> list[float]:
    if value is None:
        return [0.0, 0.0, 1.0]
    rel = _seat(value, actor)
    return [1.0 if rel == 0 else 0.0, 1.0 if rel == 1 else 0.0, 0.0]


def _number(value: Any, scale: float = 1.0) -> float:
    if value is None:
        return 0.0
    if type(value) is bool:
        raise FeatureSchemaError("bool cannot be encoded as numeric magnitude")
    if not isinstance(value, (int, float)):
        raise FeatureSchemaError(f"expected numeric value, got {type(value).__name__}")
    return float(value) / scale


def _flag(value: Any) -> float:
    if type(value) is not bool:
        raise FeatureSchemaError(f"expected bool, got {type(value).__name__}")
    return 1.0 if value else 0.0


def _card_token(stable: dict[str, Any] | None) -> int:
    if not stable:
        return 0
    card_db_id = stable.get("card_db_id")
    if type(card_db_id) is not int:
        raise FeatureSchemaError("card_db_id must be an integer token")
    return card_db_id + 1


def _card_ref_features(stable: dict[str, Any] | None, actor: str) -> list[float]:
    if stable is None:
        return [0.0, 0.0, 1.0, 0.0, 0.0, 1.0] + [0.0] * len(ZONES)
    return _seat_features(stable["owner"], actor) + _seat_features(stable["controller"], actor) + _one_hot(stable["zone"], ZONES)


def _card_public_features(card: dict[str, Any], actor: str, order_index: int = 0, source_kind: str = "card") -> tuple[list[float], int]:
    stable = card["stable"]
    characteristics = card.get("characteristics", {})
    type_flags = characteristics.get("type_flags", {})
    keywords = characteristics.get("effective_keywords", {})
    source_flags = [1.0 if source_kind == kind else 0.0 for kind in ("card", "stack", "combat", "effect", "permission")]
    features = (
        _card_ref_features(stable, actor)
        + [
            _flag(card.get("tapped", False)),
            _flag(card.get("summoning_sick", False)),
            _number(card.get("damage", 0), 20.0),
            _number(card.get("counters", {}).get("plus1_plus1", 0), 10.0),
            _number(len(card.get("attachments", [])), 8.0),
            0.0 if card.get("plotted_turn") is None else 1.0,
        ]
        + [_flag(type_flags.get(name, False)) for name in ("land", "creature", "instant", "sorcery", "artifact", "enchantment")]
        + [
            _number(characteristics.get("base_power"), 20.0),
            _number(characteristics.get("base_toughness"), 20.0),
            _number(characteristics.get("effective_power"), 20.0),
            _number(characteristics.get("effective_toughness"), 20.0),
        ]
        + [
            _flag(keywords.get(name, False))
            for name in (
                "flying",
                "reach",
                "haste",
                "vigilance",
                "trample",
                "first_strike",
                "double_strike",
                "deathtouch",
                "menace",
                "defender",
            )
        ]
        + source_flags
        + [_number(order_index, 32.0)]
    )
    return features, _card_token(stable)


def _private_card_features(card: dict[str, Any], actor: str) -> tuple[list[float], int]:
    base = {
        "stable": card["stable"],
        "tapped": False,
        "summoning_sick": False,
        "damage": 0,
        "counters": {"plus1_plus1": 0},
        "attachments": [],
        "plotted_turn": None,
        "characteristics": {
            "type_flags": {name: False for name in ("land", "creature", "instant", "sorcery", "artifact", "enchantment")},
            "base_power": None,
            "base_toughness": None,
            "effective_power": None,
            "effective_toughness": None,
            "effective_keywords": {
                name: False
                for name in (
                    "flying",
                    "reach",
                    "haste",
                    "vigilance",
                    "trample",
                    "first_strike",
                    "double_strike",
                    "deathtouch",
                    "menace",
                    "defender",
                )
            },
        },
    }
    return _card_public_features(base, actor)


def _state_features(obs: dict[str, Any]) -> list[float]:
    actor = obs["acting_player"]
    p = obs["projection"]
    state: list[float] = [_number(p["turn"], 20.0)]
    state += _one_hot(p["phase"], PHASES)
    state += _seat_features(p["active_player"], actor)
    state += _seat_features(p["priority_player"], actor)
    for rel in (0, 1):
        seat = actor if rel == 0 else ("p1" if actor == "p0" else "p0")
        idx = 0 if seat == "p0" else 1
        state.append(_number(p["life_totals"][idx], 20.0))
        state += [_number(x, 10.0) for x in p["mana_pools"][idx]]
        state.append(_number(p["hand_counts"][idx], 16.0))
        state.append(_number(p["library_counts"][idx], 64.0))
        status = p["player_status"][idx]
        state += [
            _flag(status["has_lost"]),
            _number(status["lands_played_this_turn"], 4.0),
            _flag(status["drew_from_empty"]),
            _number(status["draws_this_turn"], 8.0),
        ]
    combat = p["combat"]
    state += [
        _flag(combat["attackers_declared"]),
        _flag(combat["blockers_declared"]),
        _number(len(combat["ordered_attackers"]), 16.0),
        _number(sum(len(pair[1]) for pair in combat["attacker_to_ordered_blockers"]), 32.0),
        _number(len(p["stack"]), 32.0),
        _number(len(p["continuous_effects"]), 32.0),
        _number(len(p["exile_play_permissions"]), 32.0),
    ]
    engine = p["engine_context"]
    state += [_flag(x) for x in engine["priority_passes"]]
    state += [
        _flag(engine["stack_nonempty"]),
        _flag(engine["stack_activity_since_priority_boundary"]),
        _flag(engine["mana_activity_since_priority_boundary"]),
    ]
    state += _seat_features(engine["last_mana_ability_activator_since_priority_boundary"], actor)
    state += _one_hot(engine["current_stage"], ENGINE_STAGES)
    state += [
        1.0 if engine["pending_cast"] else 0.0,
        1.0 if engine["pending_activation"] else 0.0,
        1.0 if engine["pending_discard"] else 0.0,
        1.0 if engine["pending_optional_cost"] else 0.0,
        1.0 if engine["pending_optional_cost_sacrifice"] else 0.0,
        _number(len(engine["pending_triggers"]), 16.0),
    ]
    surface = p["surface_context"]
    state += _one_hot(surface["current_stage"], SURFACE_STAGES)
    state += [_flag(x) for x in surface["combat_priority_spent"]]
    state += [
        _flag(surface["combat_priority_rearmed_by_stack_activity"]),
        _flag(surface["combat_priority_rearmed_by_mana_activity"]),
        _flag(surface["stack_grew_since_round_open"]),
        _flag(surface["mana_activity_since_round_open"]),
        0.0 if surface["stack_length_changed_since_observed"] is None else _flag(surface["stack_length_changed_since_observed"]),
        _flag(surface["mana_activity_since_last_stack_change"]),
        1.0 if surface["madness_cast_reprompt_source"] else 0.0,
        1.0 if surface["private_blockers"] else 0.0,
        1.0 if surface["private_discard"] else 0.0,
        1.0 if surface["private_optional_cost"] else 0.0,
    ]
    return state


def _group_id(name: str) -> int:
    return OBJECT_GROUPS.index(name)


def _append_object(rows: list[list[float]], tokens: list[int], groups: list[int], features: list[float], token: int, group: str) -> None:
    rows.append(features)
    tokens.append(token)
    groups.append(_group_id(group))


def _objects(obs: dict[str, Any]) -> tuple[list[list[float]], list[int], list[int]]:
    actor = obs["acting_player"]
    p = obs["projection"]
    rows: list[list[float]] = []
    tokens: list[int] = []
    groups: list[int] = []
    for card in obs["own_hand"]:
        features, token = _private_card_features(card, actor)
        _append_object(rows, tokens, groups, features, token, "self_hand")
    seat_order = [actor, "p1" if actor == "p0" else "p0"]
    for seat in seat_order:
        seat_idx = 0 if seat == "p0" else 1
        cards = p["battlefield"][seat_idx]
        group = "self_battlefield" if seat == actor else "opponent_battlefield"
        for card in cards:
            features, token = _card_public_features(card, actor)
            _append_object(rows, tokens, groups, features, token, group)
    for seat in seat_order:
        seat_idx = 0 if seat == "p0" else 1
        cards = p["graveyards"][seat_idx]
        group = "self_graveyard" if seat == actor else "opponent_graveyard"
        for card in cards:
            features, token = _card_public_features(card, actor)
            _append_object(rows, tokens, groups, features, token, group)
    for card in p["exile"]:
        features, token = _card_public_features(card, actor)
        _append_object(rows, tokens, groups, features, token, "exile")
    for i, item in enumerate(p["stack"]):
        features, token = _card_public_features({"stable": item["source"], "tapped": False, "summoning_sick": False, "damage": 0, "counters": {"plus1_plus1": 0}, "attachments": [], "plotted_turn": None, "characteristics": {}}, actor, i, "stack")
        features += _one_hot(item["stack_item_kind"], STACK_KINDS) + [_flag(item["is_flashback"]), _number(item["mode_chosen"], 8.0), _flag(item["madness_offer"]), _flag(item["kicked"])]
        _append_object(rows, tokens, groups, features, token, "stack")
    for i, ref in enumerate(p["combat"]["ordered_attackers"]):
        features, token = _card_public_features({"stable": ref, "tapped": False, "summoning_sick": False, "damage": 0, "counters": {"plus1_plus1": 0}, "attachments": [], "plotted_turn": None, "characteristics": {}}, actor, i, "combat")
        _append_object(rows, tokens, groups, features, token, "combat")
    for effect in p["continuous_effects"]:
        for ref in effect["affected_objects"]:
            features, token = _card_public_features({"stable": ref, "tapped": False, "summoning_sick": False, "damage": 0, "counters": {"plus1_plus1": 0}, "attachments": [], "plotted_turn": None, "characteristics": {}}, actor, 0, "effect")
            features += [_number(effect["layers"], 16.0), _number(effect["power_delta"], 20.0), _number(effect["toughness_delta"], 20.0), _flag(effect["grants_haste"])]
            _append_object(rows, tokens, groups, features, token, "effect")
    for permission in p["exile_play_permissions"]:
        features, token = _card_public_features({"stable": permission["object"], "tapped": False, "summoning_sick": False, "damage": 0, "counters": {"plus1_plus1": 0}, "attachments": [], "plotted_turn": None, "characteristics": {}}, actor, 0, "permission")
        features += _seat_features(permission["holder"], actor) + [1.0 if permission["play_or_cast"] == "play" else 0.0]
        _append_object(rows, tokens, groups, features, token, "permission")
    width = max((len(row) for row in rows), default=_object_feature_dim_probe())
    padded = [row + [0.0] * (width - len(row)) for row in rows]
    return padded, tokens, groups


def _object_feature_dim_probe() -> int:
    dummy = {
        "stable": {"arena_id": 0, "card_db_id": 0, "owner": "p0", "controller": "p0", "zone": "Hand", "zone_change_count": 0},
        "tapped": False,
        "summoning_sick": False,
        "damage": 0,
        "counters": {"plus1_plus1": 0},
        "attachments": [],
        "plotted_turn": None,
        "characteristics": {
            "type_flags": {name: False for name in ("land", "creature", "instant", "sorcery", "artifact", "enchantment")},
            "base_power": None,
            "base_toughness": None,
            "effective_power": None,
            "effective_toughness": None,
            "effective_keywords": {
                name: False
                for name in (
                    "flying",
                    "reach",
                    "haste",
                    "vigilance",
                    "trample",
                    "first_strike",
                    "double_strike",
                    "deathtouch",
                    "menace",
                    "defender",
                )
            },
        },
    }
    return len(_card_public_features(dummy, "p0")[0]) + 8


def _action_features(action: dict[str, Any], actor: str) -> tuple[list[float], list[int]]:
    assert_action_classified(action)
    semantic = action["semantic"]
    kind = semantic["action_kind"]
    refs: list[dict[str, Any]] = []
    for key in ("source", "candidate", "card", "attacker"):
        if key in semantic:
            refs.append(semantic[key])
    if "target" in semantic and semantic["target"]["target_kind"] == "object":
        refs.append(semantic["target"]["object"])
    for key in ("cards", "attackers", "blockers", "pending_sources"):
        refs.extend(semantic.get(key, []))
    card_ids = [_card_token(ref) for ref in refs[:MAX_ACTION_CARD_REFS]]
    card_ids += [0] * (MAX_ACTION_CARD_REFS - len(card_ids))
    features = _one_hot(kind, ACTION_KINDS)
    features += _seat_features(semantic.get("actor"), actor)
    features += _card_ref_features(semantic.get("source") or semantic.get("candidate") or semantic.get("card") or semantic.get("attacker"), actor)
    target = semantic.get("target")
    if target:
        features += _one_hot(target["target_kind"], TARGET_KINDS)
        features += _seat_features(target.get("player"), actor) if target["target_kind"] == "player" else _seat_features(None, actor)
    else:
        features += [0.0] * len(TARGET_KINDS) + _seat_features(None, actor)
    features += [
        _number(semantic.get("ability_index", 0), 8.0),
        _number(semantic.get("remaining", 0), 8.0),
        _number(semantic.get("mode_index", 0), 8.0),
        _number(semantic.get("mode_count", 0), 8.0),
        _flag(semantic.get("pay", False)),
        _flag(semantic.get("use_cost", False)),
        _flag(semantic.get("cast_it", False)),
        _number(len(semantic.get("cards", [])), 8.0),
        _number(len(semantic.get("attackers", [])), 16.0),
        _number(len(semantic.get("blockers", [])), 16.0),
        _number(len(semantic.get("pending_sources", [])), 16.0),
        _number(len(semantic.get("order", [])), 16.0),
    ]
    features += _one_hot(semantic.get("mode", CAST_MODES[0]), CAST_MODES)
    features += _one_hot(semantic.get("cost_kind", COST_KINDS[0]), COST_KINDS)
    features += _one_hot(semantic.get("choice", OPTIONAL_COST_CHOICES[0]), OPTIONAL_COST_CHOICES)
    return features, card_ids


def _schema(state_dim: int, object_dim: int, action_dim: int) -> FeatureSchema:
    return FeatureSchema(
        version="actor-relative-v2-python-1",
        state_dim=state_dim,
        object_feature_dim=object_dim,
        action_feature_dim=action_dim,
        object_group_count=len(OBJECT_GROUPS),
        max_action_card_refs=MAX_ACTION_CARD_REFS,
    )


def encode_decision(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> EncodedDecision:
    assert_observation_classified(observation)
    if not legal_actions:
        raise FeatureSchemaError("legal action list is empty")
    actor = observation["acting_player"]
    state = _state_features(observation)
    object_rows, object_tokens, object_groups = _objects(observation)
    action_rows: list[list[float]] = []
    action_tokens: list[list[int]] = []
    for action in legal_actions:
        row, tokens = _action_features(action, actor)
        action_rows.append(row)
        action_tokens.append(tokens)
    object_dim = max((len(row) for row in object_rows), default=_object_feature_dim_probe())
    action_dim = max(len(row) for row in action_rows)
    object_rows = [row + [0.0] * (object_dim - len(row)) for row in object_rows]
    action_rows = [row + [0.0] * (action_dim - len(row)) for row in action_rows]
    schema = _schema(len(state), object_dim, action_dim)
    if not object_rows:
        object_rows = [[0.0] * object_dim]
        object_tokens = [0]
        object_groups = [0]
    return EncodedDecision(
        schema=schema,
        state=torch.tensor(state, dtype=torch.float32),
        object_features=torch.tensor(object_rows, dtype=torch.float32),
        object_card_ids=torch.tensor(object_tokens, dtype=torch.long),
        object_groups=torch.tensor(object_groups, dtype=torch.long),
        action_features=torch.tensor(action_rows, dtype=torch.float32),
        action_card_ids=torch.tensor(action_tokens, dtype=torch.long),
    )


def every_action_variant_fixture(base_ref: dict[str, Any]) -> list[dict[str, Any]]:
    target_player = {"target_kind": "player", "player": "p1"}
    return [
        {"schema_version": 2, "selected_index": i, "stable_id": f"fixture-{i}", "display_text": f"text-{i}", "semantic": semantic}
        for i, semantic in enumerate(
            [
                {"action_kind": "pass", "actor": "p0"},
                {"action_kind": "play_land", "actor": "p0", "source": base_ref},
                {"action_kind": "cast_spell", "actor": "p0", "source": base_ref},
                {"action_kind": "activate_mana_ability", "actor": "p0", "source": base_ref},
                {"action_kind": "activate_ability", "actor": "p0", "source": base_ref, "ability_index": 1},
                {"action_kind": "plot_spell", "actor": "p0", "source": base_ref},
                {"action_kind": "choose_target", "actor": "p0", "source": base_ref, "remaining": 1, "target": target_player},
                {"action_kind": "choose_cost_target", "actor": "p0", "source": base_ref, "cost_kind": "SacrificeLands", "remaining": 1, "candidate": base_ref},
                {"action_kind": "choose_cast_mode", "actor": "p0", "source": base_ref, "mode": "Normal"},
                {"action_kind": "choose_kicker", "actor": "p0", "source": base_ref, "pay": True},
                {"action_kind": "choose_spell_mode", "actor": "p0", "source": base_ref, "mode_index": 0, "mode_count": 2},
                {"action_kind": "choose_optional_cost_use", "actor": "p0", "use_cost": True},
                {"action_kind": "choose_optional_cost_which", "actor": "p0", "choice": "Discard"},
                {"action_kind": "choose_madness_cast", "actor": "p0", "card": base_ref, "cast_it": True},
                {"action_kind": "discard", "actor": "p0", "cards": [base_ref]},
                {"action_kind": "declare_attackers", "actor": "p0", "attackers": [base_ref]},
                {"action_kind": "declare_blockers_for_attacker", "actor": "p0", "attacker": base_ref, "blockers": [base_ref]},
                {"action_kind": "order_triggers", "actor": "p0", "pending_sources": [base_ref], "order": [0]},
            ]
        )
    ]
