"""Path-aware v2 feature contract and actor-relative encoder."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

MODEL_INPUT = "model_input"
OPERATIONAL_ONLY = "operational_only"
FORBIDDEN = "forbidden"
CLASSIFICATIONS = (MODEL_INPUT, OPERATIONAL_ONLY, FORBIDDEN)

FEATURE_SCHEMA_VERSION = "actor-relative-v2-python-2"
FEATURE_REGISTRY_VERSION = "rust-observation-v2-action-v1-registry-2"
ENCODING_CONTRACT_VERSION = "actor-relative-hash-plus-role-refs-2"

STATE_HASH_DIM = 96
ACTION_HASH_DIM = 96

SEATS = ["p0", "p1"]
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
DISCARD_RESUME_STAGES = ["none", "finish_cast", "finish_activation", "finish_spell_resolution", "finish_optional_cost"]
TRIGGER_KINDS = ["triggered_ability", "madness_offer"]
EFFECT_DURATIONS = ["end_of_turn"]
PLAY_OR_CAST = ["play", "cast"]
EXPIRY_KINDS = ["end_of_turn", "until_holders_next_turn"]
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
    "attachment",
    "stack_target",
    "combat_block",
    "pending_context",
    "private_context",
]
ACTION_REF_ROLES = [
    "source",
    "candidate",
    "card",
    "attacker",
    "target_object",
    "cards",
    "attackers",
    "blockers",
    "pending_sources",
]

PLAYER_INDEXED_LISTS = {
    ("observation", "projection", "life_totals"),
    ("observation", "projection", "mana_pools"),
    ("observation", "projection", "hand_counts"),
    ("observation", "projection", "library_counts"),
    ("observation", "projection", "player_status"),
    ("observation", "projection", "battlefield"),
    ("observation", "projection", "graveyards"),
    ("observation", "projection", "engine_context", "priority_passes"),
    ("observation", "projection", "surface_context", "combat_priority_spent"),
}

UNORDERED_LISTS = {
    ("observation", "own_hand"),
    ("observation", "projection", "battlefield", "[]"),
    ("observation", "projection", "graveyards", "[]"),
    ("observation", "projection", "exile"),
    ("observation", "projection", "continuous_effects"),
    ("observation", "projection", "continuous_effects", "[]", "affected_objects"),
    ("observation", "projection", "exile_play_permissions"),
    ("observation", "projection", "surface_context", "private_discard", "chosen"),
    ("observation", "projection", "surface_context", "private_discard", "remaining_choices"),
    ("legal_action", "semantic", "cards"),
    ("legal_action", "semantic", "attackers"),
    ("legal_action", "semantic", "blockers"),
}


class FeatureSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    registry_version: str
    contract_digest: str
    encoding_digest: str
    state_dim: int
    object_feature_dim: int
    action_feature_dim: int
    object_group_count: int
    action_ref_feature_dim: int


@dataclass(frozen=True)
class EncodedDecision:
    schema: FeatureSchema
    state: torch.Tensor
    object_features: torch.Tensor
    object_card_ids: torch.Tensor
    object_groups: torch.Tensor
    action_features: torch.Tensor
    action_ref_features: torch.Tensor
    action_ref_card_ids: torch.Tensor
    action_ref_action_indices: torch.Tensor


class Spec:
    def validate(self, value: Any, path: tuple[str, ...]) -> None:
        raise NotImplementedError

    def describe(self) -> Any:
        raise NotImplementedError

    def leaf_specs(self, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], "ScalarSpec"]]:
        raise NotImplementedError


@dataclass(frozen=True)
class ScalarSpec(Spec):
    kind: str
    classification: str
    enum: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    nonempty: bool = False

    def __post_init__(self) -> None:
        if self.classification not in CLASSIFICATIONS:
            raise ValueError(self.classification)

    def validate(self, value: Any, path: tuple[str, ...]) -> None:
        ctx = ".".join(path)
        if self.kind == "int":
            if type(value) is not int:
                raise FeatureSchemaError(f"{ctx} must be an integer")
            if self.minimum is not None and value < self.minimum:
                raise FeatureSchemaError(f"{ctx} must be >= {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                raise FeatureSchemaError(f"{ctx} must be <= {self.maximum}")
            return
        if self.kind == "str":
            if type(value) is not str:
                raise FeatureSchemaError(f"{ctx} must be a string")
            if self.nonempty and not value:
                raise FeatureSchemaError(f"{ctx} must be nonempty")
            return
        if self.kind == "bool":
            if type(value) is not bool:
                raise FeatureSchemaError(f"{ctx} must be a bool")
            return
        if self.kind == "seat":
            if value not in SEATS:
                raise FeatureSchemaError(f"{ctx} must be p0 or p1")
            return
        if self.kind == "enum":
            if type(value) is not str or value not in self.enum:
                raise FeatureSchemaError(f"{ctx} must be one of {list(self.enum)}")
            return
        raise FeatureSchemaError(f"unknown scalar spec kind {self.kind!r} at {ctx}")

    def describe(self) -> Any:
        return {
            "type": "scalar",
            "kind": self.kind,
            "classification": self.classification,
            "enum": list(self.enum),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "nonempty": self.nonempty,
        }

    def leaf_specs(self, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], "ScalarSpec"]]:
        return [(path, self)]


@dataclass(frozen=True)
class OptionalSpec(Spec):
    item: Spec
    classification: str

    def validate(self, value: Any, path: tuple[str, ...]) -> None:
        if value is None:
            return
        self.item.validate(value, path)

    def describe(self) -> Any:
        return {"type": "optional", "classification": self.classification, "item": self.item.describe()}

    def leaf_specs(self, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], ScalarSpec]]:
        presence = ScalarSpec("bool", self.classification)
        return [(path + ("<present>",), presence)] + self.item.leaf_specs(path)


@dataclass(frozen=True)
class ListSpec(Spec):
    item: Spec
    length: int | None = None
    min_length: int | None = None

    def validate(self, value: Any, path: tuple[str, ...]) -> None:
        ctx = ".".join(path)
        if not isinstance(value, list):
            raise FeatureSchemaError(f"{ctx} must be a list")
        if self.length is not None and len(value) != self.length:
            raise FeatureSchemaError(f"{ctx} must have length {self.length}")
        if self.min_length is not None and len(value) < self.min_length:
            raise FeatureSchemaError(f"{ctx} must have at least {self.min_length} items")
        for item in value:
            self.item.validate(item, path + ("[]",))

    def describe(self) -> Any:
        return {"type": "list", "length": self.length, "min_length": self.min_length, "item": self.item.describe()}

    def leaf_specs(self, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], ScalarSpec]]:
        return self.item.leaf_specs(path + ("[]",))


@dataclass(frozen=True)
class TupleSpec(Spec):
    items: tuple[Spec, ...]

    def validate(self, value: Any, path: tuple[str, ...]) -> None:
        ctx = ".".join(path)
        if not isinstance(value, list):
            raise FeatureSchemaError(f"{ctx} must be a JSON tuple/list")
        if len(value) != len(self.items):
            raise FeatureSchemaError(f"{ctx} must have tuple length {len(self.items)}")
        for i, (child, child_value) in enumerate(zip(self.items, value)):
            child.validate(child_value, path + (str(i),))

    def describe(self) -> Any:
        return {"type": "tuple", "items": [item.describe() for item in self.items]}

    def leaf_specs(self, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], ScalarSpec]]:
        out: list[tuple[tuple[str, ...], ScalarSpec]] = []
        for i, item in enumerate(self.items):
            out.extend(item.leaf_specs(path + (str(i),)))
        return out


@dataclass(frozen=True)
class ObjectSpec(Spec):
    fields: dict[str, Spec]

    def validate(self, value: Any, path: tuple[str, ...]) -> None:
        ctx = ".".join(path)
        if not isinstance(value, dict):
            raise FeatureSchemaError(f"{ctx} must be an object")
        expected = set(self.fields)
        actual = set(value)
        if expected != actual:
            raise FeatureSchemaError(f"{ctx} fields mismatch: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
        for key, child in self.fields.items():
            child.validate(value[key], path + (key,))

    def describe(self) -> Any:
        return {"type": "object", "fields": {key: self.fields[key].describe() for key in sorted(self.fields)}}

    def leaf_specs(self, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], ScalarSpec]]:
        out: list[tuple[tuple[str, ...], ScalarSpec]] = []
        for key in sorted(self.fields):
            out.extend(self.fields[key].leaf_specs(path + (key,)))
        return out


@dataclass(frozen=True)
class VariantSpec(Spec):
    tag: str
    variants: dict[str, ObjectSpec]
    classification: str

    def validate(self, value: Any, path: tuple[str, ...]) -> None:
        ctx = ".".join(path)
        if not isinstance(value, dict):
            raise FeatureSchemaError(f"{ctx} must be an object")
        kind = value.get(self.tag)
        if type(kind) is not str or kind not in self.variants:
            raise FeatureSchemaError(f"{ctx}.{self.tag} has unsupported variant {kind!r}")
        self.variants[kind].validate(value, path)

    def describe(self) -> Any:
        return {
            "type": "variant",
            "tag": self.tag,
            "classification": self.classification,
            "variants": {key: self.variants[key].describe() for key in sorted(self.variants)},
        }

    def leaf_specs(self, path: tuple[str, ...]) -> list[tuple[tuple[str, ...], ScalarSpec]]:
        out = [(path + (self.tag,), ScalarSpec("enum", self.classification, tuple(sorted(self.variants))))]
        for key in sorted(self.variants):
            for child_path, spec in self.variants[key].leaf_specs(path):
                out.append((("<variant:" + key + ">",) + child_path, spec))
        return out


def I(classification: str, *, minimum: int | None = 0, maximum: int | None = None) -> ScalarSpec:
    return ScalarSpec("int", classification, minimum=minimum, maximum=maximum)


def S(classification: str, *, nonempty: bool = False) -> ScalarSpec:
    return ScalarSpec("str", classification, nonempty=nonempty)


def B(classification: str) -> ScalarSpec:
    return ScalarSpec("bool", classification)


def Seat(classification: str = MODEL_INPUT) -> ScalarSpec:
    return ScalarSpec("seat", classification)


def E(values: Iterable[str], classification: str = MODEL_INPUT) -> ScalarSpec:
    return ScalarSpec("enum", classification, tuple(values))


def Opt(item: Spec, classification: str = MODEL_INPUT) -> OptionalSpec:
    return OptionalSpec(item, classification)


U8 = 255
U16 = 65_535
U32 = 4_294_967_295
U64 = 18_446_744_073_709_551_615

CARD_STABLE_REF = ObjectSpec(
    {
        "arena_id": I(OPERATIONAL_ONLY, maximum=U32),
        "card_db_id": I(MODEL_INPUT, maximum=U16),
        "owner": Seat(),
        "controller": Seat(),
        "zone": E(ZONES),
        "zone_change_count": I(OPERATIONAL_ONLY, maximum=U32),
    }
)

COUNTERS = ObjectSpec({"plus1_plus1": I(MODEL_INPUT, minimum=-128, maximum=127)})
TYPE_FLAGS = ObjectSpec({name: B(MODEL_INPUT) for name in ("land", "creature", "instant", "sorcery", "artifact", "enchantment")})
KEYWORDS = ObjectSpec(
    {
        name: B(MODEL_INPUT)
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
    }
)
CHARACTERISTICS = ObjectSpec(
    {
        "type_flags": TYPE_FLAGS,
        "base_power": Opt(I(MODEL_INPUT, minimum=None)),
        "base_toughness": Opt(I(MODEL_INPUT, minimum=None)),
        "effective_power": Opt(I(MODEL_INPUT, minimum=None)),
        "effective_toughness": Opt(I(MODEL_INPUT, minimum=None)),
        "effective_keywords": KEYWORDS,
    }
)
CARD_PUBLIC = ObjectSpec(
    {
        "stable": CARD_STABLE_REF,
        "card_name": S(FORBIDDEN),
        "tapped": B(MODEL_INPUT),
        "summoning_sick": B(MODEL_INPUT),
        "damage": I(MODEL_INPUT, maximum=U16),
        "counters": COUNTERS,
        "attachments": ListSpec(I(MODEL_INPUT, maximum=U32)),
        "plotted_turn": Opt(I(MODEL_INPUT, maximum=U32)),
        "characteristics": CHARACTERISTICS,
    }
)
CARD_PRIVATE = ObjectSpec({"stable": CARD_STABLE_REF, "card_name": S(FORBIDDEN)})

TARGET_REF = VariantSpec(
    "target_kind",
    {
        "player": ObjectSpec({"target_kind": E(["player"]), "player": Seat()}),
        "object": ObjectSpec({"target_kind": E(["object"]), "object": CARD_STABLE_REF}),
    },
    MODEL_INPUT,
)
STACK_ITEM = ObjectSpec(
    {
        "stack_index": I(MODEL_INPUT, maximum=U32),
        "source": CARD_STABLE_REF,
        "controller": Seat(),
        "targets": ListSpec(TARGET_REF),
        "stack_item_kind": E(STACK_KINDS),
        "is_flashback": B(MODEL_INPUT),
        "mode_chosen": I(MODEL_INPUT, maximum=U8),
        "madness_offer": B(MODEL_INPUT),
        "kicked": B(MODEL_INPUT),
    }
)
PLAYER_STATUS = ObjectSpec(
    {
        "has_lost": B(MODEL_INPUT),
        "lands_played_this_turn": I(MODEL_INPUT, maximum=U8),
        "drew_from_empty": B(MODEL_INPUT),
        "draws_this_turn": I(MODEL_INPUT, maximum=U32),
    }
)
COMBAT = ObjectSpec(
    {
        "attackers_declared": B(MODEL_INPUT),
        "blockers_declared": B(MODEL_INPUT),
        "ordered_attackers": ListSpec(CARD_STABLE_REF),
        "attacker_to_ordered_blockers": ListSpec(TupleSpec((CARD_STABLE_REF, ListSpec(CARD_STABLE_REF)))),
    }
)
EFFECT = ObjectSpec(
    {
        "affected_objects": ListSpec(CARD_STABLE_REF),
        "layers": I(MODEL_INPUT, maximum=U8),
        "timestamp": I(OPERATIONAL_ONLY, maximum=U64),
        "duration": E(EFFECT_DURATIONS),
        "power_delta": I(MODEL_INPUT, minimum=None),
        "toughness_delta": I(MODEL_INPUT, minimum=None),
        "grants_haste": B(MODEL_INPUT),
    }
)
EXPIRY = VariantSpec(
    "expiry_kind",
    {
        "end_of_turn": ObjectSpec({"expiry_kind": E(["end_of_turn"])}),
        "until_holders_next_turn": ObjectSpec({"expiry_kind": E(["until_holders_next_turn"]), "holder_turn_started": B(MODEL_INPUT)}),
    },
    MODEL_INPUT,
)
PERMISSION = ObjectSpec(
    {
        "object": CARD_STABLE_REF,
        "holder": Seat(),
        "play_or_cast": E(PLAY_OR_CAST),
        "zone_change_generation": I(MODEL_INPUT, maximum=U32),
        "expiry": EXPIRY,
    }
)
PENDING_CAST = ObjectSpec(
    {
        "source": Opt(CARD_STABLE_REF),
        "controller": Seat(),
        "chosen_targets": ListSpec(TARGET_REF),
        "is_flashback": B(MODEL_INPUT),
        "cast_mode": Opt(E(CAST_MODES)),
        "additional_cost_discarded": Opt(ListSpec(CARD_STABLE_REF)),
        "mode_chosen": Opt(I(MODEL_INPUT, maximum=U8)),
        "origin_zone": E(ZONES),
        "sacrifice_chosen": ListSpec(CARD_STABLE_REF),
        "kicked": Opt(B(MODEL_INPUT)),
    }
)
PENDING_ACTIVATION = ObjectSpec(
    {
        "source": Opt(CARD_STABLE_REF),
        "controller": Seat(),
        "ability_index": I(MODEL_INPUT, maximum=U8),
        "chosen_targets": ListSpec(TARGET_REF),
        "cost_discard_paid": Opt(ListSpec(CARD_STABLE_REF)),
    }
)
PENDING_DISCARD = ObjectSpec(
    {
        "player": Seat(),
        "count": I(MODEL_INPUT, maximum=U32),
        "resume_stage": E(DISCARD_RESUME_STAGES),
        "resume_source": Opt(CARD_STABLE_REF),
    }
)
PENDING_OPTIONAL_COST = ObjectSpec(
    {
        "player": Seat(),
        "source": Opt(CARD_STABLE_REF),
        "discard_cards": I(MODEL_INPUT, maximum=U8),
        "sacrifice_lands": I(MODEL_INPUT, maximum=U8),
        "discard_payable": B(MODEL_INPUT),
        "sacrifice_payable": B(MODEL_INPUT),
        "spell_resume_source": Opt(CARD_STABLE_REF),
        "spell_resume_zone": Opt(E(ZONES)),
    }
)
PENDING_OPTIONAL_COST_SAC = ObjectSpec(
    {
        "player": Seat(),
        "source": Opt(CARD_STABLE_REF),
        "remaining": I(MODEL_INPUT, maximum=U8),
        "chosen": ListSpec(CARD_STABLE_REF),
        "spell_resume_source": Opt(CARD_STABLE_REF),
        "spell_resume_zone": Opt(E(ZONES)),
    }
)
PENDING_TRIGGER = ObjectSpec(
    {
        "source": Opt(CARD_STABLE_REF),
        "controller": Seat(),
        "trigger_kind": E(TRIGGER_KINDS),
        "kicked": B(MODEL_INPUT),
    }
)
ENGINE_CONTEXT = ObjectSpec(
    {
        "priority_passes": ListSpec(B(MODEL_INPUT), length=2),
        "stack_nonempty": B(MODEL_INPUT),
        "stack_activity_since_priority_boundary": B(MODEL_INPUT),
        "mana_activity_since_priority_boundary": B(MODEL_INPUT),
        "last_mana_ability_activator_since_priority_boundary": Opt(Seat()),
        "current_stage": E(ENGINE_STAGES),
        "pending_cast": Opt(PENDING_CAST),
        "pending_activation": Opt(PENDING_ACTIVATION),
        "pending_discard": Opt(PENDING_DISCARD),
        "pending_optional_cost": Opt(PENDING_OPTIONAL_COST),
        "pending_optional_cost_sacrifice": Opt(PENDING_OPTIONAL_COST_SAC),
        "pending_triggers": ListSpec(PENDING_TRIGGER),
    }
)
PRIVATE_BLOCKERS = ObjectSpec(
    {
        "current_attacker": Opt(CARD_STABLE_REF),
        "accumulated": ListSpec(TupleSpec((CARD_STABLE_REF, CARD_STABLE_REF))),
        "remaining": ListSpec(TupleSpec((CARD_STABLE_REF, ListSpec(CARD_STABLE_REF)))),
    }
)
PRIVATE_DISCARD = ObjectSpec(
    {
        "chosen": ListSpec(CARD_STABLE_REF),
        "remaining_choices": ListSpec(CARD_STABLE_REF),
        "remaining_needed": I(MODEL_INPUT, maximum=U32),
    }
)
PRIVATE_OPTIONAL_COST = ObjectSpec(
    {
        "discard_payable": B(MODEL_INPUT),
        "sacrifice_payable": B(MODEL_INPUT),
        "stage": E(SURFACE_STAGES),
    }
)
SURFACE_CONTEXT = ObjectSpec(
    {
        "current_stage": E(SURFACE_STAGES),
        "combat_priority_spent": ListSpec(B(MODEL_INPUT), length=2),
        "combat_priority_rearmed_by_stack_activity": B(MODEL_INPUT),
        "combat_priority_rearmed_by_mana_activity": B(MODEL_INPUT),
        "stack_grew_since_round_open": B(MODEL_INPUT),
        "mana_activity_since_round_open": B(MODEL_INPUT),
        "stack_length_changed_since_observed": Opt(B(MODEL_INPUT)),
        "mana_activity_since_last_stack_change": B(MODEL_INPUT),
        "madness_cast_reprompt_source": Opt(CARD_STABLE_REF),
        "private_blockers": Opt(PRIVATE_BLOCKERS),
        "private_discard": Opt(PRIVATE_DISCARD),
        "private_optional_cost": Opt(PRIVATE_OPTIONAL_COST),
    }
)
PROJECTION = ObjectSpec(
    {
        "turn": I(MODEL_INPUT, maximum=U32),
        "phase": E(PHASES),
        "active_player": Seat(),
        "priority_player": Seat(),
        "life_totals": ListSpec(I(MODEL_INPUT, minimum=None), length=2),
        "mana_pools": ListSpec(ListSpec(I(MODEL_INPUT, maximum=U8), length=6), length=2),
        "hand_counts": ListSpec(I(MODEL_INPUT), length=2),
        "library_counts": ListSpec(I(MODEL_INPUT), length=2),
        "player_status": ListSpec(PLAYER_STATUS, length=2),
        "battlefield": ListSpec(ListSpec(CARD_PUBLIC), length=2),
        "graveyards": ListSpec(ListSpec(CARD_PUBLIC), length=2),
        "exile": ListSpec(CARD_PUBLIC),
        "stack": ListSpec(STACK_ITEM),
        "combat": COMBAT,
        "continuous_effects": ListSpec(EFFECT),
        "exile_play_permissions": ListSpec(PERMISSION),
        "engine_context": ENGINE_CONTEXT,
        "surface_context": SURFACE_CONTEXT,
    }
)
OBSERVATION_SPEC = ObjectSpec(
    {
        "schema_version": I(OPERATIONAL_ONLY, maximum=U32),
        "kernel_version": S(OPERATIONAL_ONLY),
        "surface_version": I(OPERATIONAL_ONLY, maximum=U32),
        "card_db_hash": I(OPERATIONAL_ONLY, maximum=U64),
        "acting_player": Seat(),
        "step_index": I(OPERATIONAL_ONLY, maximum=U64),
        "projection": PROJECTION,
        "own_hand": ListSpec(CARD_PRIVATE),
        "visible_projection_hash": I(FORBIDDEN, maximum=U64),
    }
)

ACTION_VARIANTS = {
    "pass": ObjectSpec({"action_kind": E(["pass"]), "actor": Seat()}),
    "play_land": ObjectSpec({"action_kind": E(["play_land"]), "actor": Seat(), "source": CARD_STABLE_REF}),
    "cast_spell": ObjectSpec({"action_kind": E(["cast_spell"]), "actor": Seat(), "source": CARD_STABLE_REF}),
    "activate_mana_ability": ObjectSpec({"action_kind": E(["activate_mana_ability"]), "actor": Seat(), "source": CARD_STABLE_REF}),
    "activate_ability": ObjectSpec({"action_kind": E(["activate_ability"]), "actor": Seat(), "source": CARD_STABLE_REF, "ability_index": I(MODEL_INPUT, maximum=U8)}),
    "plot_spell": ObjectSpec({"action_kind": E(["plot_spell"]), "actor": Seat(), "source": CARD_STABLE_REF}),
    "choose_target": ObjectSpec({"action_kind": E(["choose_target"]), "actor": Seat(), "source": CARD_STABLE_REF, "remaining": I(MODEL_INPUT, maximum=U8), "target": TARGET_REF}),
    "choose_cost_target": ObjectSpec({"action_kind": E(["choose_cost_target"]), "actor": Seat(), "source": CARD_STABLE_REF, "cost_kind": E(COST_KINDS), "remaining": I(MODEL_INPUT, maximum=U8), "candidate": CARD_STABLE_REF}),
    "choose_cast_mode": ObjectSpec({"action_kind": E(["choose_cast_mode"]), "actor": Seat(), "source": CARD_STABLE_REF, "mode": E(CAST_MODES)}),
    "choose_kicker": ObjectSpec({"action_kind": E(["choose_kicker"]), "actor": Seat(), "source": CARD_STABLE_REF, "pay": B(MODEL_INPUT)}),
    "choose_spell_mode": ObjectSpec({"action_kind": E(["choose_spell_mode"]), "actor": Seat(), "source": CARD_STABLE_REF, "mode_index": I(MODEL_INPUT, maximum=U8), "mode_count": I(MODEL_INPUT, maximum=U8)}),
    "choose_optional_cost_use": ObjectSpec({"action_kind": E(["choose_optional_cost_use"]), "actor": Seat(), "use_cost": B(MODEL_INPUT)}),
    "choose_optional_cost_which": ObjectSpec({"action_kind": E(["choose_optional_cost_which"]), "actor": Seat(), "choice": E(OPTIONAL_COST_CHOICES)}),
    "choose_madness_cast": ObjectSpec({"action_kind": E(["choose_madness_cast"]), "actor": Seat(), "card": CARD_STABLE_REF, "cast_it": B(MODEL_INPUT)}),
    "discard": ObjectSpec({"action_kind": E(["discard"]), "actor": Seat(), "cards": ListSpec(CARD_STABLE_REF)}),
    "declare_attackers": ObjectSpec({"action_kind": E(["declare_attackers"]), "actor": Seat(), "attackers": ListSpec(CARD_STABLE_REF)}),
    "declare_blockers_for_attacker": ObjectSpec({"action_kind": E(["declare_blockers_for_attacker"]), "actor": Seat(), "attacker": CARD_STABLE_REF, "blockers": ListSpec(CARD_STABLE_REF)}),
    "order_triggers": ObjectSpec({"action_kind": E(["order_triggers"]), "actor": Seat(), "pending_sources": ListSpec(CARD_STABLE_REF), "order": ListSpec(I(MODEL_INPUT))}),
}
ACTION_SEMANTIC = VariantSpec("action_kind", ACTION_VARIANTS, MODEL_INPUT)
LEGAL_ACTION_SPEC = ObjectSpec(
    {
        "schema_version": I(OPERATIONAL_ONLY, maximum=U32),
        "selected_index": I(OPERATIONAL_ONLY, maximum=U32),
        "stable_id": S(FORBIDDEN),
        "semantic": ACTION_SEMANTIC,
        "display_text": Opt(S(FORBIDDEN), FORBIDDEN),
    }
)

_OMIT = object()


def _contract_payload() -> dict[str, Any]:
    return {
        "registry_version": FEATURE_REGISTRY_VERSION,
        "observation": OBSERVATION_SPEC.describe(),
        "legal_action": LEGAL_ACTION_SPEC.describe(),
    }


def _encoding_payload() -> dict[str, Any]:
    return {
        "encoding_version": ENCODING_CONTRACT_VERSION,
        "state_hash_dim": STATE_HASH_DIM,
        "action_hash_dim": ACTION_HASH_DIM,
        "player_indexed_lists": [".".join(path) for path in sorted(PLAYER_INDEXED_LISTS)],
        "unordered_lists": [".".join(path) for path in sorted(UNORDERED_LISTS)],
        "object_groups": OBJECT_GROUPS,
        "action_ref_roles": ACTION_REF_ROLES,
        "action_kinds": ACTION_KINDS,
        "stack_kinds": STACK_KINDS,
        "surface_stages": SURFACE_STAGES,
        "engine_stages": ENGINE_STAGES,
    }


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_contract_fingerprint() -> str:
    return _sha256_json(_contract_payload())


def encoding_contract_fingerprint() -> str:
    return _sha256_json(_encoding_payload())


def model_contract_fingerprint(schema: FeatureSchema) -> str:
    return _sha256_json({"model_contract_version": "kernel-policy-value-net-2", "feature_schema": schema.__dict__})


def classification_registry() -> dict[str, str]:
    out: dict[str, str] = {}
    for root, spec in (("observation", OBSERVATION_SPEC), ("legal_action", LEGAL_ACTION_SPEC)):
        for path, scalar in spec.leaf_specs((root,)):
            out[".".join(path)] = scalar.classification
    return out


def _normalize_path(path: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for part in path:
        normalized.append("[]" if part.isdigit() else part)
    return tuple(normalized)


def iter_classified_leaves(value: Any, root: str) -> list[tuple[tuple[str, ...], str]]:
    spec = OBSERVATION_SPEC if root == "observation" else LEGAL_ACTION_SPEC
    spec.validate(value, (root,))
    out: list[tuple[tuple[str, ...], str]] = []

    def walk(v: Any, s: Spec, path: tuple[str, ...]) -> None:
        if isinstance(s, ScalarSpec):
            out.append((_normalize_path(path), s.classification))
        elif isinstance(s, OptionalSpec):
            out.append((_normalize_path(path + ("<present>",)), s.classification))
            if v is not None:
                walk(v, s.item, path)
        elif isinstance(s, ListSpec):
            for i, child in enumerate(v):
                walk(child, s.item, path + (str(i),))
        elif isinstance(s, TupleSpec):
            for i, (child_spec, child) in enumerate(zip(s.items, v)):
                walk(child, child_spec, path + (str(i),))
        elif isinstance(s, ObjectSpec):
            for key, child_spec in s.fields.items():
                walk(v[key], child_spec, path + (key,))
        elif isinstance(s, VariantSpec):
            walk(v, s.variants[v[s.tag]], path)
        else:
            raise TypeError(s)

    walk(value, spec, (root,))
    return out


def _seat(value: str, actor: str) -> int:
    if value not in SEATS or actor not in SEATS:
        raise FeatureSchemaError("seat must be p0 or p1")
    return 0 if value == actor else 1


def _relative_seat(value: str, actor: str) -> str:
    return "self" if _seat(value, actor) == 0 else "opponent"


def _seat_features(value: str | None, actor: str) -> list[float]:
    if value is None:
        return [0.0, 0.0, 1.0]
    rel = _seat(value, actor)
    return [1.0 if rel == 0 else 0.0, 1.0 if rel == 1 else 0.0, 0.0]


def _one_hot(value: str, choices: list[str]) -> list[float]:
    if value not in choices:
        raise FeatureSchemaError(f"unknown categorical value {value!r}; choices={choices}")
    return [1.0 if value == choice else 0.0 for choice in choices]


def _number(value: Any, scale: float = 1.0) -> float:
    if value is None:
        return 0.0
    if type(value) is bool:
        raise FeatureSchemaError("bool cannot be encoded as numeric magnitude")
    if not isinstance(value, (int, float)):
        raise FeatureSchemaError(f"expected numeric value, got {type(value).__name__}")
    result = float(value) / scale
    if not math.isfinite(result):
        raise FeatureSchemaError("numeric feature is not finite")
    return result


def _flag(value: Any) -> float:
    if type(value) is not bool:
        raise FeatureSchemaError(f"expected bool, got {type(value).__name__}")
    return 1.0 if value else 0.0


def _card_token(stable: dict[str, Any] | None) -> int:
    if not stable:
        return 0
    card_db_id = stable.get("card_db_id")
    if type(card_db_id) is not int or card_db_id < 0:
        raise FeatureSchemaError("card_db_id must be a nonnegative integer token")
    return card_db_id + 1


def _card_ref_features(stable: dict[str, Any] | None, actor: str) -> list[float]:
    if stable is None:
        return [0.0, 0.0, 1.0, 0.0, 0.0, 1.0] + [0.0] * len(ZONES)
    return _seat_features(stable["owner"], actor) + _seat_features(stable["controller"], actor) + _one_hot(stable["zone"], ZONES)


class _CanonicalContext:
    def __init__(self, actor: str, observation: dict[str, Any] | None = None) -> None:
        self.actor = actor
        self._arena_order: dict[int, int] = {}
        self._arena_keys: dict[int, Any] = {}
        if observation is not None:
            for ref in _iter_card_refs(observation):
                arena = ref.get("arena_id")
                if type(arena) is int and arena not in self._arena_order:
                    self._arena_order[arena] = len(self._arena_order) + 1
                    self._arena_keys[arena] = {
                        "card_db_id": ref.get("card_db_id"),
                        "owner": _relative_seat(ref.get("owner"), actor) if ref.get("owner") in SEATS else ref.get("owner"),
                        "controller": _relative_seat(ref.get("controller"), actor) if ref.get("controller") in SEATS else ref.get("controller"),
                        "zone": ref.get("zone"),
                    }

    def arena_id(self, raw: int) -> int:
        if raw not in self._arena_order:
            self._arena_order[raw] = len(self._arena_order) + 1
        return self._arena_order[raw]

    def arena_key(self, raw: int) -> Any:
        return self._arena_keys.get(raw, {"unresolved_attachment": True})


def _iter_card_refs(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"arena_id", "card_db_id", "owner", "controller", "zone", "zone_change_count"}.issubset(value):
            yield value
        for child in value.values():
            yield from _iter_card_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_card_refs(child)


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_model_value(value: Any, spec: Spec, path: tuple[str, ...], ctx: _CanonicalContext) -> Any:
    normalized = _normalize_path(path)
    if isinstance(spec, ScalarSpec):
        if spec.classification != MODEL_INPUT:
            return _OMIT
        if spec.kind == "seat":
            return _relative_seat(value, ctx.actor)
        if spec.kind == "int" and "attachments" in path:
            return ctx.arena_key(value)
        return value
    if isinstance(spec, OptionalSpec):
        if value is None:
            return None if spec.classification == MODEL_INPUT else _OMIT
        child = _canonical_model_value(value, spec.item, path, ctx)
        return child
    if isinstance(spec, ListSpec):
        values = value
        if normalized in PLAYER_INDEXED_LISTS:
            actor_index = 0 if ctx.actor == "p0" else 1
            values = [value[actor_index], value[1 - actor_index]]
        canonical: list[Any] = []
        for child in values:
            child_value = _canonical_model_value(child, spec.item, path + ("[]",), ctx)
            if child_value is not _OMIT:
                canonical.append(child_value)
        if normalized in UNORDERED_LISTS:
            canonical.sort(key=_sort_key)
        return canonical
    if isinstance(spec, TupleSpec):
        items: list[Any] = []
        for i, (child_spec, child) in enumerate(zip(spec.items, value)):
            child_value = _canonical_model_value(child, child_spec, path + (str(i),), ctx)
            if child_value is not _OMIT:
                items.append(child_value)
        return items
    if isinstance(spec, ObjectSpec):
        out: dict[str, Any] = {}
        for key in sorted(spec.fields):
            child_value = _canonical_model_value(value[key], spec.fields[key], path + (key,), ctx)
            if child_value is not _OMIT:
                out[key] = child_value
        return out if out else _OMIT
    if isinstance(spec, VariantSpec):
        return _canonical_model_value(value, spec.variants[value[spec.tag]], path, ctx)
    raise TypeError(spec)


def _digest_features(namespace: str, canonical: Any, dims: int) -> list[float]:
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    out: list[float] = []
    counter = 0
    while len(out) < dims:
        digest = hashlib.sha512(namespace.encode("utf-8") + counter.to_bytes(4, "little") + payload).digest()
        counter += 1
        for i in range(0, len(digest), 4):
            chunk = int.from_bytes(digest[i : i + 4], "little", signed=False)
            out.append((float(chunk) / float(0xFFFF_FFFF)) * 2.0 - 1.0)
            if len(out) == dims:
                break
    return out


def _card_public_features(card: dict[str, Any], actor: str, order_index: int = 0, source_kind: str = "card") -> tuple[list[float], int]:
    stable = card["stable"]
    characteristics = card.get("characteristics", {})
    type_flags = characteristics.get("type_flags", {})
    keywords = characteristics.get("effective_keywords", {})
    source_flags = [1.0 if source_kind == kind else 0.0 for kind in ("card", "stack", "combat", "effect", "permission", "attachment", "target", "pending", "private")]
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
        + [_number(order_index, 64.0)]
    )
    return features, _card_token(stable)


def _blank_public_from_ref(stable: dict[str, Any], source_kind: str = "card") -> dict[str, Any]:
    return {
        "stable": stable,
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
        "_source_kind": source_kind,
    }


def _private_card_features(card: dict[str, Any], actor: str) -> tuple[list[float], int]:
    return _card_public_features(_blank_public_from_ref(card["stable"]), actor)


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
    state += [_flag(engine["priority_passes"][0 if actor == "p0" else 1]), _flag(engine["priority_passes"][1 if actor == "p0" else 0])]
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
    state += [_flag(surface["combat_priority_spent"][0 if actor == "p0" else 1]), _flag(surface["combat_priority_spent"][1 if actor == "p0" else 0])]
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
    canonical = _canonical_model_value(obs, OBSERVATION_SPEC, ("observation",), _CanonicalContext(actor, obs))
    state += _digest_features("observation-state", canonical, STATE_HASH_DIM)
    return state


def _group_id(name: str) -> int:
    return OBJECT_GROUPS.index(name)


def _append_object(rows: list[list[float]], tokens: list[int], groups: list[int], features: list[float], token: int, group: str) -> None:
    rows.append(features)
    tokens.append(token)
    groups.append(_group_id(group))


def _arena_public_map(obs: dict[str, Any]) -> dict[int, dict[str, Any]]:
    p = obs["projection"]
    out: dict[int, dict[str, Any]] = {}
    for zone_cards in p["battlefield"] + p["graveyards"] + [p["exile"]]:
        for card in zone_cards:
            out[card["stable"]["arena_id"]] = card
    for card in obs["own_hand"]:
        out[card["stable"]["arena_id"]] = _blank_public_from_ref(card["stable"])
    return out


def _append_ref_object(rows: list[list[float]], tokens: list[int], groups: list[int], ref: dict[str, Any], actor: str, group: str, order: int, source_kind: str, extra: list[float] | None = None) -> None:
    features, token = _card_public_features(_blank_public_from_ref(ref, source_kind), actor, order, source_kind)
    if extra:
        features += extra
    _append_object(rows, tokens, groups, features, token, group)


def _pending_ref_rows(rows: list[list[float]], tokens: list[int], groups: list[int], value: Any, actor: str, role_index: int = 0) -> None:
    if isinstance(value, dict):
        if {"arena_id", "card_db_id", "owner", "controller", "zone", "zone_change_count"}.issubset(value):
            _append_ref_object(rows, tokens, groups, value, actor, "pending_context", role_index, "pending", [_number(role_index, 32.0)])
            return
        for child in value.values():
            _pending_ref_rows(rows, tokens, groups, child, actor, role_index + 1)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _pending_ref_rows(rows, tokens, groups, child, actor, role_index + i)


def _objects(obs: dict[str, Any]) -> tuple[list[list[float]], list[int], list[int]]:
    actor = obs["acting_player"]
    p = obs["projection"]
    rows: list[list[float]] = []
    tokens: list[int] = []
    groups: list[int] = []
    public_by_arena = _arena_public_map(obs)
    for card in sorted(obs["own_hand"], key=lambda c: (c["stable"]["card_db_id"], c["stable"]["owner"], c["stable"]["zone"], c["stable"]["arena_id"])):
        features, token = _private_card_features(card, actor)
        _append_object(rows, tokens, groups, features, token, "self_hand")
    seat_order = [actor, "p1" if actor == "p0" else "p0"]
    for seat in seat_order:
        seat_idx = 0 if seat == "p0" else 1
        cards = sorted(p["battlefield"][seat_idx], key=lambda c: (c["stable"]["card_db_id"], c["stable"]["controller"], c["stable"]["arena_id"]))
        group = "self_battlefield" if seat == actor else "opponent_battlefield"
        for card in cards:
            features, token = _card_public_features(card, actor)
            _append_object(rows, tokens, groups, features, token, group)
            for attach_order, attachment_arena in enumerate(sorted(card["attachments"])):
                attached = public_by_arena.get(attachment_arena)
                if attached is not None:
                    afeatures, atoken = _card_public_features(attached, actor, attach_order, "attachment")
                    afeatures += _card_ref_features(card["stable"], actor)
                    _append_object(rows, tokens, groups, afeatures, atoken, "attachment")
    for seat in seat_order:
        seat_idx = 0 if seat == "p0" else 1
        cards = sorted(p["graveyards"][seat_idx], key=lambda c: (c["stable"]["card_db_id"], c["stable"]["controller"], c["stable"]["arena_id"]))
        group = "self_graveyard" if seat == actor else "opponent_graveyard"
        for card in cards:
            features, token = _card_public_features(card, actor)
            _append_object(rows, tokens, groups, features, token, group)
    for card in sorted(p["exile"], key=lambda c: (c["stable"]["card_db_id"], c["stable"]["controller"], c["stable"]["arena_id"])):
        features, token = _card_public_features(card, actor)
        _append_object(rows, tokens, groups, features, token, "exile")
    for i, item in enumerate(p["stack"]):
        features, token = _card_public_features(_blank_public_from_ref(item["source"], "stack"), actor, i, "stack")
        features += (
            _seat_features(item["controller"], actor)
            + _one_hot(item["stack_item_kind"], STACK_KINDS)
            + [_flag(item["is_flashback"]), _number(item["mode_chosen"], 8.0), _flag(item["madness_offer"]), _flag(item["kicked"])]
        )
        _append_object(rows, tokens, groups, features, token, "stack")
        for target_index, target in enumerate(item["targets"]):
            if target["target_kind"] == "object":
                extra = _card_ref_features(item["source"], actor) + _one_hot("object", TARGET_KINDS) + [_number(target_index, 16.0)]
                _append_ref_object(rows, tokens, groups, target["object"], actor, "stack_target", target_index, "target", extra)
            else:
                extra = _card_ref_features(item["source"], actor) + _one_hot("player", TARGET_KINDS) + _seat_features(target["player"], actor) + [_number(target_index, 16.0)]
                _append_object(rows, tokens, groups, [0.0] * _object_feature_dim_probe() + extra, 0, "stack_target")
    for i, ref in enumerate(p["combat"]["ordered_attackers"]):
        _append_ref_object(rows, tokens, groups, ref, actor, "combat", i, "combat", [_number(i, 16.0)])
    for attacker_order, pair in enumerate(p["combat"]["attacker_to_ordered_blockers"]):
        attacker, blockers = pair
        for blocker_order, blocker in enumerate(blockers):
            extra = _card_ref_features(attacker, actor) + [_number(attacker_order, 16.0), _number(blocker_order, 16.0)]
            _append_ref_object(rows, tokens, groups, blocker, actor, "combat_block", blocker_order, "combat", extra)
    for effect in p["continuous_effects"]:
        for ref in effect["affected_objects"]:
            extra = [_number(effect["layers"], 16.0), _number(effect["power_delta"], 20.0), _number(effect["toughness_delta"], 20.0), _flag(effect["grants_haste"])] + _one_hot(effect["duration"], EFFECT_DURATIONS)
            _append_ref_object(rows, tokens, groups, ref, actor, "effect", 0, "effect", extra)
    for permission in p["exile_play_permissions"]:
        extra = _seat_features(permission["holder"], actor) + _one_hot(permission["play_or_cast"], PLAY_OR_CAST) + [_number(permission["zone_change_generation"], 8.0)]
        expiry = permission["expiry"]
        extra += _one_hot(expiry["expiry_kind"], EXPIRY_KINDS)
        extra += [_flag(expiry.get("holder_turn_started", False))]
        _append_ref_object(rows, tokens, groups, permission["object"], actor, "permission", 0, "permission", extra)
    engine = p["engine_context"]
    for key in ("pending_cast", "pending_activation", "pending_discard", "pending_optional_cost", "pending_optional_cost_sacrifice"):
        if engine[key] is not None:
            _pending_ref_rows(rows, tokens, groups, engine[key], actor)
    _pending_ref_rows(rows, tokens, groups, engine["pending_triggers"], actor)
    surface = p["surface_context"]
    for key in ("madness_cast_reprompt_source", "private_blockers", "private_discard"):
        if surface[key] is not None:
            before = len(rows)
            _pending_ref_rows(rows, tokens, groups, surface[key], actor)
            for j in range(before, len(rows)):
                groups[j] = _group_id("private_context")
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
    return len(_card_public_features(dummy, "p0")[0]) + 24


def _action_kind(action: dict[str, Any]) -> str:
    return action["semantic"]["action_kind"]


def _semantic_actor(action: dict[str, Any]) -> str:
    return action["semantic"]["actor"]


def _action_card_refs(semantic: dict[str, Any]) -> list[tuple[str, int, dict[str, Any], int]]:
    refs: list[tuple[str, int, dict[str, Any], int]] = []
    for role in ("source", "candidate", "card", "attacker"):
        if role in semantic:
            refs.append((role, 0, semantic[role], 0))
    if "target" in semantic and semantic["target"]["target_kind"] == "object":
        refs.append(("target_object", 0, semantic["target"]["object"], 0))
    for role in ("cards", "attackers", "blockers"):
        for i, ref in enumerate(sorted(semantic.get(role, []), key=lambda r: (r["card_db_id"], r["owner"], r["controller"], r["zone"], r["arena_id"]))):
            refs.append((role, i, ref, 0))
    if semantic.get("action_kind") == "order_triggers":
        order = semantic["order"]
        for i, ref in enumerate(semantic["pending_sources"]):
            refs.append(("pending_sources", i, ref, int(order[i])))
    return refs


def _action_ref_row(role: str, order_index: int, ref: dict[str, Any], actor: str, associated_order: int) -> tuple[list[float], int]:
    features = _one_hot(role, ACTION_REF_ROLES) + _card_ref_features(ref, actor) + [_number(order_index, 32.0), _number(associated_order, 32.0)]
    return features, _card_token(ref)


def _action_features(action: dict[str, Any], actor: str) -> tuple[list[float], list[list[float]], list[int]]:
    assert_action_classified(action)
    semantic = action["semantic"]
    kind = semantic["action_kind"]
    features = _one_hot(kind, ACTION_KINDS)
    features += _seat_features(semantic.get("actor"), actor)
    source_like = semantic.get("source") or semantic.get("candidate") or semantic.get("card") or semantic.get("attacker")
    features += _card_ref_features(source_like, actor)
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
    canonical = _canonical_model_value(action, LEGAL_ACTION_SPEC, ("legal_action",), _CanonicalContext(actor))
    features += _digest_features("legal-action", canonical, ACTION_HASH_DIM)
    ref_rows: list[list[float]] = []
    ref_tokens: list[int] = []
    for role, order_index, ref, associated_order in _action_card_refs(semantic):
        row, token = _action_ref_row(role, order_index, ref, actor, associated_order)
        ref_rows.append(row)
        ref_tokens.append(token)
    return features, ref_rows, ref_tokens


def _schema(state_dim: int, object_dim: int, action_dim: int, action_ref_dim: int) -> FeatureSchema:
    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        registry_version=FEATURE_REGISTRY_VERSION,
        contract_digest=feature_contract_fingerprint(),
        encoding_digest=encoding_contract_fingerprint(),
        state_dim=state_dim,
        object_feature_dim=object_dim,
        action_feature_dim=action_dim,
        object_group_count=len(OBJECT_GROUPS),
        action_ref_feature_dim=action_ref_dim,
    )


def _validate_order_trigger_semantic(semantic: dict[str, Any]) -> None:
    if semantic["action_kind"] != "order_triggers":
        return
    pending = semantic["pending_sources"]
    order = semantic["order"]
    if len(order) != len(pending):
        raise FeatureSchemaError("order_triggers.order length must match pending_sources")
    if sorted(order) != list(range(len(order))):
        raise FeatureSchemaError("order_triggers.order must be a permutation of pending source indexes")


def assert_observation_classified(observation: dict[str, Any]) -> None:
    OBSERVATION_SPEC.validate(observation, ("observation",))


def assert_action_classified(action: dict[str, Any]) -> None:
    LEGAL_ACTION_SPEC.validate(action, ("legal_action",))
    semantic = action["semantic"]
    if semantic["action_kind"] == "choose_spell_mode" and semantic["mode_index"] >= semantic["mode_count"]:
        raise FeatureSchemaError("choose_spell_mode.mode_index must be < mode_count")
    if semantic["action_kind"] == "choose_spell_mode" and semantic["mode_count"] <= 0:
        raise FeatureSchemaError("choose_spell_mode.mode_count must be positive")
    _validate_order_trigger_semantic(semantic)


def validate_legal_actions_contract(actions: Any, acting_player: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(actions, list):
        raise FeatureSchemaError("legal_actions must be a list")
    if not actions:
        raise FeatureSchemaError("decision has no legal actions")
    seen_stable: set[str] = set()
    for i, action in enumerate(actions):
        assert_action_classified(action)
        if action["schema_version"] != 2:
            raise FeatureSchemaError("legal action schema mismatch")
        if action["selected_index"] != i:
            raise FeatureSchemaError("legal action selected_index is not contiguous")
        if action["stable_id"] in seen_stable:
            raise FeatureSchemaError("duplicate legal action stable_id")
        seen_stable.add(action["stable_id"])
        if acting_player is not None and _semantic_actor(action) != acting_player:
            raise FeatureSchemaError("legal action actor does not match acting_player")
    return actions


def encode_decision(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> EncodedDecision:
    assert_observation_classified(observation)
    if observation["schema_version"] != 2:
        raise FeatureSchemaError("observation schema mismatch")
    validate_legal_actions_contract(legal_actions, observation["acting_player"])
    actor = observation["acting_player"]
    state = _state_features(observation)
    object_rows, object_tokens, object_groups = _objects(observation)
    action_rows: list[list[float]] = []
    action_ref_rows: list[list[float]] = []
    action_ref_tokens: list[int] = []
    action_ref_indices: list[int] = []
    for action_index, action in enumerate(legal_actions):
        row, ref_rows, ref_tokens = _action_features(action, actor)
        action_rows.append(row)
        for row_ref, token_ref in zip(ref_rows, ref_tokens):
            action_ref_rows.append(row_ref)
            action_ref_tokens.append(token_ref)
            action_ref_indices.append(action_index)
    object_dim = max((len(row) for row in object_rows), default=_object_feature_dim_probe())
    action_dim = max(len(row) for row in action_rows)
    action_ref_dim = max((len(row) for row in action_ref_rows), default=len(ACTION_REF_ROLES) + 6 + len(ZONES) + 2)
    object_rows = [row + [0.0] * (object_dim - len(row)) for row in object_rows]
    action_rows = [row + [0.0] * (action_dim - len(row)) for row in action_rows]
    action_ref_rows = [row + [0.0] * (action_ref_dim - len(row)) for row in action_ref_rows]
    if not object_rows:
        object_rows = [[0.0] * object_dim]
        object_tokens = [0]
        object_groups = [0]
    if not action_ref_rows:
        action_ref_rows = torch.empty((0, action_ref_dim), dtype=torch.float32)
        action_ref_tokens_tensor = torch.empty((0,), dtype=torch.long)
        action_ref_indices_tensor = torch.empty((0,), dtype=torch.long)
    else:
        action_ref_rows = torch.tensor(action_ref_rows, dtype=torch.float32)
        action_ref_tokens_tensor = torch.tensor(action_ref_tokens, dtype=torch.long)
        action_ref_indices_tensor = torch.tensor(action_ref_indices, dtype=torch.long)
    schema = _schema(len(state), object_dim, action_dim, action_ref_dim)
    return EncodedDecision(
        schema=schema,
        state=torch.tensor(state, dtype=torch.float32),
        object_features=torch.tensor(object_rows, dtype=torch.float32),
        object_card_ids=torch.tensor(object_tokens, dtype=torch.long),
        object_groups=torch.tensor(object_groups, dtype=torch.long),
        action_features=torch.tensor(action_rows, dtype=torch.float32),
        action_ref_features=action_ref_rows if isinstance(action_ref_rows, torch.Tensor) else torch.tensor(action_ref_rows, dtype=torch.float32),
        action_ref_card_ids=action_ref_tokens_tensor,
        action_ref_action_indices=action_ref_indices_tensor,
    )


def every_action_variant_fixture(base_ref: dict[str, Any]) -> list[dict[str, Any]]:
    target_player = {"target_kind": "player", "player": "p1"}
    target_object = {"target_kind": "object", "object": {**base_ref, "arena_id": base_ref["arena_id"] + 1, "card_db_id": base_ref["card_db_id"] + 1}}
    second_ref = {**base_ref, "arena_id": base_ref["arena_id"] + 2, "card_db_id": base_ref["card_db_id"] + 2}
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
                {"action_kind": "choose_target", "actor": "p0", "source": base_ref, "remaining": 1, "target": target_object},
                {"action_kind": "choose_cost_target", "actor": "p0", "source": base_ref, "cost_kind": "SacrificeLands", "remaining": 1, "candidate": second_ref},
                {"action_kind": "choose_cast_mode", "actor": "p0", "source": base_ref, "mode": "Normal"},
                {"action_kind": "choose_kicker", "actor": "p0", "source": base_ref, "pay": True},
                {"action_kind": "choose_spell_mode", "actor": "p0", "source": base_ref, "mode_index": 0, "mode_count": 2},
                {"action_kind": "choose_optional_cost_use", "actor": "p0", "use_cost": True},
                {"action_kind": "choose_optional_cost_which", "actor": "p0", "choice": "Discard"},
                {"action_kind": "choose_madness_cast", "actor": "p0", "card": base_ref, "cast_it": True},
                {"action_kind": "discard", "actor": "p0", "cards": [base_ref, second_ref]},
                {"action_kind": "declare_attackers", "actor": "p0", "attackers": [base_ref, second_ref]},
                {"action_kind": "declare_blockers_for_attacker", "actor": "p0", "attacker": base_ref, "blockers": [second_ref]},
                {"action_kind": "order_triggers", "actor": "p0", "pending_sources": [base_ref, second_ref], "order": [1, 0]},
            ]
        )
    ]
