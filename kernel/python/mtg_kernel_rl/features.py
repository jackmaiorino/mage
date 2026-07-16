"""Path-aware v4 feature contract and actor-relative encoder."""

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

FEATURE_SCHEMA_VERSION = "actor-relative-v4-python-3"
FEATURE_REGISTRY_VERSION = "rust-observation-v4-action-v4-registry-3"
ENCODING_CONTRACT_VERSION = "actor-relative-node-graph-9"

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
    "pending_spell_copy",
    "pending_effect",
    "pending_triggers",
    "halted",
]
SURFACE_STAGES = ["priority", "declare_blockers_for_attacker", "discard_pick", "optional_cost_use", "optional_cost_which"]
STACK_KINDS = ["spell", "activated_ability", "triggered_ability", "madness_offer"]
CAST_METHODS = ["normal", "alternative", "flashback", "madness", "plotted", "escape", "bestow", "omen"]
MANA_COLORS = ["W", "U", "B", "R", "G", "C"]
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
    "choose_effect_option",
    "choose_effect_target",
    "finish_effect_selection",
    "choose_effect_color",
    "choose_effect_number",
    "choose_effect_boolean",
    "finish_target_selection",
    "choose_optional_cost_use",
    "choose_optional_cost_which",
    "choose_spell_copy_payment",
    "choose_spell_copy_retarget",
    "choose_madness_cast",
    "discard",
    "declare_attackers",
    "declare_blockers_for_attacker",
    "order_triggers",
]
CAST_MODES = ["Normal", "Alternative"]
COST_KINDS = [
    "SacrificeLands",
    "SacrificePermanents",
    "SacrificeCreatures",
    "SacrificeArtifacts",
    "DiscardCards",
    "ExileFromGraveyard",
    "TapPermanents",
    "ReturnPermanentsToHand",
    "PayLife",
    "RemoveCounters",
    "PutCounters",
]
OPTIONAL_COST_CHOICES = ["Decline", "Discard", "SacrificeLand"]
DISCARD_RESUME_STAGES = ["none", "finish_cast", "finish_activation", "finish_spell_resolution", "finish_optional_cost"]
TRIGGER_KINDS = ["triggered_ability", "madness_offer"]
SPELL_COPY_STAGES = ["payment", "retarget", "target"]
EFFECT_DURATIONS = ["end_of_turn", "until_controllers_next_turn", "while_attached", "while_source_present"]
TARGET_SELECTION_PURPOSES = [
    "effect_targets",
    "card_selection",
    "permanent_selection",
    "player_selection",
    "damage_division",
    "cost_payment",
    "library_order",
    "search_result",
]
BOOLEAN_CHOICE_PURPOSES = ["optional_effect", "shuffle", "pay_cost"]
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
    "known_library_self",
    "known_library_opponent",
    "known_hand_self",
    "known_hand_opponent",
    "paid_cost",
]
EDGE_ROLES = [
    "attachment",
    "stack_target",
    "combat_attacker",
    "combat_blocker",
    "effect_affected",
    "effect_source",
    "permission",
    "pending_context",
    "private_context",
    "known_library",
    "known_hand",
    "attached_to",
    "exiled_by",
    "paid_cost",
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

OBJECT_SOURCE_KINDS = ["card", "stack", "combat", "effect", "permission", "attachment", "target", "pending", "private", "known_library", "known_hand", "paid_cost"]
CARD_REF_FEATURE_DIM = 6 + len(ZONES)
OBJECT_FEATURE_DIM = (
    CARD_REF_FEATURE_DIM
    + 9  # tapped/sick/damage, five counter kinds, legacy attachment count
    + 3  # plotted-turn relation
    + 2  # token and face
    + 1 + len(MANA_COLORS)  # optional chosen color
    + 3  # entered-battlefield turn relation
    + 5  # ability-use count/total/max-index and per-kind summaries
    + 1  # skip-next-untap
    + 3  # goad count/self/opponent
    + 6  # type flags
    + 4  # printed/effective power and toughness
    + len(MANA_COLORS) + 1  # effective color mask and subtype count
    + 14  # boolean keyword flags
    + 2 + len(MANA_COLORS)  # ward, minimum blockers, landwalk mask
    + len(OBJECT_SOURCE_KINDS)
    + 1  # order
)
EDGE_FEATURE_DIM = len(EDGE_ROLES) + 3 + 24
ACTION_REF_FEATURE_DIM = len(ACTION_REF_ROLES) + CARD_REF_FEATURE_DIM + 2
ACTION_FEATURE_DIM = (
    len(ACTION_KINDS)
    + 3
    + CARD_REF_FEATURE_DIM
    + len(TARGET_KINDS)
    + 3
    + 22
    + 1 + (2 * len(MANA_COLORS))
    + len(CAST_MODES)
    + len(COST_KINDS)
    + len(OPTIONAL_COST_CHOICES)
    + ACTION_HASH_DIM
)
STATE_FEATURE_DIM = (
    len(PHASES)
    + 9  # active player, priority player, initiative
    + (2 * 19)  # per-player public resources, status, and dungeon summaries
    + 9  # combat/stack/effect/permission and relation summaries
    + 4  # known-library and known-hand counts by relative owner
    + 26  # engine context
    + len(SURFACE_STAGES)
    + 2
    + 10
    + STATE_HASH_DIM
)
CARD_TOKEN_VOCAB_SIZE = 65_537

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
    ("observation", "known_library_cards"),
    ("observation", "known_hand_cards"),
}

SET_LIKE_LISTS = {
    ("observation", "projection", "continuous_effects", "[]", "affected_objects"),
    ("observation", "projection", "exile_play_permissions"),
    ("observation", "projection", "object_relations"),
    ("observation", "projection", "continuous_effects", "[]", "add_subtype_ids"),
    ("observation", "projection", "continuous_effects", "[]", "remove_subtype_ids"),
    ("observation", "projection", "continuous_effects", "[]", "affected_players"),
    ("observation", "projection", "battlefield", "[]", "[]", "goaded_by"),
    ("observation", "projection", "graveyards", "[]", "[]", "goaded_by"),
    ("observation", "projection", "exile", "[]", "goaded_by"),
    ("observation", "projection", "player_status", "[]", "dungeon", "completed_dungeons"),
    ("observation", "known_hand_cards", "[]"),
    ("legal_action", "semantic", "cards"),
    ("legal_action", "semantic", "attackers"),
    ("legal_action", "semantic", "blockers"),
}
UNORDERED_LISTS = SET_LIKE_LISTS

PENDING_CONTEXT_SUBROLES = [
    ("pending_cast", "source"),
    ("pending_cast", "chosen_targets", "[]", "object"),
    ("pending_cast", "additional_cost_discarded", "[]"),
    ("pending_cast", "sacrifice_chosen", "[]"),
    ("pending_activation", "source"),
    ("pending_activation", "chosen_targets", "[]", "object"),
    ("pending_activation", "cost_discard_paid", "[]"),
    ("pending_discard", "resume_source"),
    ("pending_optional_cost", "source"),
    ("pending_optional_cost", "spell_resume_source"),
    ("pending_optional_cost_sacrifice", "source"),
    ("pending_optional_cost_sacrifice", "chosen", "[]"),
    ("pending_optional_cost_sacrifice", "spell_resume_source"),
    ("pending_spell_copy", "parent"),
    ("pending_spell_copy", "inherited_target", "object"),
    ("pending_spell_copy", "copy"),
    ("pending_effect", "source"),
    ("pending_effect", "choice", "selected_targets", "[]", "object"),
    ("pending_effect", "choice", "legal_targets", "[]", "object"),
    ("pending_triggers", "[]", "source"),
]
PRIVATE_CONTEXT_SUBROLES = [
    ("madness_cast_reprompt_source",),
    ("private_blockers", "current_attacker"),
    ("private_blockers", "accumulated", "[]", "0"),
    ("private_blockers", "accumulated", "[]", "1"),
    ("private_blockers", "remaining", "[]", "0"),
    ("private_blockers", "remaining", "[]", "1", "[]"),
    ("private_discard", "chosen", "[]"),
    ("private_discard", "remaining_choices", "[]"),
]
CONTEXT_SUBROLE_IDS = {path: i for i, path in enumerate(PENDING_CONTEXT_SUBROLES + PRIVATE_CONTEXT_SUBROLES)}
DETACHED_CONTEXT_REF_ALLOWLIST = {
    # rl.rs::pending_discard_semantic_v2 can expose a resolving spell source
    # after resolve_top_of_stack has popped the public stack item but before
    # apply_discard moves the object out of Stack.
    ("pending_discard", "resume_source"): frozenset({"Stack"}),
    # effect.rs::MayPayCostThen is staged from the resolving item's ExecCtx
    # after resolve_top_of_stack has popped that item from the public stack
    # vector; engine.rs keeps the source object in Zone::Stack until the
    # optional-cost branch completes and performs the deferred zone move.
    ("pending_optional_cost", "source"): frozenset({"Stack"}),
    ("pending_optional_cost", "spell_resume_source"): frozenset({"Stack"}),
    ("pending_optional_cost_sacrifice", "source"): frozenset({"Stack"}),
    ("pending_optional_cost_sacrifice", "spell_resume_source"): frozenset({"Stack"}),
}

# Derived from rl.rs::engine_context_v2 priority order and
# surface_v2.rs::next_decision_inner reshape order. Keep fail-closed: a new
# emitted pairing must be added here with a Rust citation before Python accepts it.
ENGINE_SURFACE_TUPLE_CONTRACT = [
    "priority+priority: ordinary priority, declare attackers, or direct engine decisions without a surface reshape",
    "priority+declare_blockers_for_attacker: HarnessSurfaceV2 blockers reshape over engine DeclareBlockers",
    "pending_cast+priority: cast target/mode/cost-target/kicker subdecision",
    "pending_cast+discard_pick: cast paused by additional-cost discard",
    "pending_activation+priority: activation target subdecision",
    "pending_activation+discard_pick: activation paused by discard cost",
    "pending_discard+discard_pick: direct cleanup/effect/optional-cost discard, optionally with queued triggers",
    "pending_optional_cost+optional_cost_use: optional-cost use gate",
    "pending_optional_cost+optional_cost_which: optional-cost discard-vs-sacrifice gate when both are payable",
    "pending_optional_cost_sacrifice+priority: sacrifice-land cost-target subdecision",
    "pending_spell_copy+priority: spell-copy payment, retarget, or target subdecision",
    "pending_effect+priority: generic resumable effect choice",
    "pending_triggers+priority: trigger ordering subdecision",
    "halted+priority: halted engine branch with no surface reshape",
]


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
    edge_feature_dim: int
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
    object_node_ids: torch.Tensor
    edge_features: torch.Tensor
    edge_source_indices: torch.Tensor
    edge_target_indices: torch.Tensor
    action_features: torch.Tensor
    action_ref_features: torch.Tensor
    action_ref_card_ids: torch.Tensor
    action_ref_action_indices: torch.Tensor
    action_ref_node_indices: torch.Tensor


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
I32_MIN = -2_147_483_648
I32_MAX = 2_147_483_647

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

COUNTER_NAMES = ("plus1_plus1", "minus1_minus1", "minus0_minus1", "stun", "lore")
BOOLEAN_KEYWORD_NAMES = (
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
    "lifelink",
    "hexproof",
    "indestructible",
    "protection_from_monocolored",
)

COUNTERS = ObjectSpec({name: I(MODEL_INPUT, minimum=-32_768, maximum=32_767) for name in COUNTER_NAMES})
TYPE_FLAGS = ObjectSpec({name: B(MODEL_INPUT) for name in ("land", "creature", "instant", "sorcery", "artifact", "enchantment")})
KEYWORDS = ObjectSpec(
    {
        **{name: B(MODEL_INPUT) for name in BOOLEAN_KEYWORD_NAMES},
        "ward_generic": I(MODEL_INPUT, maximum=U16),
        "minimum_blockers": I(MODEL_INPUT, maximum=U8),
        "landwalk_mask": I(MODEL_INPUT, maximum=U8),
    }
)
CHARACTERISTICS = ObjectSpec(
    {
        "type_flags": TYPE_FLAGS,
        "base_power": Opt(I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX)),
        "base_toughness": Opt(I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX)),
        "effective_power": Opt(I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX)),
        "effective_toughness": Opt(I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX)),
        "effective_color_mask": I(MODEL_INPUT, maximum=U8),
        "effective_subtype_ids": ListSpec(I(MODEL_INPUT, maximum=U16)),
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
        "attachments": ListSpec(I(OPERATIONAL_ONLY, maximum=U32)),
        "plotted_turn": Opt(I(MODEL_INPUT, maximum=U32)),
        "is_token": B(MODEL_INPUT),
        "face_index": I(MODEL_INPUT, maximum=U8),
        "chosen_color": Opt(E(MANA_COLORS)),
        "entered_battlefield_turn": Opt(I(MODEL_INPUT, maximum=U32)),
        "ability_uses_this_turn": ListSpec(
            ObjectSpec(
                {
                    "ability_kind": E(["mana", "activated"]),
                    "ability_index": I(MODEL_INPUT, maximum=U16),
                    "uses": I(MODEL_INPUT, maximum=U16),
                }
            )
        ),
        "skip_next_untap": B(MODEL_INPUT),
        "goaded_by": ListSpec(
            ObjectSpec(
                {
                    "player": Seat(),
                    "expires_at_turn": I(MODEL_INPUT, maximum=U32),
                }
            )
        ),
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
        "stack_index": I(OPERATIONAL_ONLY, maximum=U32),
        "source": CARD_STABLE_REF,
        "controller": Seat(),
        "targets": ListSpec(TARGET_REF),
        "stack_item_kind": E(STACK_KINDS),
        "is_copy": B(MODEL_INPUT),
        "is_flashback": B(MODEL_INPUT),
        "mode_chosen": I(MODEL_INPUT, maximum=U8),
        "madness_offer": B(MODEL_INPUT),
        "kicked": B(MODEL_INPUT),
        "cast_method": Opt(E(CAST_METHODS)),
        "face_index": I(MODEL_INPUT, maximum=U8),
        "x_value": I(MODEL_INPUT, maximum=U16),
        "paid_cost_refs": ListSpec(CARD_STABLE_REF),
    }
)
DUNGEON = ObjectSpec(
    {
        "dungeon_id": Opt(I(MODEL_INPUT, maximum=U16)),
        "room_id": Opt(I(MODEL_INPUT, maximum=U16)),
        "completed_dungeons": ListSpec(I(MODEL_INPUT, maximum=U16)),
    }
)
PLAYER_STATUS = ObjectSpec(
    {
        "has_lost": B(MODEL_INPUT),
        "lands_played_this_turn": I(MODEL_INPUT, maximum=U8),
        "drew_from_empty": B(MODEL_INPUT),
        "draws_this_turn": I(MODEL_INPUT, maximum=U32),
        "spells_cast_this_turn": I(MODEL_INPUT, maximum=U16),
        "dungeon": DUNGEON,
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
        "source": Opt(CARD_STABLE_REF),
        "controller": Opt(Seat()),
        "affected_objects": ListSpec(CARD_STABLE_REF),
        "affected_players": ListSpec(Seat()),
        "global": B(MODEL_INPUT),
        "layers": I(MODEL_INPUT, maximum=U8),
        "timestamp": I(OPERATIONAL_ONLY, maximum=U64),
        "duration": E(EFFECT_DURATIONS),
        "power_delta": I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX),
        "toughness_delta": I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX),
        "grants_haste": B(MODEL_INPUT),
        "set_power": Opt(I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX)),
        "set_toughness": Opt(I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX)),
        "add_color_mask": I(MODEL_INPUT, maximum=U8),
        "remove_color_mask": I(MODEL_INPUT, maximum=U8),
        "add_subtype_ids": ListSpec(I(MODEL_INPUT, maximum=U16)),
        "remove_subtype_ids": ListSpec(I(MODEL_INPUT, maximum=U16)),
        "add_keyword_mask": I(MODEL_INPUT, maximum=U32),
        "remove_keyword_mask": I(MODEL_INPUT, maximum=U32),
        "ward_generic_delta": I(MODEL_INPUT, minimum=-32_768, maximum=32_767),
        "minimum_blockers": Opt(I(MODEL_INPUT, maximum=U8)),
        "add_landwalk_mask": I(MODEL_INPUT, maximum=U8),
        "remove_landwalk_mask": I(MODEL_INPUT, maximum=U8),
        "prevent_damage_from_color_mask": I(MODEL_INPUT, maximum=U8),
        "damage_cannot_be_prevented": B(MODEL_INPUT),
    }
)
OBJECT_RELATION = VariantSpec(
    "relation_kind",
    {
        "attached_to": ObjectSpec(
            {
                "relation_kind": E(["attached_to"]),
                "object": CARD_STABLE_REF,
                "attached_to": CARD_STABLE_REF,
            }
        ),
        "exiled_by": ObjectSpec(
            {
                "relation_kind": E(["exiled_by"]),
                "object": CARD_STABLE_REF,
                "exiled_by": CARD_STABLE_REF,
            }
        ),
    },
    MODEL_INPUT,
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
        "zone_change_generation": I(OPERATIONAL_ONLY, maximum=U32),
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
PENDING_SPELL_COPY = ObjectSpec(
    {
        "parent": Opt(CARD_STABLE_REF),
        "player": Seat(),
        "inherited_target": TARGET_REF,
        "stage": E(SPELL_COPY_STAGES),
        "copy": Opt(CARD_STABLE_REF),
    }
)
PENDING_EFFECT_CHOICE = VariantSpec(
    "choice_kind",
    {
        "options": ObjectSpec(
            {
                "choice_kind": E(["options"]),
                "player": Seat(),
                "structural_path": ListSpec(I(MODEL_INPUT, maximum=U16)),
                "option_count": I(MODEL_INPUT, maximum=U16),
            }
        ),
        "targets": ObjectSpec(
            {
                "choice_kind": E(["targets"]),
                "player": Seat(),
                "structural_path": ListSpec(I(MODEL_INPUT, maximum=U16)),
                "selected_targets": ListSpec(TARGET_REF),
                "legal_targets": ListSpec(TARGET_REF),
                "min_targets": I(MODEL_INPUT, maximum=U16),
                "max_targets": I(MODEL_INPUT, maximum=U16),
                "can_finish": B(MODEL_INPUT),
                "ordered": B(MODEL_INPUT),
                "purpose": E(TARGET_SELECTION_PURPOSES),
            }
        ),
        "color": ObjectSpec(
            {
                "choice_kind": E(["color"]),
                "player": Seat(),
                "structural_path": ListSpec(I(MODEL_INPUT, maximum=U16)),
                "legal_colors": ListSpec(E(MANA_COLORS)),
            }
        ),
        "number": ObjectSpec(
            {
                "choice_kind": E(["number"]),
                "player": Seat(),
                "structural_path": ListSpec(I(MODEL_INPUT, maximum=U16)),
                "minimum": I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX),
                "maximum": I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX),
            }
        ),
        "boolean": ObjectSpec(
            {
                "choice_kind": E(["boolean"]),
                "player": Seat(),
                "structural_path": ListSpec(I(MODEL_INPUT, maximum=U16)),
                "default": Opt(B(MODEL_INPUT)),
                "purpose": E(BOOLEAN_CHOICE_PURPOSES),
            }
        ),
    },
    MODEL_INPUT,
)
PENDING_EFFECT = ObjectSpec(
    {
        "source": Opt(CARD_STABLE_REF),
        "controller": Seat(),
        "choice": Opt(PENDING_EFFECT_CHOICE),
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
        "pending_spell_copy": Opt(PENDING_SPELL_COPY),
        "pending_effect": Opt(PENDING_EFFECT),
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
PENDING_CONTEXT_ROOTS = (
    ("pending_cast", PENDING_CAST, ("pending_cast",)),
    ("pending_activation", PENDING_ACTIVATION, ("pending_activation",)),
    ("pending_discard", PENDING_DISCARD, ("pending_discard",)),
    ("pending_optional_cost", PENDING_OPTIONAL_COST, ("pending_optional_cost",)),
    ("pending_optional_cost_sacrifice", PENDING_OPTIONAL_COST_SAC, ("pending_optional_cost_sacrifice",)),
    ("pending_spell_copy", PENDING_SPELL_COPY, ("pending_spell_copy",)),
    ("pending_effect", PENDING_EFFECT, ("pending_effect",)),
    ("pending_triggers", ListSpec(PENDING_TRIGGER), ("pending_triggers",)),
)
PRIVATE_CONTEXT_ROOTS = (
    ("madness_cast_reprompt_source", CARD_STABLE_REF, ("madness_cast_reprompt_source",)),
    ("private_blockers", PRIVATE_BLOCKERS, ("private_blockers",)),
    ("private_discard", PRIVATE_DISCARD, ("private_discard",)),
    ("private_optional_cost", PRIVATE_OPTIONAL_COST, ("private_optional_cost",)),
)
PROJECTION = ObjectSpec(
    {
        "turn": I(OPERATIONAL_ONLY, maximum=U32),
        "phase": E(PHASES),
        "active_player": Seat(),
        "priority_player": Seat(),
        "initiative": Opt(Seat()),
        "life_totals": ListSpec(I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX), length=2),
        "mana_pools": ListSpec(ListSpec(I(MODEL_INPUT, maximum=U8), length=6), length=2),
        "hand_counts": ListSpec(I(MODEL_INPUT, maximum=U64), length=2),
        "library_counts": ListSpec(I(MODEL_INPUT, maximum=U64), length=2),
        "player_status": ListSpec(PLAYER_STATUS, length=2),
        "battlefield": ListSpec(ListSpec(CARD_PUBLIC), length=2),
        "graveyards": ListSpec(ListSpec(CARD_PUBLIC), length=2),
        "exile": ListSpec(CARD_PUBLIC),
        "stack": ListSpec(STACK_ITEM),
        "combat": COMBAT,
        "continuous_effects": ListSpec(EFFECT),
        "object_relations": ListSpec(OBJECT_RELATION),
        "exile_play_permissions": ListSpec(PERMISSION),
        "engine_context": ENGINE_CONTEXT,
        "surface_context": SURFACE_CONTEXT,
    }
)
KNOWN_LIBRARY_CARD = ObjectSpec(
    {
        "position": I(MODEL_INPUT, maximum=U32),
        "card": CARD_PRIVATE,
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
        "known_library_cards": ListSpec(ListSpec(KNOWN_LIBRARY_CARD), length=2),
        "known_hand_cards": ListSpec(ListSpec(CARD_PRIVATE), length=2),
        "visible_projection_hash": I(FORBIDDEN, maximum=U64),
    }
)

ACTION_VARIANTS = {
    "pass": ObjectSpec({"action_kind": E(["pass"]), "actor": Seat()}),
    "play_land": ObjectSpec({"action_kind": E(["play_land"]), "actor": Seat(), "source": CARD_STABLE_REF}),
    "cast_spell": ObjectSpec({"action_kind": E(["cast_spell"]), "actor": Seat(), "source": CARD_STABLE_REF}),
    "activate_mana_ability": ObjectSpec({"action_kind": E(["activate_mana_ability"]), "actor": Seat(), "source": CARD_STABLE_REF, "mana_choice": Opt(E(MANA_COLORS))}),
    "activate_ability": ObjectSpec({"action_kind": E(["activate_ability"]), "actor": Seat(), "source": CARD_STABLE_REF, "ability_index": I(MODEL_INPUT, maximum=U8)}),
    "plot_spell": ObjectSpec({"action_kind": E(["plot_spell"]), "actor": Seat(), "source": CARD_STABLE_REF}),
    "choose_target": ObjectSpec({"action_kind": E(["choose_target"]), "actor": Seat(), "source": CARD_STABLE_REF, "remaining": I(MODEL_INPUT, maximum=U8), "target": TARGET_REF}),
    "choose_cost_target": ObjectSpec({"action_kind": E(["choose_cost_target"]), "actor": Seat(), "source": CARD_STABLE_REF, "cost_kind": E(COST_KINDS), "remaining": I(MODEL_INPUT, maximum=U8), "candidate": CARD_STABLE_REF}),
    "choose_cast_mode": ObjectSpec({"action_kind": E(["choose_cast_mode"]), "actor": Seat(), "source": CARD_STABLE_REF, "mode": E(CAST_MODES)}),
    "choose_kicker": ObjectSpec({"action_kind": E(["choose_kicker"]), "actor": Seat(), "source": CARD_STABLE_REF, "pay": B(MODEL_INPUT)}),
    "choose_spell_mode": ObjectSpec({"action_kind": E(["choose_spell_mode"]), "actor": Seat(), "source": CARD_STABLE_REF, "mode_index": I(MODEL_INPUT, maximum=U8), "mode_count": I(MODEL_INPUT, maximum=U8)}),
    "choose_effect_option": ObjectSpec({"action_kind": E(["choose_effect_option"]), "actor": Seat(), "source": CARD_STABLE_REF, "option_index": I(MODEL_INPUT, maximum=U16), "option_count": I(MODEL_INPUT, maximum=U16)}),
    "choose_effect_target": ObjectSpec({"action_kind": E(["choose_effect_target"]), "actor": Seat(), "source": CARD_STABLE_REF, "target": TARGET_REF, "selected_count": I(MODEL_INPUT, maximum=U16), "min_targets": I(MODEL_INPUT, maximum=U16), "max_targets": I(MODEL_INPUT, maximum=U16)}),
    "finish_effect_selection": ObjectSpec({"action_kind": E(["finish_effect_selection"]), "actor": Seat(), "source": CARD_STABLE_REF, "selected_count": I(MODEL_INPUT, maximum=U16)}),
    "choose_effect_color": ObjectSpec({"action_kind": E(["choose_effect_color"]), "actor": Seat(), "source": CARD_STABLE_REF, "color": E(MANA_COLORS)}),
    "choose_effect_number": ObjectSpec({"action_kind": E(["choose_effect_number"]), "actor": Seat(), "source": CARD_STABLE_REF, "number": I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX), "minimum": I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX), "maximum": I(MODEL_INPUT, minimum=I32_MIN, maximum=I32_MAX)}),
    "choose_effect_boolean": ObjectSpec({"action_kind": E(["choose_effect_boolean"]), "actor": Seat(), "source": CARD_STABLE_REF, "value": B(MODEL_INPUT)}),
    "finish_target_selection": ObjectSpec({"action_kind": E(["finish_target_selection"]), "actor": Seat(), "source": CARD_STABLE_REF, "selected_count": I(MODEL_INPUT, maximum=U16)}),
    "choose_optional_cost_use": ObjectSpec({"action_kind": E(["choose_optional_cost_use"]), "actor": Seat(), "use_cost": B(MODEL_INPUT)}),
    "choose_optional_cost_which": ObjectSpec({"action_kind": E(["choose_optional_cost_which"]), "actor": Seat(), "choice": E(OPTIONAL_COST_CHOICES)}),
    "choose_spell_copy_payment": ObjectSpec({"action_kind": E(["choose_spell_copy_payment"]), "actor": Seat(), "source": CARD_STABLE_REF, "pay": B(MODEL_INPUT)}),
    "choose_spell_copy_retarget": ObjectSpec({"action_kind": E(["choose_spell_copy_retarget"]), "actor": Seat(), "source": CARD_STABLE_REF, "change_target": B(MODEL_INPUT)}),
    "choose_madness_cast": ObjectSpec({"action_kind": E(["choose_madness_cast"]), "actor": Seat(), "card": CARD_STABLE_REF, "cast_it": B(MODEL_INPUT)}),
    "discard": ObjectSpec({"action_kind": E(["discard"]), "actor": Seat(), "cards": ListSpec(CARD_STABLE_REF)}),
    "declare_attackers": ObjectSpec({"action_kind": E(["declare_attackers"]), "actor": Seat(), "attackers": ListSpec(CARD_STABLE_REF)}),
    "declare_blockers_for_attacker": ObjectSpec({"action_kind": E(["declare_blockers_for_attacker"]), "actor": Seat(), "attacker": CARD_STABLE_REF, "blockers": ListSpec(CARD_STABLE_REF)}),
    "order_triggers": ObjectSpec({"action_kind": E(["order_triggers"]), "actor": Seat(), "pending_sources": ListSpec(CARD_STABLE_REF), "order": ListSpec(I(MODEL_INPUT, maximum=U64))}),
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
        "set_like_lists": [".".join(path) for path in sorted(SET_LIKE_LISTS)],
        "ordered_lists_contract": [
            "own_hand_oldest_first",
            "battlefield_entry_order",
            "graveyard_recorded_order_last_is_top",
            "exile_recorded_order",
            "stack_bottom_to_top_last_is_top",
            "combat_attackers_and_blockers_recorded_order",
            "target_pending_private_trigger_sequences",
        ],
        "object_groups": OBJECT_GROUPS,
        "object_source_kinds": OBJECT_SOURCE_KINDS,
        "edge_roles": EDGE_ROLES,
        "action_ref_roles": ACTION_REF_ROLES,
        "fixed_dimensions": {
            "state_dim": STATE_FEATURE_DIM,
            "object_feature_dim": OBJECT_FEATURE_DIM,
            "edge_feature_dim": EDGE_FEATURE_DIM,
            "action_feature_dim": ACTION_FEATURE_DIM,
            "action_ref_feature_dim": ACTION_REF_FEATURE_DIM,
        },
        "card_token_vocab_size": CARD_TOKEN_VOCAB_SIZE,
        "context_subroles": [".".join(path) for path in PENDING_CONTEXT_SUBROLES + PRIVATE_CONTEXT_SUBROLES],
        "engine_surface_tuple_contract": ENGINE_SURFACE_TUPLE_CONTRACT,
        "action_kinds": ACTION_KINDS,
        "stack_kinds": STACK_KINDS,
        "surface_stages": SURFACE_STAGES,
        "engine_stages": ENGINE_STAGES,
        "node_registry": "per_decision_keys_arena_id_zone_change_count_handles_from_actor_relative_order",
    }


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_contract_fingerprint() -> str:
    return _sha256_json(_contract_payload())


def encoding_contract_fingerprint() -> str:
    return _sha256_json(_encoding_payload())


def model_contract_fingerprint(schema: FeatureSchema) -> str:
    return _sha256_json({"model_contract_version": "kernel-policy-value-net-5", "feature_schema": schema.__dict__})


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
    if type(card_db_id) is not int or card_db_id < 0 or card_db_id > U16:
        raise FeatureSchemaError("card_db_id must be inside the admitted u16 domain")
    return card_db_id + 1


def _card_ref_features(stable: dict[str, Any] | None, actor: str) -> list[float]:
    if stable is None:
        return [0.0, 0.0, 1.0, 0.0, 0.0, 1.0] + [0.0] * len(ZONES)
    return _seat_features(stable["owner"], actor) + _seat_features(stable["controller"], actor) + _one_hot(stable["zone"], ZONES)


class _CanonicalContext:
    def __init__(self, actor: str, observation: dict[str, Any] | None = None) -> None:
        self.actor = actor
        self.turn = 0
        self._arena_handles: dict[int, int] = {}
        self._ref_keys: dict[tuple[int, int], Any] = {}
        if observation is not None:
            self.turn = observation.get("projection", {}).get("turn", 0)
            for ref in _iter_card_refs_by_schema(observation, OBSERVATION_SPEC):
                arena = ref.get("arena_id")
                generation = ref.get("zone_change_count")
                if type(arena) is int and arena not in self._arena_handles:
                    self._arena_handles[arena] = len(self._arena_handles)
                if type(arena) is int and type(generation) is int:
                    key = (arena, generation)
                    self._ref_keys.setdefault(
                        key,
                        {
                            "handle": self._arena_handles[arena],
                            "card_db_id": ref.get("card_db_id"),
                            "owner": _relative_seat(ref.get("owner"), actor) if ref.get("owner") in SEATS else ref.get("owner"),
                            "controller": _relative_seat(ref.get("controller"), actor) if ref.get("controller") in SEATS else ref.get("controller"),
                            "zone": ref.get("zone"),
                        },
                    )

    def arena_key(self, raw: int) -> Any:
        if raw not in self._arena_handles:
            raise FeatureSchemaError("attachment reference does not resolve to an observed arena id")
        return {"handle": self._arena_handles[raw]}

    def turn_relation(self, value: int, field: str) -> str:
        if type(value) is not int:
            raise FeatureSchemaError(f"{field} must be an integer when present")
        if type(self.turn) is not int:
            raise FeatureSchemaError("projection.turn must be an integer")
        if value > self.turn:
            raise FeatureSchemaError(f"{field} cannot be in the future relative to projection.turn")
        if value == self.turn:
            return "this_turn"
        return "earlier_turn"

    def future_turn_delta(self, value: int, field: str) -> int:
        if type(value) is not int or type(self.turn) is not int:
            raise FeatureSchemaError(f"{field} and projection.turn must be integers")
        if value < self.turn:
            raise FeatureSchemaError(f"{field} cannot be earlier than projection.turn")
        return value - self.turn


def _iter_card_refs(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"arena_id", "card_db_id", "owner", "controller", "zone", "zone_change_count"}.issubset(value):
            yield value
        for child in value.values():
            yield from _iter_card_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_card_refs(child)


def _iter_card_refs_by_schema(value: Any, spec: Spec) -> Iterable[dict[str, Any]]:
    if value is None:
        return
    if spec is CARD_STABLE_REF:
        yield value
        return
    if isinstance(spec, OptionalSpec):
        if value is not None:
            yield from _iter_card_refs_by_schema(value, spec.item)
        return
    if isinstance(spec, ListSpec):
        for child in value:
            yield from _iter_card_refs_by_schema(child, spec.item)
        return
    if isinstance(spec, TupleSpec):
        for child_spec, child in zip(spec.items, value):
            yield from _iter_card_refs_by_schema(child, child_spec)
        return
    if isinstance(spec, ObjectSpec):
        for key, child_spec in spec.fields.items():
            yield from _iter_card_refs_by_schema(value[key], child_spec)
        return
    if isinstance(spec, VariantSpec):
        yield from _iter_card_refs_by_schema(value, spec.variants[value[spec.tag]])
        return


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_model_value(value: Any, spec: Spec, path: tuple[str, ...], ctx: _CanonicalContext) -> Any:
    normalized = _normalize_path(path)
    if isinstance(spec, ScalarSpec):
        if spec.classification != MODEL_INPUT:
            return _OMIT
        if spec.kind == "seat":
            return _relative_seat(value, ctx.actor)
        if spec.kind == "int" and path and path[-1] in ("plotted_turn", "entered_battlefield_turn"):
            return ctx.turn_relation(value, path[-1])
        if spec.kind == "int" and path and path[-1] == "expires_at_turn":
            return ctx.future_turn_delta(value, path[-1])
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


def _turn_relation_features(value: int | None, current_turn: int, field: str) -> list[float]:
    if value is None:
        return [1.0, 0.0, 0.0]
    if type(value) is not int or type(current_turn) is not int:
        raise FeatureSchemaError(f"{field} and projection.turn must be integers")
    if value > current_turn:
        raise FeatureSchemaError(f"{field} cannot be in the future relative to projection.turn")
    if value == current_turn:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _mask_bits(value: int, width: int, field: str) -> list[float]:
    if type(value) is not int or value < 0 or value >= (1 << width):
        raise FeatureSchemaError(f"{field} must fit inside {width} reserved bits")
    return [1.0 if value & (1 << bit) else 0.0 for bit in range(width)]


def _card_public_features(card: dict[str, Any], actor: str, order_index: int = 0, source_kind: str = "card", current_turn: int = 0) -> tuple[list[float], int]:
    stable = card["stable"]
    characteristics = card.get("characteristics", {})
    type_flags = characteristics.get("type_flags", {})
    keywords = characteristics.get("effective_keywords", {})
    counters = card.get("counters", {})
    chosen_color = card.get("chosen_color")
    ability_uses = card.get("ability_uses_this_turn", [])
    goaded_by = card.get("goaded_by", [])
    source_flags = [1.0 if source_kind == kind else 0.0 for kind in OBJECT_SOURCE_KINDS]
    features = (
        _card_ref_features(stable, actor)
        + [
            _flag(card.get("tapped", False)),
            _flag(card.get("summoning_sick", False)),
            _number(card.get("damage", 0), 20.0),
        ]
        + [_number(counters.get(name, 0), 10.0) for name in COUNTER_NAMES]
        + [
            _number(len(card.get("attachments", [])), 8.0),
        ]
        + _turn_relation_features(card.get("plotted_turn"), current_turn, "plotted_turn")
        + [
            _flag(card.get("is_token", False)),
            _number(card.get("face_index", 0), 8.0),
            1.0 if chosen_color is not None else 0.0,
        ]
        + (_one_hot(chosen_color, MANA_COLORS) if chosen_color is not None else [0.0] * len(MANA_COLORS))
        + _turn_relation_features(card.get("entered_battlefield_turn"), current_turn, "entered_battlefield_turn")
        + [
            _number(len(ability_uses), 8.0),
            _number(sum(item["uses"] for item in ability_uses), 16.0),
            _number(max((item["ability_index"] for item in ability_uses), default=0), 16.0),
            _number(sum(item["uses"] for item in ability_uses if item["ability_kind"] == "mana"), 16.0),
            _number(sum(item["uses"] for item in ability_uses if item["ability_kind"] == "activated"), 16.0),
            _flag(card.get("skip_next_untap", False)),
            _number(len(goaded_by), 2.0),
            1.0 if any(item["player"] == actor for item in goaded_by) else 0.0,
            1.0 if any(item["player"] == ("p1" if actor == "p0" else "p0") for item in goaded_by) else 0.0,
        ]
        + [_flag(type_flags.get(name, False)) for name in ("land", "creature", "instant", "sorcery", "artifact", "enchantment")]
        + [
            _number(characteristics.get("base_power"), 20.0),
            _number(characteristics.get("base_toughness"), 20.0),
            _number(characteristics.get("effective_power"), 20.0),
            _number(characteristics.get("effective_toughness"), 20.0),
        ]
        + _mask_bits(characteristics.get("effective_color_mask", 0), len(MANA_COLORS), "effective_color_mask")
        + [_number(len(characteristics.get("effective_subtype_ids", [])), 16.0)]
        + [_flag(keywords.get(name, False)) for name in BOOLEAN_KEYWORD_NAMES]
        + [
            _number(keywords.get("ward_generic", 0), 16.0),
            _number(keywords.get("minimum_blockers", 0), 8.0),
        ]
        + _mask_bits(keywords.get("landwalk_mask", 0), len(MANA_COLORS), "landwalk_mask")
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
        "counters": {name: 0 for name in COUNTER_NAMES},
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
            "type_flags": {name: False for name in ("land", "creature", "instant", "sorcery", "artifact", "enchantment")},
            "base_power": None,
            "base_toughness": None,
            "effective_power": None,
            "effective_toughness": None,
            "effective_color_mask": 0,
            "effective_subtype_ids": [],
            "effective_keywords": {
                **{name: False for name in BOOLEAN_KEYWORD_NAMES},
                "ward_generic": 0,
                "minimum_blockers": 0,
                "landwalk_mask": 0,
            },
        },
        "_source_kind": source_kind,
    }


def _private_card_features(card: dict[str, Any], actor: str) -> tuple[list[float], int]:
    return _card_public_features(_blank_public_from_ref(card["stable"]), actor)


def _state_features(obs: dict[str, Any]) -> list[float]:
    actor = obs["acting_player"]
    p = obs["projection"]
    state: list[float] = []
    state += _one_hot(p["phase"], PHASES)
    state += _seat_features(p["active_player"], actor)
    state += _seat_features(p["priority_player"], actor)
    state += _seat_features(p["initiative"], actor)
    for rel in (0, 1):
        seat = actor if rel == 0 else ("p1" if actor == "p0" else "p0")
        idx = 0 if seat == "p0" else 1
        state.append(_number(p["life_totals"][idx], 20.0))
        state += [_number(x, 10.0) for x in p["mana_pools"][idx]]
        state.append(_number(p["hand_counts"][idx], 16.0))
        state.append(_number(p["library_counts"][idx], 64.0))
        status = p["player_status"][idx]
        dungeon = status["dungeon"]
        state += [
            _flag(status["has_lost"]),
            _number(status["lands_played_this_turn"], 4.0),
            _flag(status["drew_from_empty"]),
            _number(status["draws_this_turn"], 8.0),
            _number(status["spells_cast_this_turn"], 8.0),
            1.0 if dungeon["dungeon_id"] is not None else 0.0,
            _number(dungeon["dungeon_id"], 32.0),
            1.0 if dungeon["room_id"] is not None else 0.0,
            _number(dungeon["room_id"], 32.0),
            _number(len(dungeon["completed_dungeons"]), 8.0),
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
        _number(sum(1 for relation in p["object_relations"] if relation["relation_kind"] == "attached_to"), 32.0),
        _number(sum(1 for relation in p["object_relations"] if relation["relation_kind"] == "exiled_by"), 32.0),
    ]
    for rel in (0, 1):
        seat = actor if rel == 0 else ("p1" if actor == "p0" else "p0")
        idx = 0 if seat == "p0" else 1
        state.append(_number(len(obs["known_library_cards"][idx]), 16.0))
    for rel in (0, 1):
        seat = actor if rel == 0 else ("p1" if actor == "p0" else "p0")
        idx = 0 if seat == "p0" else 1
        state.append(_number(len(obs["known_hand_cards"][idx]), 16.0))
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
        1.0 if engine["pending_spell_copy"] else 0.0,
        1.0 if engine["pending_effect"] else 0.0,
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


def _edge_role_id(name: str) -> int:
    return EDGE_ROLES.index(name)


def _stable_key(ref: dict[str, Any]) -> tuple[int, int]:
    arena = ref.get("arena_id")
    generation = ref.get("zone_change_count")
    if type(arena) is not int or type(generation) is not int:
        raise FeatureSchemaError("card reference must carry integer arena_id and zone_change_count")
    return arena, generation


def _stable_identity(ref: dict[str, Any]) -> tuple[int, str, str, str]:
    return (ref["card_db_id"], ref["owner"], ref["controller"], ref["zone"])


@dataclass
class _NodeRegistry:
    actor: str
    current_turn: int

    def __post_init__(self) -> None:
        self.rows: list[list[float]] = []
        self.tokens: list[int] = []
        self.groups: list[int] = []
        self.node_ids: list[int] = []
        self._node_by_key: dict[tuple[int, int], int] = {}
        self._identity_by_key: dict[tuple[int, int], tuple[int, str, str, str]] = {}
        self._node_by_arena: dict[int, int] = {}

    def validate_ref(self, ref: dict[str, Any]) -> tuple[int, int]:
        key = _stable_key(ref)
        identity = _stable_identity(ref)
        previous = self._identity_by_key.get(key)
        if previous is not None and previous != identity:
            raise FeatureSchemaError("inconsistent repeated stable reference for same incarnation")
        self._identity_by_key[key] = identity
        return key

    def add_card(self, card: dict[str, Any], group: str, order: int, source_kind: str = "card") -> int:
        stable = card["stable"]
        key = self.validate_ref(stable)
        features, token = _card_public_features(card, self.actor, order, source_kind, self.current_turn)
        return self._add_node(key, features, token, group)

    def add_ref_node(self, ref: dict[str, Any], group: str, order: int, source_kind: str) -> int:
        key = self.validate_ref(ref)
        features, token = _card_public_features(_blank_public_from_ref(ref, source_kind), self.actor, order, source_kind, self.current_turn)
        return self._add_node(key, features, token, group)

    def add_historical_ref_node(self, ref: dict[str, Any], order: int) -> int:
        """Register payment-time provenance without claiming a current arena incarnation."""
        key = self.validate_ref(ref)
        features, token = _card_public_features(
            _blank_public_from_ref(ref, "paid_cost"),
            self.actor,
            order,
            "paid_cost",
            self.current_turn,
        )
        return self._add_node(key, features, token, "paid_cost", register_arena=False)

    def resolve_context_ref_node(self, ref: dict[str, Any], group: str, order: int, source_kind: str, path: tuple[str, ...]) -> int:
        key = self.validate_ref(ref)
        if key in self._node_by_key:
            return self._node_by_key[key]
        allowed_zones = DETACHED_CONTEXT_REF_ALLOWLIST.get(path)
        if allowed_zones is None or ref["zone"] not in allowed_zones:
            raise FeatureSchemaError(f"context stable reference does not resolve to an observed object node: {'.'.join(path)}")
        return self.add_ref_node(ref, group, order, source_kind)

    def _add_node(self, key: tuple[int, int], features: list[float], token: int, group: str, *, register_arena: bool = True) -> int:
        if key in self._node_by_key:
            return self._node_by_key[key]
        arena = key[0]
        if register_arena and arena in self._node_by_arena:
            raise FeatureSchemaError("multiple visible incarnations share one arena_id in a single decision")
        node_id = len(self.rows)
        self._node_by_key[key] = node_id
        if register_arena:
            self._node_by_arena[arena] = node_id
        self.rows.append(features)
        self.tokens.append(token)
        self.groups.append(_group_id(group))
        self.node_ids.append(node_id)
        return node_id

    def resolve(self, ref: dict[str, Any]) -> int:
        key = self.validate_ref(ref)
        if key not in self._node_by_key:
            raise FeatureSchemaError("visible stable reference does not resolve to a registered object node")
        return self._node_by_key[key]

    def resolve_attachment_arena(self, raw: int) -> int:
        if type(raw) is not int:
            raise FeatureSchemaError("attachment id must be an integer")
        if raw not in self._node_by_arena:
            raise FeatureSchemaError("attachment reference does not resolve to a registered object node")
        return self._node_by_arena[raw]


def _edge_row(role: str, primary_order: int = 0, secondary_order: int = 0, associated_order: int = 0, extra: list[float] | None = None) -> list[float]:
    row = _one_hot(role, EDGE_ROLES) + [_number(primary_order, 64.0), _number(secondary_order, 64.0), _number(associated_order, 64.0)]
    if extra:
        row += extra
    if len(row) > EDGE_FEATURE_DIM:
        raise FeatureSchemaError("edge feature row exceeds fixed contract width")
    row += [0.0] * (EDGE_FEATURE_DIM - len(row))
    return row


def _append_edge(
    edge_rows: list[list[float]],
    edge_sources: list[int],
    edge_targets: list[int],
    source_node: int,
    target_node: int,
    role: str,
    primary_order: int = 0,
    secondary_order: int = 0,
    associated_order: int = 0,
    extra: list[float] | None = None,
) -> None:
    edge_rows.append(_edge_row(role, primary_order, secondary_order, associated_order, extra))
    edge_sources.append(source_node)
    edge_targets.append(target_node)


def _context_ref_edges(
    registry: _NodeRegistry,
    edge_rows: list[list[float]],
    edge_sources: list[int],
    edge_targets: list[int],
    value: Any,
    spec: Spec,
    path: tuple[str, ...],
    role: str,
    order_counter: list[int],
) -> None:
    if value is None:
        return
    if spec is CARD_STABLE_REF:
        normalized = path
        try:
            subrole = CONTEXT_SUBROLE_IDS[normalized]
        except KeyError as exc:
            raise FeatureSchemaError(f"context card reference path is not declared for edge encoding: {'.'.join(normalized)}") from exc
        order = order_counter[0]
        order_counter[0] += 1
        group = "pending_context" if role == "pending_context" else "private_context"
        source_kind = "pending" if role == "pending_context" else "private"
        node = registry.resolve_context_ref_node(value, group, order, source_kind, normalized)
        _append_edge(edge_rows, edge_sources, edge_targets, node, node, role, order, subrole)
        return
    if isinstance(spec, OptionalSpec):
        if value is not None:
            _context_ref_edges(registry, edge_rows, edge_sources, edge_targets, value, spec.item, path, role, order_counter)
        return
    if isinstance(spec, ListSpec):
        for child in value:
            _context_ref_edges(registry, edge_rows, edge_sources, edge_targets, child, spec.item, path + ("[]",), role, order_counter)
        return
    if isinstance(spec, TupleSpec):
        for i, (child_spec, child) in enumerate(zip(spec.items, value)):
            _context_ref_edges(registry, edge_rows, edge_sources, edge_targets, child, child_spec, path + (str(i),), role, order_counter)
        return
    if isinstance(spec, ObjectSpec):
        for key, child_spec in spec.fields.items():
            _context_ref_edges(registry, edge_rows, edge_sources, edge_targets, value[key], child_spec, path + (key,), role, order_counter)
        return
    if isinstance(spec, VariantSpec):
        _context_ref_edges(registry, edge_rows, edge_sources, edge_targets, value, spec.variants[value[spec.tag]], path, role, order_counter)
        return


def _objects(obs: dict[str, Any]) -> tuple[_NodeRegistry, list[list[float]], list[int], list[int], list[int], list[list[float]], list[int], list[int]]:
    actor = obs["acting_player"]
    p = obs["projection"]
    registry = _NodeRegistry(actor, p["turn"])
    edge_rows: list[list[float]] = []
    edge_sources: list[int] = []
    edge_targets: list[int] = []

    for i, card in enumerate(obs["own_hand"]):
        registry.add_card(_blank_public_from_ref(card["stable"]), "self_hand", i, "card")
    seat_order = [actor, "p1" if actor == "p0" else "p0"]
    for owner_order, seat in enumerate(seat_order):
        seat_idx = 0 if seat == "p0" else 1
        group = "known_library_self" if seat == actor else "known_library_opponent"
        for entry in obs["known_library_cards"][seat_idx]:
            position = entry["position"]
            node = registry.add_card(
                _blank_public_from_ref(entry["card"]["stable"]),
                group,
                position,
                "known_library",
            )
            _append_edge(
                edge_rows,
                edge_sources,
                edge_targets,
                node,
                node,
                "known_library",
                owner_order,
                position,
                extra=_seat_features(seat, actor),
            )
    for owner_order, seat in enumerate(seat_order):
        seat_idx = 0 if seat == "p0" else 1
        group = "known_hand_self" if seat == actor else "known_hand_opponent"
        for reveal_order, card in enumerate(obs["known_hand_cards"][seat_idx]):
            node = registry.add_card(
                _blank_public_from_ref(card["stable"]),
                group,
                reveal_order,
                "known_hand",
            )
            _append_edge(
                edge_rows,
                edge_sources,
                edge_targets,
                node,
                node,
                "known_hand",
                owner_order,
                reveal_order,
                extra=_seat_features(seat, actor),
            )
    for seat in seat_order:
        seat_idx = 0 if seat == "p0" else 1
        group = "self_battlefield" if seat == actor else "opponent_battlefield"
        for i, card in enumerate(p["battlefield"][seat_idx]):
            registry.add_card(card, group, i)
    for seat in seat_order:
        seat_idx = 0 if seat == "p0" else 1
        group = "self_graveyard" if seat == actor else "opponent_graveyard"
        for i, card in enumerate(p["graveyards"][seat_idx]):
            registry.add_card(card, group, i)
    for i, card in enumerate(p["exile"]):
        registry.add_card(card, "exile", i)
    for i, item in enumerate(p["stack"]):
        registry.add_ref_node(item["source"], "stack", i, "stack")
    historical_paid_order = 0
    for item in p["stack"]:
        for paid_ref in item["paid_cost_refs"]:
            registry.add_historical_ref_node(paid_ref, historical_paid_order)
            historical_paid_order += 1

    attachment_edges: list[tuple[int, int]] = []
    actor_relative_zones = []
    for seat in seat_order:
        seat_idx = 0 if seat == "p0" else 1
        actor_relative_zones.append(p["battlefield"][seat_idx])
    for seat in seat_order:
        seat_idx = 0 if seat == "p0" else 1
        actor_relative_zones.append(p["graveyards"][seat_idx])
    actor_relative_zones.append(p["exile"])
    for card in [c for zone in actor_relative_zones for c in zone]:
        host = registry.resolve(card["stable"])
        for attachment in [registry.resolve_attachment_arena(raw) for raw in card["attachments"]]:
            attachment_edges.append((host, attachment))
    for attach_order, (host, attachment) in enumerate(sorted(attachment_edges)):
        _append_edge(edge_rows, edge_sources, edge_targets, host, attachment, "attachment", attach_order)
    relation_edges: list[tuple[int, int, str]] = []
    for relation in p["object_relations"]:
        source = registry.resolve(relation["object"])
        kind = relation["relation_kind"]
        target = registry.resolve(relation["attached_to"] if kind == "attached_to" else relation["exiled_by"])
        relation_edges.append((source, target, kind))
    for relation_order, (source, target, kind) in enumerate(sorted(relation_edges, key=lambda edge: (edge[2], edge[0], edge[1]))):
        _append_edge(edge_rows, edge_sources, edge_targets, source, target, kind, relation_order)
    for i, item in enumerate(p["stack"]):
        source = registry.resolve(item["source"])
        for target_index, target in enumerate(item["targets"]):
            if target["target_kind"] == "object":
                target_node = registry.resolve(target["object"])
                _append_edge(edge_rows, edge_sources, edge_targets, source, target_node, "stack_target", i, target_index)
        for paid_order, paid_ref in enumerate(item["paid_cost_refs"]):
            paid_node = registry.resolve(paid_ref)
            _append_edge(edge_rows, edge_sources, edge_targets, source, paid_node, "paid_cost", i, paid_order)
    for i, ref in enumerate(p["combat"]["ordered_attackers"]):
        node = registry.resolve(ref)
        _append_edge(edge_rows, edge_sources, edge_targets, node, node, "combat_attacker", i)
    for attacker_order, pair in enumerate(p["combat"]["attacker_to_ordered_blockers"]):
        attacker, blockers = pair
        attacker_node = registry.resolve(attacker)
        for blocker_order, blocker in enumerate(blockers):
            blocker_node = registry.resolve(blocker)
            _append_edge(edge_rows, edge_sources, edge_targets, attacker_node, blocker_node, "combat_blocker", attacker_order, blocker_order)
    for effect_order, effect in enumerate(p["continuous_effects"]):
        affected = sorted(registry.resolve(ref) for ref in effect["affected_objects"])
        controller = effect["controller"]
        affected_players = effect["affected_players"]
        extra = [
            _number(effect["layers"], 16.0),
            _number(effect["power_delta"], 20.0),
            _number(effect["toughness_delta"], 20.0),
            _flag(effect["grants_haste"]),
        ] + _one_hot(effect["duration"], EFFECT_DURATIONS) + [
            _flag(effect["global"]),
        ] + _seat_features(controller, actor) + [
            1.0 if actor in affected_players else 0.0,
            1.0 if ("p1" if actor == "p0" else "p0") in affected_players else 0.0,
            1.0 if effect["set_power"] is not None else 0.0,
            _number(effect["set_power"], 20.0),
            1.0 if effect["set_toughness"] is not None else 0.0,
            _number(effect["set_toughness"], 20.0),
            _number(effect["add_color_mask"], 63.0),
            _number(effect["remove_color_mask"], 63.0),
            _number(effect["ward_generic_delta"], 16.0),
            _number(effect["minimum_blockers"], 8.0),
            _number(effect["prevent_damage_from_color_mask"], 63.0),
            _flag(effect["damage_cannot_be_prevented"]),
        ]
        source_node = registry.resolve(effect["source"]) if effect["source"] is not None else None
        if source_node is not None:
            _append_edge(edge_rows, edge_sources, edge_targets, source_node, source_node, "effect_source", effect_order, extra=extra)
        for affected_order, node in enumerate(affected):
            _append_edge(edge_rows, edge_sources, edge_targets, source_node if source_node is not None else node, node, "effect_affected", effect_order, affected_order, extra=extra)
    permission_edges: list[tuple[int, list[float]]] = []
    for permission in p["exile_play_permissions"]:
        obj = permission["object"]
        if permission["zone_change_generation"] != obj["zone_change_count"]:
            raise FeatureSchemaError("permission zone_change_generation does not match object incarnation")
        node = registry.resolve(obj)
        extra = _seat_features(permission["holder"], actor) + _one_hot(permission["play_or_cast"], PLAY_OR_CAST)
        expiry = permission["expiry"]
        extra += _one_hot(expiry["expiry_kind"], EXPIRY_KINDS)
        extra += [_flag(expiry.get("holder_turn_started", False))]
        permission_edges.append((node, extra))
    for permission_order, (node, extra) in enumerate(sorted(permission_edges, key=lambda item: (item[0], item[1]))):
        _append_edge(edge_rows, edge_sources, edge_targets, node, node, "permission", permission_order, extra=extra)
    engine = p["engine_context"]
    pending_order = [0]
    for key, spec, path in PENDING_CONTEXT_ROOTS:
        if engine[key] is not None:
            _context_ref_edges(registry, edge_rows, edge_sources, edge_targets, engine[key], spec, path, "pending_context", pending_order)
    surface = p["surface_context"]
    private_order = [0]
    for key, spec, path in PRIVATE_CONTEXT_ROOTS:
        if surface[key] is not None:
            _context_ref_edges(registry, edge_rows, edge_sources, edge_targets, surface[key], spec, path, "private_context", private_order)
    return registry, registry.rows, registry.tokens, registry.groups, registry.node_ids, edge_rows, edge_sources, edge_targets


def _object_feature_dim_probe() -> int:
    return OBJECT_FEATURE_DIM


def _action_kind(action: dict[str, Any]) -> str:
    return action["semantic"]["action_kind"]


def _semantic_actor(action: dict[str, Any]) -> str:
    return action["semantic"]["actor"]


def _action_card_refs(semantic: dict[str, Any], registry: _NodeRegistry) -> list[tuple[str, int, dict[str, Any], int, int]]:
    refs: list[tuple[str, int, dict[str, Any], int, int]] = []
    for role in ("source", "candidate", "card", "attacker"):
        if role in semantic:
            refs.append((role, 0, semantic[role], 0, registry.resolve(semantic[role])))
    if "target" in semantic and semantic["target"]["target_kind"] == "object":
        target = semantic["target"]["object"]
        refs.append(("target_object", 0, target, 0, registry.resolve(target)))
    for role in ("cards", "attackers", "blockers"):
        sorted_refs = sorted(((registry.resolve(ref), ref) for ref in semantic.get(role, [])), key=lambda item: item[0])
        for i, (node_id, ref) in enumerate(sorted_refs):
            refs.append((role, i, ref, 0, node_id))
    if semantic.get("action_kind") == "order_triggers":
        order = semantic["order"]
        for i, ref in enumerate(semantic["pending_sources"]):
            refs.append(("pending_sources", i, ref, int(order[i]), registry.resolve(ref)))
    return refs


def _action_ref_row(role: str, order_index: int, ref: dict[str, Any], actor: str, associated_order: int) -> tuple[list[float], int]:
    features = _one_hot(role, ACTION_REF_ROLES) + _card_ref_features(ref, actor) + [_number(order_index, 32.0), _number(associated_order, 32.0)]
    return features, _card_token(ref)


def _action_features(action: dict[str, Any], actor: str, registry: _NodeRegistry) -> tuple[list[float], list[list[float]], list[int], list[int]]:
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
        _number(semantic.get("option_index", 0), 16.0),
        _number(semantic.get("option_count", 0), 16.0),
        _flag(semantic.get("pay", False)),
        _flag(semantic.get("change_target", False)),
        _flag(semantic.get("use_cost", False)),
        _flag(semantic.get("cast_it", False)),
        _number(len(semantic.get("cards", [])), 8.0),
        _number(len(semantic.get("attackers", [])), 16.0),
        _number(len(semantic.get("blockers", [])), 16.0),
        _number(len(semantic.get("pending_sources", [])), 16.0),
        _number(len(semantic.get("order", [])), 16.0),
        _number(semantic.get("selected_count", 0), 16.0),
        _number(semantic.get("min_targets", 0), 16.0),
        _number(semantic.get("max_targets", 0), 16.0),
        _number(semantic.get("number", 0), 16.0),
        _number(semantic.get("minimum", 0), 16.0),
        _number(semantic.get("maximum", 0), 16.0),
        _flag(semantic.get("value", False)),
    ]
    mana_choice = semantic.get("mana_choice")
    features += [1.0 if mana_choice is not None else 0.0]
    features += _one_hot(mana_choice, MANA_COLORS) if mana_choice is not None else [0.0] * len(MANA_COLORS)
    color = semantic.get("color")
    features += _one_hot(color, MANA_COLORS) if color is not None else [0.0] * len(MANA_COLORS)
    features += _one_hot(semantic.get("mode", CAST_MODES[0]), CAST_MODES)
    features += _one_hot(semantic.get("cost_kind", COST_KINDS[0]), COST_KINDS)
    features += _one_hot(semantic.get("choice", OPTIONAL_COST_CHOICES[0]), OPTIONAL_COST_CHOICES)
    canonical = _canonical_model_value(action, LEGAL_ACTION_SPEC, ("legal_action",), _CanonicalContext(actor))
    features += _digest_features("legal-action", canonical, ACTION_HASH_DIM)
    ref_rows: list[list[float]] = []
    ref_tokens: list[int] = []
    ref_nodes: list[int] = []
    for role, order_index, ref, associated_order, node_id in _action_card_refs(semantic, registry):
        row, token = _action_ref_row(role, order_index, ref, actor, associated_order)
        ref_rows.append(row)
        ref_tokens.append(token)
        ref_nodes.append(node_id)
    return features, ref_rows, ref_tokens, ref_nodes


def _schema(state_dim: int, object_dim: int, edge_dim: int, action_dim: int, action_ref_dim: int) -> FeatureSchema:
    expected = (STATE_FEATURE_DIM, OBJECT_FEATURE_DIM, EDGE_FEATURE_DIM, ACTION_FEATURE_DIM, ACTION_REF_FEATURE_DIM)
    actual = (state_dim, object_dim, edge_dim, action_dim, action_ref_dim)
    if actual != expected:
        raise FeatureSchemaError(f"encoded schema dimensions drifted from contract: expected={expected} actual={actual}")
    return FeatureSchema(
        version=FEATURE_SCHEMA_VERSION,
        registry_version=FEATURE_REGISTRY_VERSION,
        contract_digest=feature_contract_fingerprint(),
        encoding_digest=encoding_contract_fingerprint(),
        state_dim=state_dim,
        object_feature_dim=object_dim,
        edge_feature_dim=edge_dim,
        action_feature_dim=action_dim,
        object_group_count=len(OBJECT_GROUPS),
        action_ref_feature_dim=action_ref_dim,
    )


def _pad_rows(rows: list[list[float]], width: int, name: str) -> list[list[float]]:
    out: list[list[float]] = []
    for row in rows:
        if len(row) > width:
            raise FeatureSchemaError(f"{name} row exceeds fixed contract width")
        out.append(row + [0.0] * (width - len(row)))
    return out


def _validate_order_trigger_semantic(semantic: dict[str, Any]) -> None:
    if semantic["action_kind"] != "order_triggers":
        return
    pending = semantic["pending_sources"]
    order = semantic["order"]
    if len(order) != len(pending):
        raise FeatureSchemaError("order_triggers.order length must match pending_sources")
    if sorted(order) != list(range(len(order))):
        raise FeatureSchemaError("order_triggers.order must be a permutation of pending source indexes")


def _require_ref(ref: dict[str, Any] | None, label: str) -> dict[str, Any]:
    if ref is None:
        raise FeatureSchemaError(f"{label} must be visible in this active Rust branch")
    return ref


def _require_ref_zone(ref: dict[str, Any] | None, label: str, zones: Iterable[str]) -> dict[str, Any]:
    value = _require_ref(ref, label)
    allowed = tuple(zones)
    if value["zone"] not in allowed:
        raise FeatureSchemaError(f"{label} has zone {value['zone']!r}, expected one of {allowed}")
    return value


def _require_ref_controller(ref: dict[str, Any], controller: str, label: str) -> None:
    if ref["controller"] != controller:
        raise FeatureSchemaError(f"{label} controller does not match Rust decision owner")


def _require_actor(actor: str, owner: str, label: str) -> None:
    if actor != owner:
        raise FeatureSchemaError(f"acting_player does not match Rust decision owner for {label}")


def _validate_engine_context(
    engine: dict[str, Any], actor: str, stack: list[dict[str, Any]]
) -> None:
    pending_keys = (
        "pending_cast",
        "pending_activation",
        "pending_discard",
        "pending_optional_cost",
        "pending_optional_cost_sacrifice",
        "pending_spell_copy",
        "pending_effect",
        "pending_triggers",
    )
    rust_stage_order = pending_keys
    present = []
    for key in rust_stage_order:
        if key == "pending_triggers":
            if engine[key]:
                present.append(key)
        elif engine[key] is not None:
            present.append(key)
    stage = engine["current_stage"]
    if stage in ("priority", "halted"):
        if present:
            raise FeatureSchemaError("engine priority/halted stage cannot carry pending contexts")
    else:
        expected_stage = present[0] if present else None
        if expected_stage != stage:
            raise FeatureSchemaError("engine current_stage does not match Rust pending-context priority")
        extras = present[1:]
        allowed_extras = {
            "pending_cast": {(), ("pending_discard",)},
            "pending_activation": {(), ("pending_discard",)},
            "pending_discard": {(), ("pending_triggers",)},
            "pending_optional_cost": {()},
            "pending_optional_cost_sacrifice": {()},
            "pending_spell_copy": {()},
            "pending_effect": {()},
            "pending_triggers": {()},
        }
        if tuple(extras) not in allowed_extras[stage]:
            raise FeatureSchemaError("engine pending contexts do not match a permitted Rust coexistence shape")
        if extras == ["pending_discard"]:
            expected_resume = "finish_cast" if stage == "pending_cast" else "finish_activation"
            if engine["pending_discard"]["resume_stage"] != expected_resume:
                raise FeatureSchemaError("pending_discard resume_stage does not match paused engine context")

    pending_cast = engine["pending_cast"]
    if pending_cast is not None:
        source = _require_ref_zone(pending_cast["source"], "pending_cast.source", ("Stack",))
        _require_ref_controller(source, pending_cast["controller"], "pending_cast.source")
    pending_activation = engine["pending_activation"]
    if pending_activation is not None:
        source = _require_ref_zone(pending_activation["source"], "pending_activation.source", ("Battlefield",))
        _require_ref_controller(source, pending_activation["controller"], "pending_activation.source")

    pending_discard = engine["pending_discard"]
    if pending_discard is not None:
        resume_stage = pending_discard["resume_stage"]
        resume_source = pending_discard["resume_source"]
        if resume_stage in ("none", "finish_cast", "finish_activation") and resume_source is not None:
            raise FeatureSchemaError("pending_discard resume_source must be absent for this resume_stage")
        if resume_stage in ("finish_spell_resolution", "finish_optional_cost") and resume_source is None:
            raise FeatureSchemaError("pending_discard resume_source is required for this resume_stage")
        if resume_source is not None and resume_source["zone"] != "Stack":
            raise FeatureSchemaError("pending_discard resume_source must still be in Stack until the deferred resume finishes")

    for key in ("pending_optional_cost", "pending_optional_cost_sacrifice"):
        pending = engine[key]
        if pending is None:
            continue
        source = _require_ref_zone(pending["source"], f"{key}.source", ("Stack",))
        _require_ref_controller(source, pending["player"], f"{key}.source")
        source_present = pending["spell_resume_source"] is not None
        zone_present = pending["spell_resume_zone"] is not None
        if source_present != zone_present:
            raise FeatureSchemaError(f"{key} spell_resume_source and spell_resume_zone must be both present or both absent")
        if pending["spell_resume_source"] is not None:
            spell_resume_source = _require_ref_zone(pending["spell_resume_source"], f"{key}.spell_resume_source", ("Stack",))
            _require_ref_controller(spell_resume_source, pending["player"], f"{key}.spell_resume_source")
    optional = engine["pending_optional_cost"]
    if optional is not None and not (optional["discard_payable"] or optional["sacrifice_payable"]):
        raise FeatureSchemaError("pending_optional_cost must have at least one payable branch")
    optional_sacrifice = engine["pending_optional_cost_sacrifice"]
    if optional_sacrifice is not None and len(optional_sacrifice["chosen"]) >= optional_sacrifice["remaining"]:
        raise FeatureSchemaError("pending_optional_cost_sacrifice must still need a land choice")

    pending_spell_copy = engine["pending_spell_copy"]
    if pending_spell_copy is not None:
        parent = _require_ref_zone(pending_spell_copy["parent"], "pending_spell_copy.parent", ("Stack",))
        stage = pending_spell_copy["stage"]
        copy = pending_spell_copy["copy"]
        if stage == "payment":
            if copy is not None:
                raise FeatureSchemaError("pending_spell_copy payment stage cannot already carry a copy")
        else:
            copy = _require_ref_zone(copy, "pending_spell_copy.copy", ("Stack",))
            _require_ref_controller(copy, pending_spell_copy["player"], "pending_spell_copy.copy")
            if copy["arena_id"] == parent["arena_id"]:
                raise FeatureSchemaError("pending_spell_copy copy must be distinct from its parent")

    pending_effect = engine["pending_effect"]
    if pending_effect is not None:
        source = _require_ref_zone(pending_effect["source"], "pending_effect.source", ("Stack",))
        _require_ref_controller(source, pending_effect["controller"], "pending_effect.source")
        if not stack or stack[-1]["source"] != source:
            raise FeatureSchemaError("pending_effect.source must be the top resolving stack item")
        choice = pending_effect["choice"]
        if choice is None:
            raise FeatureSchemaError("an observed pending_effect must be waiting for a choice")
        _require_actor(actor, choice["player"], "pending_effect")
        choice_kind = choice["choice_kind"]
        if choice_kind == "options":
            if choice["option_count"] < 2:
                raise FeatureSchemaError("pending effect option choice must expose at least two options")
        elif choice_kind == "targets":
            selected_count = len(choice["selected_targets"])
            if choice["min_targets"] > choice["max_targets"]:
                raise FeatureSchemaError("pending effect target choice min_targets exceeds max_targets")
            if selected_count > choice["max_targets"]:
                raise FeatureSchemaError("pending effect target choice selected_targets exceeds max_targets")
            if choice["can_finish"] != (selected_count >= choice["min_targets"]):
                raise FeatureSchemaError("pending effect target choice can_finish disagrees with its minimum")
            selected_keys = [_sort_key(target) for target in choice["selected_targets"]]
            legal_keys = [_sort_key(target) for target in choice["legal_targets"]]
            if len(selected_keys) != len(set(selected_keys)) or len(legal_keys) != len(set(legal_keys)):
                raise FeatureSchemaError("pending effect target choices must not repeat targets")
            if set(selected_keys).intersection(legal_keys):
                raise FeatureSchemaError("pending effect selected_targets and legal_targets must be disjoint")
            if selected_count + len(legal_keys) < choice["min_targets"]:
                raise FeatureSchemaError("pending effect target choice cannot still reach min_targets")
            if selected_count >= choice["max_targets"] and legal_keys:
                raise FeatureSchemaError("pending effect target choice cannot expose legal targets at max_targets")
        elif choice_kind == "color":
            if not choice["legal_colors"] or len(choice["legal_colors"]) != len(set(choice["legal_colors"])):
                raise FeatureSchemaError("pending effect color choice must expose unique legal colors")
        elif choice_kind == "number":
            if choice["minimum"] > choice["maximum"]:
                raise FeatureSchemaError("pending effect number choice minimum exceeds maximum")


def _validate_surface_context(surface: dict[str, Any], projection: dict[str, Any], actor: str) -> None:
    stage = surface["current_stage"]
    private_blockers = surface["private_blockers"]
    private_discard = surface["private_discard"]
    private_optional = surface["private_optional_cost"]
    present = [name for name, value in (("private_blockers", private_blockers), ("private_discard", private_discard), ("private_optional_cost", private_optional)) if value is not None]
    if stage == "priority":
        if present:
            raise FeatureSchemaError("surface priority stage cannot carry private contexts")
        return
    if stage == "declare_blockers_for_attacker":
        defender = "p1" if projection["active_player"] == "p0" else "p0"
        if actor != defender:
            raise FeatureSchemaError("declare_blockers private context leaked to non-defending actor")
        if present != ["private_blockers"]:
            raise FeatureSchemaError("declare_blockers stage requires only private_blockers")
        current_attacker = private_blockers["current_attacker"]
        if current_attacker is None:
            raise FeatureSchemaError("declare_blockers stage requires current_attacker")
        if current_attacker["controller"] != projection["active_player"]:
            raise FeatureSchemaError("declare_blockers current_attacker must be controlled by the active player")
        for blocker, attacker in private_blockers["accumulated"]:
            if attacker["controller"] != projection["active_player"] or blocker["controller"] != actor:
                raise FeatureSchemaError("declare_blockers accumulated tuple has impossible attacker/blocker ownership")
        for attacker, blockers in private_blockers["remaining"]:
            if attacker["controller"] != projection["active_player"]:
                raise FeatureSchemaError("declare_blockers remaining attacker must be controlled by the active player")
            for blocker in blockers:
                if blocker["controller"] != actor:
                    raise FeatureSchemaError("declare_blockers remaining blocker must be controlled by the defending actor")
        return
    if stage == "discard_pick":
        if present != ["private_discard"]:
            raise FeatureSchemaError("discard_pick stage requires only private_discard")
        if private_discard["remaining_needed"] <= 0:
            raise FeatureSchemaError("discard_pick remaining_needed must be positive")
        return
    if stage in ("optional_cost_use", "optional_cost_which"):
        if present != ["private_optional_cost"]:
            raise FeatureSchemaError("optional cost stage requires only private_optional_cost")
        if private_optional["stage"] != stage:
            raise FeatureSchemaError("private_optional_cost.stage must match surface current_stage")
        if stage == "optional_cost_which" and not (private_optional["discard_payable"] and private_optional["sacrifice_payable"]):
            raise FeatureSchemaError("optional_cost_which requires both branches to be payable")
        return
    raise FeatureSchemaError(f"unknown surface stage {stage!r}")


def _validate_engine_surface_tuple(projection: dict[str, Any], engine: dict[str, Any], surface: dict[str, Any], actor: str) -> None:
    engine_stage = engine["current_stage"]
    surface_stage = surface["current_stage"]
    if surface_stage == "priority":
        if engine_stage in ("priority", "halted"):
            return
        if engine_stage in ("pending_cast", "pending_activation") and engine["pending_discard"] is None:
            pending_key = engine_stage
            _require_actor(actor, engine[pending_key]["controller"], engine_stage)
            return
        if engine_stage == "pending_spell_copy":
            _require_actor(actor, engine["pending_spell_copy"]["player"], engine_stage)
            return
        if engine_stage == "pending_effect":
            choice = engine["pending_effect"]["choice"]
            if choice is None:
                raise FeatureSchemaError("pending_effect priority tuple requires an active choice")
            _require_actor(actor, choice["player"], engine_stage)
            return
        if engine_stage == "pending_optional_cost_sacrifice":
            _require_actor(actor, engine["pending_optional_cost_sacrifice"]["player"], engine_stage)
            return
        if engine_stage == "pending_triggers":
            pending_triggers = engine["pending_triggers"]
            if len(pending_triggers) < 2 or pending_triggers[1]["controller"] != pending_triggers[0]["controller"]:
                raise FeatureSchemaError("pending_triggers priority tuple requires a same-controller first trigger group")
            _require_actor(actor, pending_triggers[0]["controller"], engine_stage)
            return
        raise FeatureSchemaError("engine/surface tuple is impossible for Rust priority projection")

    if surface_stage == "declare_blockers_for_attacker":
        if engine_stage != "priority":
            raise FeatureSchemaError("blockers surface reshape must sit over engine priority")
        private_blockers = surface["private_blockers"]
        current_attacker = private_blockers["current_attacker"]
        if current_attacker is None:
            raise FeatureSchemaError("blockers surface reshape requires a current attacker")
        if current_attacker["controller"] != projection["active_player"]:
            raise FeatureSchemaError("blockers current attacker must match active player")
        _require_actor(actor, "p1" if current_attacker["controller"] == "p0" else "p0", surface_stage)
        return

    if surface_stage == "discard_pick":
        discard = engine["pending_discard"]
        if discard is None:
            raise FeatureSchemaError("discard surface reshape requires engine pending_discard")
        if actor != discard["player"]:
            raise FeatureSchemaError("discard surface actor must be the pending discard player")
        if engine_stage == "pending_cast":
            pending_cast = engine["pending_cast"]
            if pending_cast is None or discard["resume_stage"] != "finish_cast" or discard["player"] != pending_cast["controller"]:
                raise FeatureSchemaError("paused cast discard tuple does not match Rust FinishCast shape")
            return
        if engine_stage == "pending_activation":
            pending_activation = engine["pending_activation"]
            if pending_activation is None or discard["resume_stage"] != "finish_activation" or discard["player"] != pending_activation["controller"]:
                raise FeatureSchemaError("paused activation discard tuple does not match Rust FinishActivation shape")
            return
        if engine_stage == "pending_discard":
            if discard["resume_stage"] in ("finish_cast", "finish_activation"):
                raise FeatureSchemaError("FinishCast/FinishActivation discard must coexist with its paused engine context")
            return
        raise FeatureSchemaError("discard surface reshape cannot coexist with this engine stage")

    if surface_stage in ("optional_cost_use", "optional_cost_which"):
        optional = engine["pending_optional_cost"]
        private_optional = surface["private_optional_cost"]
        if engine_stage != "pending_optional_cost" or optional is None:
            raise FeatureSchemaError("optional-cost surface reshape requires engine pending_optional_cost")
        if actor != optional["player"]:
            raise FeatureSchemaError("optional-cost surface actor must be the pending optional-cost player")
        if private_optional["discard_payable"] != optional["discard_payable"] or private_optional["sacrifice_payable"] != optional["sacrifice_payable"]:
            raise FeatureSchemaError("surface optional-cost payable flags must match engine pending_optional_cost")
        if surface_stage == "optional_cost_which" and not (optional["discard_payable"] and optional["sacrifice_payable"]):
            raise FeatureSchemaError("optional_cost_which is only emitted when both costs are payable")
        return

    raise FeatureSchemaError(f"unknown surface stage {surface_stage!r}")


def _validate_observation_semantics(observation: dict[str, Any]) -> None:
    p = observation["projection"]
    turn = p["turn"]
    actor_index = 0 if observation["acting_player"] == "p0" else 1
    public_cards = [
        card
        for zone in p["battlefield"] + p["graveyards"] + [p["exile"]]
        for card in zone
    ]
    observed_node_keys = {
        _stable_key(card["stable"])
        for card in public_cards
    }
    observed_node_keys.update(_stable_key(card["stable"]) for card in observation["own_hand"])
    observed_node_keys.update(
        _stable_key(entry["card"]["stable"])
        for owner_entries in observation["known_library_cards"]
        for entry in owner_entries
    )
    observed_node_keys.update(
        _stable_key(card["stable"])
        for owner_entries in observation["known_hand_cards"]
        for card in owner_entries
    )
    identity_by_key: dict[tuple[int, int], tuple[int, str, str, str]] = {}
    for ref in _iter_card_refs_by_schema(observation, OBSERVATION_SPEC):
        key = _stable_key(ref)
        identity = _stable_identity(ref)
        previous = identity_by_key.get(key)
        if previous is not None and previous != identity:
            raise FeatureSchemaError("inconsistent repeated stable reference for same incarnation")
        identity_by_key[key] = identity
    for card in public_cards:
        characteristics = card["characteristics"]
        keywords = characteristics["effective_keywords"]
        if characteristics["effective_color_mask"] >= (1 << len(MANA_COLORS)):
            raise FeatureSchemaError("effective_color_mask uses bits outside the stable WUBRGC mask")
        if keywords["landwalk_mask"] >= (1 << len(MANA_COLORS)):
            raise FeatureSchemaError("landwalk_mask uses bits outside the stable WUBRGC mask")
        subtype_ids = characteristics["effective_subtype_ids"]
        if subtype_ids != sorted(set(subtype_ids)):
            raise FeatureSchemaError("effective_subtype_ids must be sorted and unique")
        uses = card["ability_uses_this_turn"]
        use_keys = [(entry["ability_kind"], entry["ability_index"]) for entry in uses]
        kind_order = {"mana": 0, "activated": 1}
        sorted_keys = sorted(set(use_keys), key=lambda key: (kind_order[key[0]], key[1]))
        if use_keys != sorted_keys or any(entry["uses"] == 0 for entry in uses):
            raise FeatureSchemaError("ability_uses_this_turn must have sorted unique kind/index keys and positive uses")
        goads = card["goaded_by"]
        goad_players = [entry["player"] for entry in goads]
        if goad_players != sorted(set(goad_players), key=SEATS.index):
            raise FeatureSchemaError("goaded_by must be sorted and unique by player")
        if any(entry["expires_at_turn"] < turn for entry in goads):
            raise FeatureSchemaError("goaded_by expiry cannot precede projection.turn")
        for field in ("plotted_turn", "entered_battlefield_turn"):
            value = card[field]
            if value is not None and value > turn:
                raise FeatureSchemaError(f"{field} cannot be in the future relative to projection.turn")
    for status in p["player_status"]:
        dungeon = status["dungeon"]
        if (dungeon["dungeon_id"] is None) != (dungeon["room_id"] is None):
            raise FeatureSchemaError("dungeon_id and room_id must be present or absent together")
        completed = dungeon["completed_dungeons"]
        if completed != sorted(set(completed)):
            raise FeatureSchemaError("completed_dungeons must be sorted and unique")
    stack_by_key: dict[tuple[int, int], tuple[int, dict[str, Any]]] = {}
    for i, item in enumerate(p["stack"]):
        if item["stack_index"] != i:
            raise FeatureSchemaError("stack_index must match recorded stack position")
        if item["is_copy"] and item["stack_item_kind"] != "spell":
            raise FeatureSchemaError("only spell stack items may be marked as copies")
        key = _stable_key(item["source"])
        if key in stack_by_key:
            raise FeatureSchemaError("stack contains duplicate stable object incarnations")
        stack_by_key[key] = (i, item)
        observed_node_keys.add(key)
        if item["stack_item_kind"] == "spell":
            if item["cast_method"] is None:
                raise FeatureSchemaError("spell stack items must expose cast_method")
        elif item["cast_method"] is not None:
            raise FeatureSchemaError("non-spell stack items cannot expose cast_method")
        paid_keys = [_stable_key(ref) for ref in item["paid_cost_refs"]]
        if len(paid_keys) != len(set(paid_keys)):
            raise FeatureSchemaError("paid_cost_refs must not repeat an object incarnation")
    _validate_engine_context(p["engine_context"], observation["acting_player"], p["stack"])
    for owner_index, entries in enumerate(observation["known_library_cards"]):
        owner = "p0" if owner_index == 0 else "p1"
        positions: list[int] = []
        identities: set[tuple[int, int]] = set()
        for entry in entries:
            position = entry["position"]
            ref = entry["card"]["stable"]
            if ref["zone"] != "Library" or ref["owner"] != owner:
                raise FeatureSchemaError("known library card must belong to the indexed owner and remain in Library")
            if position >= p["library_counts"][owner_index]:
                raise FeatureSchemaError("known library position is outside the visible library count")
            if _stable_key(ref) in identities:
                raise FeatureSchemaError("known library knowledge repeats one object incarnation")
            identities.add(_stable_key(ref))
            positions.append(position)
        if positions != sorted(set(positions)):
            raise FeatureSchemaError("known library positions must be unique and strictly sorted")
    if observation["known_hand_cards"][actor_index]:
        raise FeatureSchemaError("known_hand_cards must not duplicate the acting player's own_hand")
    for owner_index, entries in enumerate(observation["known_hand_cards"]):
        owner = "p0" if owner_index == 0 else "p1"
        identities: set[tuple[int, int]] = set()
        for card in entries:
            ref = card["stable"]
            if ref["zone"] != "Hand" or ref["owner"] != owner:
                raise FeatureSchemaError("known hand card must belong to the indexed owner and remain in Hand")
            key = _stable_key(ref)
            if key in identities:
                raise FeatureSchemaError("known hand knowledge repeats one object incarnation")
            identities.add(key)
        if len(entries) > p["hand_counts"][owner_index]:
            raise FeatureSchemaError("known hand identities exceed the visible hand count")
    relation_keys: set[str] = set()
    for relation in p["object_relations"]:
        kind = relation["relation_kind"]
        object_key = _stable_key(relation["object"])
        related_key = _stable_key(relation["attached_to"] if kind == "attached_to" else relation["exiled_by"])
        if object_key not in observed_node_keys or related_key not in observed_node_keys:
            raise FeatureSchemaError("object relation endpoints must resolve to observed object nodes")
        relation_key = _sort_key(relation)
        if relation_key in relation_keys:
            raise FeatureSchemaError("object_relations must not repeat a relation")
        relation_keys.add(relation_key)
    for effect in p["continuous_effects"]:
        for field in ("add_color_mask", "remove_color_mask", "add_landwalk_mask", "remove_landwalk_mask", "prevent_damage_from_color_mask"):
            if effect[field] >= (1 << len(MANA_COLORS)):
                raise FeatureSchemaError(f"{field} uses bits outside the stable WUBRGC mask")
        for field in ("add_subtype_ids", "remove_subtype_ids"):
            values = effect[field]
            if values != sorted(set(values)):
                raise FeatureSchemaError(f"{field} must be sorted and unique")
        if effect["source"] is not None and _stable_key(effect["source"]) not in observed_node_keys:
            raise FeatureSchemaError("continuous effect source must resolve to an observed object node")
        if any(_stable_key(ref) not in observed_node_keys for ref in effect["affected_objects"]):
            raise FeatureSchemaError("continuous effect affected_objects must resolve to observed object nodes")
        affected_players = effect["affected_players"]
        if affected_players != sorted(set(affected_players), key=SEATS.index):
            raise FeatureSchemaError("continuous effect affected_players must be sorted and unique")
        if not effect["global"] and not effect["affected_objects"] and not affected_players:
            raise FeatureSchemaError("continuous effect must declare an object, player, or global scope")
    pending_spell_copy = p["engine_context"]["pending_spell_copy"]
    if pending_spell_copy is not None:
        parent = _require_ref(pending_spell_copy["parent"], "pending_spell_copy.parent")
        parent_entry = stack_by_key.get(_stable_key(parent))
        if parent_entry is None or parent_entry[1]["stack_item_kind"] != "spell":
            raise FeatureSchemaError("pending_spell_copy parent must resolve to a live spell on the stack")
        parent_index, parent_item = parent_entry
        if parent_item["targets"] != [pending_spell_copy["inherited_target"]]:
            raise FeatureSchemaError("pending_spell_copy inherited_target must match its parent spell target")
        if pending_spell_copy["stage"] == "payment":
            if parent_index != len(p["stack"]) - 1:
                raise FeatureSchemaError("pending_spell_copy payment parent must be the top stack item")
        else:
            copy = _require_ref(pending_spell_copy["copy"], "pending_spell_copy.copy")
            copy_entry = stack_by_key.get(_stable_key(copy))
            if copy_entry is None:
                raise FeatureSchemaError("pending_spell_copy copy must resolve to a live stack item")
            copy_index, copy_item = copy_entry
            if not copy_item["is_copy"] or copy_item["stack_item_kind"] != "spell":
                raise FeatureSchemaError("pending_spell_copy copy must resolve to a copied spell")
            if copy_item["controller"] != pending_spell_copy["player"]:
                raise FeatureSchemaError("pending_spell_copy stack-item controller must match the copying player")
            if copy_item["targets"] != [pending_spell_copy["inherited_target"]]:
                raise FeatureSchemaError("pending_spell_copy inherited_target must match the copied spell target")
            if copy_index != len(p["stack"]) - 1 or copy_index != parent_index + 1:
                raise FeatureSchemaError("pending_spell_copy copy must be directly above its parent on the stack")
    pending_effect = p["engine_context"]["pending_effect"]
    if pending_effect is not None:
        source = _require_ref(pending_effect["source"], "pending_effect.source")
        entry = stack_by_key.get(_stable_key(source))
        if entry is None or entry[0] != len(p["stack"]) - 1:
            raise FeatureSchemaError("pending_effect source must be the top public stack item")
    _validate_surface_context(p["surface_context"], p, observation["acting_player"])
    _validate_engine_surface_tuple(p, p["engine_context"], p["surface_context"], observation["acting_player"])
    for permission in p["exile_play_permissions"]:
        if permission["zone_change_generation"] != permission["object"]["zone_change_count"]:
            raise FeatureSchemaError("permission zone_change_generation does not match object incarnation")


def assert_observation_classified(observation: dict[str, Any]) -> None:
    OBSERVATION_SPEC.validate(observation, ("observation",))
    _validate_observation_semantics(observation)


def assert_action_classified(action: dict[str, Any]) -> None:
    LEGAL_ACTION_SPEC.validate(action, ("legal_action",))
    semantic = action["semantic"]
    if semantic["action_kind"] == "choose_spell_mode" and semantic["mode_index"] >= semantic["mode_count"]:
        raise FeatureSchemaError("choose_spell_mode.mode_index must be < mode_count")
    if semantic["action_kind"] == "choose_spell_mode" and semantic["mode_count"] <= 0:
        raise FeatureSchemaError("choose_spell_mode.mode_count must be positive")
    if semantic["action_kind"] == "choose_effect_option":
        if semantic["option_count"] < 2:
            raise FeatureSchemaError("choose_effect_option.option_count must be at least two")
        if semantic["option_index"] >= semantic["option_count"]:
            raise FeatureSchemaError("choose_effect_option.option_index must be < option_count")
    if semantic["action_kind"] == "choose_effect_target":
        if semantic["min_targets"] > semantic["max_targets"]:
            raise FeatureSchemaError("choose_effect_target.min_targets must be <= max_targets")
        if semantic["selected_count"] >= semantic["max_targets"]:
            raise FeatureSchemaError("choose_effect_target must leave room for the selected target")
    if semantic["action_kind"] == "choose_effect_number":
        if semantic["minimum"] > semantic["maximum"]:
            raise FeatureSchemaError("choose_effect_number.minimum must be <= maximum")
        if not semantic["minimum"] <= semantic["number"] <= semantic["maximum"]:
            raise FeatureSchemaError("choose_effect_number.number must be inside its legal range")
    _validate_order_trigger_semantic(semantic)


def validate_legal_actions_contract(actions: Any, acting_player: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(actions, list):
        raise FeatureSchemaError("legal_actions must be a list")
    if not actions:
        raise FeatureSchemaError("decision has no legal actions")
    seen_stable: set[str] = set()
    for i, action in enumerate(actions):
        assert_action_classified(action)
        if action["schema_version"] != 4:
            raise FeatureSchemaError("legal action schema mismatch")
        if action["selected_index"] != i:
            raise FeatureSchemaError("legal action selected_index is not contiguous")
        if not action["stable_id"].startswith("legal-action-v4:"):
            raise FeatureSchemaError("legal action stable_id must use the legal-action-v4 prefix")
        if action["stable_id"] in seen_stable:
            raise FeatureSchemaError("duplicate legal action stable_id")
        seen_stable.add(action["stable_id"])
        if acting_player is not None and _semantic_actor(action) != acting_player:
            raise FeatureSchemaError("legal action actor does not match acting_player")
    return actions


def _validate_spell_copy_legal_actions(
    observation: dict[str, Any], actions: list[dict[str, Any]]
) -> None:
    pending = observation["projection"]["engine_context"]["pending_spell_copy"]
    if pending is None:
        return
    stage = pending["stage"]
    if stage == "payment":
        expected_kind = "choose_spell_copy_payment"
        expected_source = _require_ref(pending["parent"], "pending_spell_copy.parent")
        expected_flags = [True, False]
        flag_key = "pay"
    elif stage == "retarget":
        expected_kind = "choose_spell_copy_retarget"
        expected_source = _require_ref(pending["copy"], "pending_spell_copy.copy")
        expected_flags = [True, False]
        flag_key = "change_target"
    else:
        expected_kind = "choose_target"
        expected_source = _require_ref(pending["copy"], "pending_spell_copy.copy")
        expected_flags = None
        flag_key = None

    for action in actions:
        semantic = action["semantic"]
        if semantic["action_kind"] != expected_kind:
            raise FeatureSchemaError(
                f"pending_spell_copy {stage} stage requires only {expected_kind} legal actions"
            )
        if semantic["source"] != expected_source:
            raise FeatureSchemaError(
                "pending_spell_copy legal action source does not match the active parent/copy"
            )
        if stage == "target" and semantic["remaining"] != 1:
            raise FeatureSchemaError("pending_spell_copy target action must choose its one target")

    if expected_flags is not None:
        actual_flags = [action["semantic"][flag_key] for action in actions]
        if actual_flags != expected_flags:
            raise FeatureSchemaError(
                f"pending_spell_copy {stage} actions must be ordered yes then no"
            )


def _validate_pending_effect_legal_actions(
    observation: dict[str, Any], actions: list[dict[str, Any]]
) -> None:
    pending = observation["projection"]["engine_context"]["pending_effect"]
    if pending is None:
        return

    choice = pending["choice"]
    source = _require_ref(pending["source"], "pending_effect.source")
    actor = observation["acting_player"]
    choice_kind = choice["choice_kind"]
    if choice_kind == "options":
        option_count = choice["option_count"]
        if len(actions) != option_count:
            raise FeatureSchemaError(
                "pending effect option actions must correspond one-for-one to every option"
            )
        for option_index, action in enumerate(actions):
            semantic = action["semantic"]
            if semantic["action_kind"] != "choose_effect_option":
                raise FeatureSchemaError(
                    "pending effect option choice requires only choose_effect_option actions"
                )
            if semantic["actor"] != actor or semantic["source"] != source:
                raise FeatureSchemaError(
                    "pending effect option action actor/source does not match the active choice"
                )
            if semantic["option_count"] != option_count:
                raise FeatureSchemaError(
                    "pending effect option action count does not match the active choice"
                )
            if semantic["option_index"] != option_index:
                raise FeatureSchemaError(
                    "pending effect option actions must enumerate every option in order"
                )
        return
    if choice_kind != "targets":
        return

    selected_count = len(choice["selected_targets"])
    target_actions: list[dict[str, Any]] = []
    finish_actions: list[dict[str, Any]] = []

    for action in actions:
        semantic = action["semantic"]
        kind = semantic["action_kind"]
        if kind == "choose_effect_target":
            target_actions.append(semantic)
            if semantic["actor"] != actor or semantic["source"] != source:
                raise FeatureSchemaError(
                    "pending effect target action actor/source does not match the active choice"
                )
            if semantic["selected_count"] != selected_count:
                raise FeatureSchemaError(
                    "pending effect target action selected_count does not match selected_targets"
                )
            if semantic["min_targets"] != choice["min_targets"] or semantic["max_targets"] != choice["max_targets"]:
                raise FeatureSchemaError(
                    "pending effect target action bounds do not match the active choice"
                )
        elif kind == "finish_effect_selection":
            finish_actions.append(semantic)
            if semantic["actor"] != actor or semantic["source"] != source:
                raise FeatureSchemaError(
                    "pending effect finish action actor/source does not match the active choice"
                )
            if semantic["selected_count"] != selected_count:
                raise FeatureSchemaError(
                    "pending effect finish action selected_count does not match selected_targets"
                )
        else:
            raise FeatureSchemaError(
                "pending effect target choice requires only choose_effect_target and finish_effect_selection actions"
            )

    actual_targets = [semantic["target"] for semantic in target_actions]
    if actual_targets != choice["legal_targets"]:
        raise FeatureSchemaError(
            "pending effect target actions must correspond one-for-one, in order, to legal_targets"
        )
    expected_finish_count = 1 if choice["can_finish"] else 0
    if len(finish_actions) != expected_finish_count:
        raise FeatureSchemaError(
            "pending effect finish action must exist exactly when can_finish is true"
        )
    expected_kinds = ["choose_effect_target"] * len(choice["legal_targets"])
    if choice["can_finish"]:
        expected_kinds.append("finish_effect_selection")
    if [action["semantic"]["action_kind"] for action in actions] != expected_kinds:
        raise FeatureSchemaError(
            "pending effect target actions must precede the optional finish action"
        )


def encode_decision(observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> EncodedDecision:
    assert_observation_classified(observation)
    if observation["schema_version"] != 4:
        raise FeatureSchemaError("observation schema mismatch")
    validate_legal_actions_contract(legal_actions, observation["acting_player"])
    _validate_spell_copy_legal_actions(observation, legal_actions)
    _validate_pending_effect_legal_actions(observation, legal_actions)
    actor = observation["acting_player"]
    state = _state_features(observation)
    registry, object_rows, object_tokens, object_groups, object_node_ids, edge_rows, edge_sources, edge_targets = _objects(observation)
    action_rows: list[list[float]] = []
    action_ref_rows: list[list[float]] = []
    action_ref_tokens: list[int] = []
    action_ref_indices: list[int] = []
    action_ref_nodes: list[int] = []
    for action_index, action in enumerate(legal_actions):
        row, ref_rows, ref_tokens, ref_nodes = _action_features(action, actor, registry)
        action_rows.append(row)
        for row_ref, token_ref, node_ref in zip(ref_rows, ref_tokens, ref_nodes):
            action_ref_rows.append(row_ref)
            action_ref_tokens.append(token_ref)
            action_ref_indices.append(action_index)
            action_ref_nodes.append(node_ref)
    object_dim = OBJECT_FEATURE_DIM
    edge_dim = EDGE_FEATURE_DIM
    action_dim = ACTION_FEATURE_DIM
    action_ref_dim = ACTION_REF_FEATURE_DIM
    object_rows = _pad_rows(object_rows, object_dim, "object_features")
    edge_rows = _pad_rows(edge_rows, edge_dim, "edge_features")
    action_rows = _pad_rows(action_rows, action_dim, "action_features")
    action_ref_rows = _pad_rows(action_ref_rows, action_ref_dim, "action_ref_features")
    if not object_rows:
        object_rows = [[0.0] * object_dim]
        object_tokens = [0]
        object_groups = [0]
        object_node_ids = [0]
    if not edge_rows:
        edge_rows_tensor = torch.empty((0, edge_dim), dtype=torch.float32)
        edge_source_tensor = torch.empty((0,), dtype=torch.long)
        edge_target_tensor = torch.empty((0,), dtype=torch.long)
    else:
        edge_rows_tensor = torch.tensor(edge_rows, dtype=torch.float32)
        edge_source_tensor = torch.tensor(edge_sources, dtype=torch.long)
        edge_target_tensor = torch.tensor(edge_targets, dtype=torch.long)
    if not action_ref_rows:
        action_ref_rows = torch.empty((0, action_ref_dim), dtype=torch.float32)
        action_ref_tokens_tensor = torch.empty((0,), dtype=torch.long)
        action_ref_indices_tensor = torch.empty((0,), dtype=torch.long)
        action_ref_nodes_tensor = torch.empty((0,), dtype=torch.long)
    else:
        action_ref_rows = torch.tensor(action_ref_rows, dtype=torch.float32)
        action_ref_tokens_tensor = torch.tensor(action_ref_tokens, dtype=torch.long)
        action_ref_indices_tensor = torch.tensor(action_ref_indices, dtype=torch.long)
        action_ref_nodes_tensor = torch.tensor(action_ref_nodes, dtype=torch.long)
    schema = _schema(len(state), object_dim, edge_dim, action_dim, action_ref_dim)
    return EncodedDecision(
        schema=schema,
        state=torch.tensor(state, dtype=torch.float32),
        object_features=torch.tensor(object_rows, dtype=torch.float32),
        object_card_ids=torch.tensor(object_tokens, dtype=torch.long),
        object_groups=torch.tensor(object_groups, dtype=torch.long),
        object_node_ids=torch.tensor(object_node_ids, dtype=torch.long),
        edge_features=edge_rows_tensor,
        edge_source_indices=edge_source_tensor,
        edge_target_indices=edge_target_tensor,
        action_features=torch.tensor(action_rows, dtype=torch.float32),
        action_ref_features=action_ref_rows if isinstance(action_ref_rows, torch.Tensor) else torch.tensor(action_ref_rows, dtype=torch.float32),
        action_ref_card_ids=action_ref_tokens_tensor,
        action_ref_action_indices=action_ref_indices_tensor,
        action_ref_node_indices=action_ref_nodes_tensor,
    )


def every_action_variant_fixture(base_ref: dict[str, Any], target_ref: dict[str, Any] | None = None, second_ref: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    target_player = {"target_kind": "player", "player": "p1"}
    if target_ref is None:
        target_ref = {**base_ref, "arena_id": base_ref["arena_id"] + 1, "card_db_id": base_ref["card_db_id"] + 1}
    if second_ref is None:
        second_ref = {**base_ref, "arena_id": base_ref["arena_id"] + 2, "card_db_id": base_ref["card_db_id"] + 2}
    target_object = {"target_kind": "object", "object": target_ref}
    return [
        {"schema_version": 4, "selected_index": i, "stable_id": f"legal-action-v4:fixture-{i}", "display_text": f"text-{i}", "semantic": semantic}
        for i, semantic in enumerate(
            [
                {"action_kind": "pass", "actor": "p0"},
                {"action_kind": "play_land", "actor": "p0", "source": base_ref},
                {"action_kind": "cast_spell", "actor": "p0", "source": base_ref},
                {"action_kind": "activate_mana_ability", "actor": "p0", "source": base_ref, "mana_choice": "R"},
                {"action_kind": "activate_ability", "actor": "p0", "source": base_ref, "ability_index": 1},
                {"action_kind": "plot_spell", "actor": "p0", "source": base_ref},
                {"action_kind": "choose_target", "actor": "p0", "source": base_ref, "remaining": 1, "target": target_player},
                {"action_kind": "choose_target", "actor": "p0", "source": base_ref, "remaining": 1, "target": target_object},
                {"action_kind": "choose_cost_target", "actor": "p0", "source": base_ref, "cost_kind": "SacrificeLands", "remaining": 1, "candidate": second_ref},
                {"action_kind": "choose_cast_mode", "actor": "p0", "source": base_ref, "mode": "Normal"},
                {"action_kind": "choose_kicker", "actor": "p0", "source": base_ref, "pay": True},
                {"action_kind": "choose_spell_mode", "actor": "p0", "source": base_ref, "mode_index": 0, "mode_count": 2},
                {"action_kind": "choose_effect_option", "actor": "p0", "source": base_ref, "option_index": 1, "option_count": 2},
                {"action_kind": "choose_effect_target", "actor": "p0", "source": base_ref, "target": target_object, "selected_count": 0, "min_targets": 1, "max_targets": 2},
                {"action_kind": "finish_effect_selection", "actor": "p0", "source": base_ref, "selected_count": 1},
                {"action_kind": "choose_effect_color", "actor": "p0", "source": base_ref, "color": "R"},
                {"action_kind": "choose_effect_number", "actor": "p0", "source": base_ref, "number": 2, "minimum": -1, "maximum": 3},
                {"action_kind": "choose_effect_boolean", "actor": "p0", "source": base_ref, "value": True},
                {"action_kind": "finish_target_selection", "actor": "p0", "source": base_ref, "selected_count": 1},
                {"action_kind": "choose_optional_cost_use", "actor": "p0", "use_cost": True},
                {"action_kind": "choose_optional_cost_which", "actor": "p0", "choice": "Discard"},
                {"action_kind": "choose_spell_copy_payment", "actor": "p0", "source": base_ref, "pay": True},
                {"action_kind": "choose_spell_copy_retarget", "actor": "p0", "source": base_ref, "change_target": True},
                {"action_kind": "choose_madness_cast", "actor": "p0", "card": base_ref, "cast_it": True},
                {"action_kind": "discard", "actor": "p0", "cards": [base_ref, second_ref]},
                {"action_kind": "declare_attackers", "actor": "p0", "attackers": [base_ref, second_ref]},
                {"action_kind": "declare_blockers_for_attacker", "actor": "p0", "attacker": base_ref, "blockers": [second_ref]},
                {"action_kind": "order_triggers", "actor": "p0", "pending_sources": [base_ref, second_ref], "order": [1, 0]},
            ]
        )
    ]
