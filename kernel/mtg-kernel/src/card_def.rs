//! Static card database. `CARD_DEFS` (and `card_id_by_name`,
//! `KERNEL_CARDDB_HASH`) are generated at build time by `build.rs` from
//! `kernel/data/cards_v1.json` -- see that file for the codegen and its
//! validation (duplicate names / empty deck coverage / schema-version
//! mismatch all fail the build).
//!
//! Mono-Red Burn's 21 cards all carry a real `spell_effect`/`mana_ability`
//! program as of this increment: basic Mountain, the 4 "N damage to any
//! target" burn spells (Lightning Bolt, Fiery Temper, Fireblast, Lava
//! Dart), the 4 creatures as vanilla bodies (Guttersnipe, Masked Meower,
//! Voldaren Epicure, Sneaky Snacker -- keyword abilities/triggers ignored,
//! they're just a castable body), Faithless Looting, Grab the Prize,
//! Highway Robbery (+ Plot), Fiery Temper's Madness, Searing Blaze
//! (landfall + 2 related targets), and Pyroblast/Red Elemental Blast
//! (modal, color-checked counter/destroy). Relic of Progenitus is the one
//! remaining deferred card -- graveyard-card targeting doesn't fit any
//! existing `TargetSpec` shape and is sideboard-only, so it's lower
//! priority; see `still_deferred_burn_cards_are_out_of_scope_this_increment`.
//! Definitions without an explicit registry `engine_capability` remain
//! `NoEffect`. Supported ordinary permanents and intrinsic basic-land mana
//! are generated from metadata only after that capability gate, so registry
//! metadata alone can never make a card playable.
//!
//! Mono Red Rally's 18 cards (6 shared with Burn: Lightning Bolt, Mountain,
//! Red Elemental Blast, Relic of Progenitus, Searing Blaze, Voldaren
//! Epicure) are implemented as of the Rally increment: Burning-Tree
//! Emissary (ETB mana), Chain Lightning (mandatory damage plus the complete
//! recursive pay/copy/retarget loop -- see `effect::EffectOp::
//! OfferAffectedPlayerSpellCopy` and `engine::PendingSpellCopy`), Clockwork
//! Percussionist
//! (haste + dies-trigger impulse draw), End the
//! Festivities (mass damage to the opponent + their creatures), Experimental
//! Synthesizer (ETB/leaves impulse draw + sac-for-a-token ability), Galvanic
//! Blast (Metalcraft), Goblin Bushwhacker (Kicker-gated team pump/haste),
//! Goblin Tomb Raider (static self-boost), Great Furnace (a second Mountain),
//! Rally at the Hornburg (tokens + Human haste), and Reckless Impulse
//! (impulse draw). Cast into the Fire remains deferred (sideboard-only,
//! modal with a 0-2 variable-count target mode this kernel's `TargetSpec`
//! shape doesn't support) -- see `local-training/kernel_oracle/rally/
//! coverage_ledger.md` for the full per-card ledger.

use crate::effect::{
    CreatureFilter, EffectCond, EffectOp, ImpulseDuration, ObjectRef, PlayerRef, TargetRef,
};
use crate::mana::{Cost, ManaColor, Pip};
use crate::state::Zone;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardType {
    Land,
    Creature,
    Instant,
    Sorcery,
    Artifact,
    Enchantment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Supertype {
    Basic,
    Snow,
}

/// Fail-closed engine readiness for one registry definition. This is
/// generated from `cards_v1.json` and is deliberately independent of card
/// type: a fully supported land is executable even though it is played, not
/// cast; a token can be fully supported even though it can only be created
/// by another effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardCapability {
    /// No executable program may be exposed by the engine.
    NoEffect,
    /// Some behavior is executable, but a full-deck preflight must reject
    /// the definition because a reachable branch is still unsupported.
    Partial,
    /// Every reachable in-scope branch is implemented.
    Full,
}

impl CardCapability {
    pub const fn is_executable(self) -> bool {
        !matches!(self, CardCapability::NoEffect)
    }

    pub const fn is_fully_supported(self) -> bool {
        matches!(self, CardCapability::Full)
    }
}

/// 105.1's subtype line, typed (per external review: "subtype queries
/// structured, typed access, not string contains"). A fully closed set --
/// one named variant per distinct subtype string across the whole
/// 135-card pool -- rather than a named-variants-plus-string-fallback
/// design: `Subtype` is embedded in `effect::CreatureFilter` /
/// `effect::EffectOp`, which need to derive `Deserialize`, and a
/// `&'static str` payload (needed for a fallback variant to round-trip
/// arbitrary text) can't implement that (same reason `mana::Cost` doesn't
/// derive `Serialize`/`Deserialize` either -- see its own doc). A query
/// against a named variant (`Subtype::Human`) is a typed enum comparison
/// that can never silently match the wrong thing via a typo or a
/// differently-cased duplicate -- this pool's own JSON data has real
/// examples of the latter (`"Human"` and `"HUMAN"` on different cards,
/// likely an ingestion artifact upstream of this codegen, not a meaningful
/// distinction -- preserved as two distinct variants here rather than
/// silently merged, so this table stays a faithful mirror of the source
/// data). `build.rs::subtype_variant` panics on an unrecognized string,
/// same as `card_type_variant`/`supertype_variant`/`color_variant`, since
/// this is now a closed set the same way those are.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Subtype {
    Ape,
    Aura,
    /// "BIRD" verbatim -- see the module doc's note on case-duplicated
    /// subtype strings.
    BirdAllCaps,
    Bird,
    Blood,
    Cat,
    Detective,
    Dragon,
    Drone,
    Druid,
    Eldrazi,
    Elf,
    Equipment,
    /// "FAERIE" verbatim -- see the module doc's note.
    FaerieAllCaps,
    Faerie,
    Food,
    Forest,
    Gate,
    Goblin,
    /// "HUMAN" verbatim -- see the module doc's note.
    HumanAllCaps,
    Hero,
    Human,
    Hydra,
    Island,
    Knight,
    /// "MONK" verbatim.
    Monk,
    /// "MOONFOLK" verbatim.
    Moonfolk,
    Monkey,
    Mountain,
    Myr,
    /// "NINJA" verbatim -- see the module doc's note.
    NinjaAllCaps,
    Ninja,
    Ouphe,
    Pirate,
    Plains,
    /// "ROGUE" verbatim -- see the module doc's note.
    RogueAllCaps,
    Ranger,
    Rat,
    Rogue,
    /// "SERPENT" verbatim.
    Serpent,
    Saga,
    Samurai,
    Shaman,
    Shapeshifter,
    Soldier,
    Spider,
    Spirit,
    Swamp,
    Toy,
    Treefolk,
    Vampire,
    /// "WIZARD" verbatim -- see the module doc's note.
    WizardAllCaps,
    Warrior,
    Wizard,
    Zombie,
}

impl Subtype {
    /// Schema-v4 observation id. Existing discriminants are append-only:
    /// feature encoders may sort and embed these ids without depending on
    /// source spelling or locale-sensitive string ordering.
    pub const fn stable_id(self) -> u16 {
        self as u16
    }
}

/// What a spell/ability needs targeted at cast/activation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetSpec {
    None,
    /// Exactly 1 target: any creature on either battlefield, or either
    /// player.
    AnyTarget,
    /// Exactly 2 targets, in order, the second dependent on the first
    /// (Searing Blaze): target any player, then target a creature *that
    /// player controls*. `engine::legal_targets_for`'s second-pick pool is
    /// computed from the first pick, not independently.
    PlayerThenTheirCreature,
    /// Exactly 1 target: any spell currently on the stack, regardless of
    /// card type or color (Counterspell; also Pyroblast's counter mode,
    /// whose color is checked at resolution rather than targeting -- see
    /// `EffectCond::TargetIsColor`).
    AnySpellOnStack,
    /// Exactly 1 target: an instant spell currently on the stack (Dispel).
    /// This filters by the targeted stack object's card definition, so both
    /// physical instant spells and instant spell copies qualify while
    /// activated/triggered abilities do not.
    InstantSpellOnStack,
    /// Exactly 1 target: a *blue* spell currently on the stack (Red
    /// Elemental Blast's counter mode -- color is filtered at targeting).
    BlueSpellOnStack,
    /// Exactly 1 target: any permanent on either battlefield (Pyroblast's
    /// destroy mode).
    AnyPermanent,
    /// Exactly 1 target: any *blue* permanent on either battlefield (Red
    /// Elemental Blast's destroy mode).
    BluePermanent,
}

/// Combat-relevant keyword abilities, as a bitset. Only `Flying`/`Reach`
/// (blocker legality) and `Haste` (summoning-sickness exemption) are
/// actually set by any card in this increment's pool (Sneaky Snacker,
/// Masked Meower); the rest exist so the shape is right the next time a
/// keyword-bearing card needs one -- in particular `FIRST_STRIKE`/
/// `DOUBLE_STRIKE` back the two-wave combat-damage hook in `engine.rs`
/// even though nothing in Mono-Red Burn has first strike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, Serialize, Deserialize)]
pub struct Keywords(pub u32);

impl Keywords {
    pub const NONE: Keywords = Keywords(0);
    pub const FLYING: Keywords = Keywords(1 << 0);
    pub const REACH: Keywords = Keywords(1 << 1);
    pub const HASTE: Keywords = Keywords(1 << 2);
    pub const VIGILANCE: Keywords = Keywords(1 << 3);
    pub const TRAMPLE: Keywords = Keywords(1 << 4);
    pub const FIRST_STRIKE: Keywords = Keywords(1 << 5);
    pub const DOUBLE_STRIKE: Keywords = Keywords(1 << 6);
    pub const DEATHTOUCH: Keywords = Keywords(1 << 7);
    pub const MENACE: Keywords = Keywords(1 << 8);
    pub const DEFENDER: Keywords = Keywords(1 << 9);
    pub const LIFELINK: Keywords = Keywords(1 << 10);
    pub const HEXPROOF: Keywords = Keywords(1 << 11);
    pub const INDESTRUCTIBLE: Keywords = Keywords(1 << 12);
    pub const PROTECTION_FROM_MONOCOLORED: Keywords = Keywords(1 << 13);

    pub const fn has(self, other: Keywords) -> bool {
        self.0 & other.0 != 0
    }
}

/// Stable W/U/B/R/G/C bit positions used by schema-v4 object colors,
/// landwalk, and color-selection contracts.
pub const fn mana_color_mask(color: ManaColor) -> u8 {
    match color {
        ManaColor::W => 1 << 0,
        ManaColor::U => 1 << 1,
        ManaColor::B => 1 << 2,
        ManaColor::R => 1 << 3,
        ManaColor::G => 1 << 4,
        ManaColor::C => 1 << 5,
    }
}

pub fn mana_colors_mask(colors: &[ManaColor]) -> u8 {
    colors
        .iter()
        .fold(0, |mask, &color| mask | mana_color_mask(color))
}

impl std::ops::BitOr for Keywords {
    type Output = Keywords;
    fn bitor(self, rhs: Keywords) -> Keywords {
        Keywords(self.0 | rhs.0)
    }
}

/// One component of a non-mana cost. Composable (a real cost is `&'static
/// [CostComponent]`) rather than card-shaped, matching the `EffectOp`
/// philosophy in `effect.rs`: "sacrifice 2 Mountains" is
/// `SacrificeLands(2)`, not a `FireblastCost` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostComponent {
    /// Tap the source permanent (activated abilities only -- casting a
    /// spell has no source permanent to tap).
    Tap,
    /// Sacrifice the source permanent itself.
    SacrificeSelf,
    /// Exile the source permanent/card itself.
    ExileSelf,
    /// Discard `n` cards from hand, chosen by the payer (`engine::Decision::Discard`).
    DiscardCards(u8),
    /// Sacrifice `n` controlled lands. This pool's only land is Mountain,
    /// so "sacrifice a land" and "sacrifice a Mountain" coincide; which
    /// specific lands are picked is not a real decision (they're
    /// interchangeable) so it's resolved the same deterministic way the
    /// mana solver auto-picks tap sources -- see `engine::sacrifice_lands`.
    SacrificeLands(u8),
    /// An ordinary mana payment, solved by `mana::solve` same as a spell's
    /// printed cost.
    Mana(Cost),
}

/// The cost of casting a card from the graveyard via flashback (702.10),
/// exiling it instead of returning it to the graveyard on resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashbackCost {
    Mana(Cost),
    SacrificeLands(u8),
}

pub struct FlashbackDef {
    pub cost: FlashbackCost,
}

/// A non-mana activated ability (605/602 use the stack, unlike a mana
/// ability). Only the shapes Masked Meower's and Blood's abilities need:
/// no-target, resolves off the stack as an inline `EffectOp` program (see
/// `state::StackItem::inline_effect`).
pub struct ActivatedAbilityDef {
    pub cost: &'static [CostComponent],
    pub target_spec: TargetSpec,
    pub effect: fn() -> EffectOp,
    /// True iff this ability may only be activated at sorcery speed
    /// (`ActivateAsSorceryActivatedAbility` in Java -- Experimental
    /// Synthesizer's "Activate only as a sorcery."). Checked by
    /// `engine::available_activatable_abilities` via the same
    /// `sorcery_speed_timing_ok` helper a sorcery-speed cast/Plot action
    /// uses. `false` for Masked Meower's and the Blood token's abilities,
    /// which have no such restriction in their Java source.
    pub sorcery_speed_only: bool,
}

/// A spell's alternative mode (Pyroblast's/Red Elemental Blast's "Choose
/// one --" destroy mode): its own target shape and its own resolution
/// program, entirely independent of the card's primary
/// `CardDef::target_spec`/`CardDef::spell_effect`. `engine::Decision::
/// ChooseSpellMode` picks between the primary mode (index 0) and this one
/// (index 1) before targeting begins, for any card with `CardDef::mode2 ==
/// Some(_)`.
pub struct ModeDef {
    pub target_spec: TargetSpec,
    pub effect: fn() -> EffectOp,
}

pub struct CardDef {
    pub name: &'static str,
    pub capability: CardCapability,
    pub cost: Cost,
    pub types: &'static [CardType],
    /// This card's creature/land/artifact subtypes (105.1's subtype line),
    /// e.g. `[Subtype::Human, Subtype::Shaman]` for Burning-Tree Emissary --
    /// see `Subtype`'s own doc for why this is a fully-enumerated closed set.
    /// Read where a card's *own* effect needs it (Rally at the Hornburg's
    /// `CreatureFilter::ControlledWithSubtype(Subtype::Human)` -- see
    /// `effect.rs`).
    pub subtypes: &'static [Subtype],
    pub supertypes: &'static [Supertype],
    pub power: Option<i16>,
    pub toughness: Option<i16>,
    pub is_land: bool,
    pub produces_mana: &'static [ManaColor],
    /// This card's color identity per 105.1/202.2 (the color of mana
    /// symbols in its mana cost) -- empty for a colorless card. Only used
    /// by Pyroblast/Red Elemental Blast's "if it's blue"/"blue spell"/"blue
    /// permanent" checks this increment (`EffectCond::TargetIsColor`,
    /// `engine::legal_targets_for`'s `BlueSpellOnStack`/`BluePermanent`).
    pub colors: &'static [ManaColor],
    pub target_spec: TargetSpec,
    pub keywords: Keywords,
    /// Program run when the spell resolves off the stack. `None` = not
    /// implemented this increment (present in the table, not castable).
    pub spell_effect: fn() -> Option<EffectOp>,
    /// Program run when the card's mana ability is activated. `None` = no
    /// mana ability (or not implemented).
    pub mana_ability: fn() -> Option<EffectOp>,
    /// `Some` iff this card has an alternative cost you may pay instead of
    /// its mana cost (Fireblast). Choosing between them is a real decision
    /// (`engine::Decision::ChooseCastMode`) when both are legal.
    pub alt_cost: Option<&'static [CostComponent]>,
    /// `Some` iff this card has Kicker (`KickerAbility`): an optional
    /// additional cost you may pay as you cast it, stamped onto the spell's
    /// own `state::StackItem::kicked` once paid (`engine::finalize_cast`)
    /// and carried from there into its resolution/ETB context so a later
    /// triggered ability can check `EffectCond::WasKicked`. Only Goblin
    /// Bushwhacker's `Kicker {R}` this increment. Unlike
    /// `additional_cost` (mandatory) or `alt_cost` (replaces the printed
    /// cost), this is paid *in addition to* whichever of those two costs
    /// this cast otherwise settles on -- see `engine::Decision::
    /// ChooseKicker`/`mana::can_pay_combined`.
    pub kicker_cost: Option<Cost>,
    /// `Some` iff this card has a mandatory additional cost paid on top of
    /// its mana cost (Grab the Prize's discard).
    pub additional_cost: Option<&'static [CostComponent]>,
    /// `Some` iff this card can be cast from the graveyard for its
    /// flashback cost (Faithless Looting, Lava Dart).
    pub flashback: Option<FlashbackDef>,
    pub activated_abilities: &'static [ActivatedAbilityDef],
    /// `Some` iff this card can be Plotted (`PlotAbility`): exiled from
    /// hand for this cost at sorcery speed, then castable for free (any
    /// later turn, still sorcery speed) -- `engine::plot_action_candidates`/
    /// `engine::is_plotted_castable_now`. Only Highway Robbery in this pool.
    pub plot_cost: Option<Cost>,
    /// `Some` iff this card has Madness (`MadnessAbility`): whenever it
    /// would be discarded, it's exiled instead, and its owner may cast it
    /// for this cost rather than putting it into the graveyard --
    /// `engine::PendingMadness`/`Decision::ChooseMadnessCast`. Only Fiery
    /// Temper in this pool.
    pub madness_cost: Option<Cost>,
    /// `Some` iff this spell is modal with a second mode (Pyroblast's/Red
    /// Elemental Blast's destroy mode) -- see `ModeDef`'s doc.
    pub mode2: Option<ModeDef>,
    /// A permanent token (`cards_v1.json`'s own `is_token`, e.g. Blood),
    /// never itself a deck card -- read by `trigger::sba_fixed_point` for
    /// 111.8/704.5d ("if a token is in a zone other than the battlefield,
    /// it ceases to exist -- this is a state-based action"). Only `Blood
    /// Token` this increment.
    pub is_token: bool,
}

impl CardDef {
    pub fn has_type(&self, t: CardType) -> bool {
        self.types.contains(&t)
    }

    pub fn is_castable(&self) -> bool {
        self.is_executable() && !self.is_land && !self.is_token
    }

    pub const fn is_executable(&self) -> bool {
        self.capability.is_executable()
    }

    pub const fn has_full_support(&self) -> bool {
        self.capability.is_fully_supported()
    }

    pub fn mana_ability_program(&self) -> Option<EffectOp> {
        self.is_executable()
            .then(|| (self.mana_ability)())
            .flatten()
    }

    pub const fn is_playable_land(&self) -> bool {
        self.is_land && self.is_executable()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeckPreflightError {
    UnknownCardDefinition {
        index: usize,
        card_def: u16,
    },
    TokenInDeck {
        index: usize,
        card_def: u16,
        name: &'static str,
    },
    NotFullySupported {
        index: usize,
        card_def: u16,
        name: &'static str,
        capability: CardCapability,
    },
}

impl fmt::Display for DeckPreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeckPreflightError::UnknownCardDefinition { index, card_def } => {
                write!(f, "deck card {index} references unknown definition {card_def}")
            }
            DeckPreflightError::TokenInDeck {
                index,
                card_def,
                name,
            } => write!(
                f,
                "deck card {index} ({name}, definition {card_def}) is a token and cannot be a deck entry"
            ),
            DeckPreflightError::NotFullySupported {
                index,
                card_def,
                name,
                capability,
            } => write!(
                f,
                "deck card {index} ({name}, definition {card_def}) is not fully supported: {capability:?}"
            ),
        }
    }
}

impl std::error::Error for DeckPreflightError {}

/// Rejects a deck at environment construction time unless every definition
/// is explicitly `Full`. Missing records and newly added records whose
/// capability was omitted both fail closed.
pub fn preflight_fully_supported_deck(card_defs: &[u16]) -> Result<(), DeckPreflightError> {
    for (index, &card_def) in card_defs.iter().enumerate() {
        let Some(def) = CARD_DEFS.get(card_def as usize) else {
            return Err(DeckPreflightError::UnknownCardDefinition { index, card_def });
        };
        if def.is_token {
            return Err(DeckPreflightError::TokenInDeck {
                index,
                card_def,
                name: def.name,
            });
        }
        if !def.has_full_support() {
            return Err(DeckPreflightError::NotFullySupported {
                index,
                card_def,
                name: def.name,
                capability: def.capability,
            });
        }
    }
    Ok(())
}

pub fn no_effect() -> Option<EffectOp> {
    None
}

include!(concat!(env!("OUT_DIR"), "/card_defs.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::{EffectCond, ObjectRef, PlayerRef, TargetRef};
    use crate::state::Zone;

    #[test]
    fn card_defs_len_matches_pool() {
        // 132 real pool cards + 3 tokens (Blood, created by Voldaren
        // Epicure's ETB trigger; Human Soldier Token/Samurai Token, created
        // by Rally at the Hornburg/Experimental Synthesizer -- see
        // `trigger.rs`/`build.rs::activated_abilities_for`).
        assert_eq!(CARD_DEFS.len(), 135);
    }

    #[test]
    fn card_db_hash_v2_is_frozen() {
        assert_eq!(KERNEL_CARDDB_HASH, 0x5c13_381b_3494_f9af);
    }

    #[test]
    fn card_names_are_unique() {
        let mut names: Vec<&str> = CARD_DEFS.iter().map(|c| c.name).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate card names in CARD_DEFS");
    }

    #[test]
    fn lookup_by_name_round_trips_id() {
        for (i, def) in CARD_DEFS.iter().enumerate() {
            assert_eq!(
                card_id_by_name(def.name),
                Some(i as u16),
                "name={}",
                def.name
            );
        }
        assert_eq!(card_id_by_name("Not A Real Card"), None);
    }

    #[test]
    fn intrinsic_basic_lands_derive_the_exact_subtype_mana_ability() {
        for (name, color) in [
            ("Mountain", ManaColor::R),
            ("Island", ManaColor::U),
            ("Forest", ManaColor::G),
            ("Swamp", ManaColor::B),
            ("Snow-Covered Forest", ManaColor::G),
        ] {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            let def = &CARD_DEFS[id as usize];
            assert!(def.is_land, "{name}");
            assert!(def.supertypes.contains(&Supertype::Basic), "{name}");
            assert_eq!(def.produces_mana, &[color], "{name}");
            assert!(def.is_executable(), "{name}");
            assert!(def.has_full_support(), "{name}");
            assert!(!def.is_castable(), "lands aren't cast: {name}");
            match def.mana_ability_program() {
                Some(EffectOp::Sequence(ops)) => {
                    assert_eq!(
                        ops,
                        vec![
                            EffectOp::TapObject {
                                object: ObjectRef::ThisSource
                            },
                            EffectOp::AddMana {
                                player: PlayerRef::Controller,
                                colors: vec![color]
                            },
                        ],
                        "{name}"
                    );
                }
                other => panic!("{name}: expected tap+add-mana sequence, got {other:?}"),
            }
        }
    }

    #[test]
    fn capability_is_the_fail_closed_source_for_programs_and_deck_preflight() {
        let full = CARD_DEFS
            .iter()
            .filter(|def| def.capability == CardCapability::Full)
            .count();
        assert_eq!(full, 35, "32 deck cards plus three required tokens");
        assert_eq!(
            CARD_DEFS
                .iter()
                .filter(|def| def.capability == CardCapability::Partial)
                .count(),
            0
        );

        let supported = ["Island", "Counterspell", "Mountain"]
            .map(|name| card_id_by_name(name).expect("card in registry"));
        assert!(preflight_fully_supported_deck(&supported).is_ok());
        let unsupported = ["Island", "Tolarian Terror"]
            .map(|name| card_id_by_name(name).expect("card in registry"));
        let err = preflight_fully_supported_deck(&unsupported)
            .expect_err("Tolarian Terror is still deferred");
        assert!(matches!(
            err,
            DeckPreflightError::NotFullySupported {
                index: 1,
                name: "Tolarian Terror",
                capability: CardCapability::NoEffect,
                ..
            }
        ));
        assert!(preflight_fully_supported_deck(&[
            card_id_by_name("Island").unwrap(),
            card_id_by_name("Mountain").unwrap(),
        ])
        .is_ok());
        assert!(matches!(
            preflight_fully_supported_deck(&[u16::MAX]),
            Err(DeckPreflightError::UnknownCardDefinition { .. })
        ));
        assert!(matches!(
            preflight_fully_supported_deck(&[card_id_by_name("Blood Token").unwrap()]),
            Err(DeckPreflightError::TokenInDeck {
                name: "Blood Token",
                ..
            })
        ));
    }

    #[test]
    fn unsupported_permanents_and_nonbasic_mana_metadata_remain_inert() {
        let terror = &CARD_DEFS[card_id_by_name("Tolarian Terror").unwrap() as usize];
        assert!(terror.has_type(CardType::Creature));
        assert!(!terror.is_executable());
        assert!(!terror.is_castable());
        assert!((terror.spell_effect)().is_none());

        for name in [
            "Burning-Tree Emissary",
            "Azorius Guildgate",
            "Twisted Landscape",
            "Vault of Whispers",
        ] {
            let def = &CARD_DEFS[card_id_by_name(name).unwrap() as usize];
            assert!(
                !def.produces_mana.is_empty(),
                "test requires mana metadata: {name}"
            );
            assert!(
                def.mana_ability_program().is_none(),
                "metadata alone must not grant {name} a tappable mana ability"
            );
        }
    }

    #[test]
    fn the_four_burn_spells_deal_exactly_their_printed_damage_to_any_target() {
        let expected = [
            ("Lightning Bolt", 3),
            ("Fiery Temper", 3),
            ("Fireblast", 4),
            ("Lava Dart", 1),
        ];
        for (name, amount) in expected {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            let def = &CARD_DEFS[id as usize];
            assert_eq!(def.target_spec, TargetSpec::AnyTarget, "{name}");
            match (def.spell_effect)() {
                Some(EffectOp::DealDamage {
                    target: TargetRef::Target(0),
                    amount: a,
                }) => {
                    assert_eq!(a, amount, "{name}");
                }
                other => panic!("{name}: expected DealDamage to Target(0), got {other:?}"),
            }
        }
    }

    #[test]
    fn vanilla_creatures_resolve_straight_to_battlefield() {
        for name in [
            "Guttersnipe",
            "Masked Meower",
            "Voldaren Epicure",
            "Sneaky Snacker",
        ] {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            let def = &CARD_DEFS[id as usize];
            assert!(def.has_type(CardType::Creature), "{name}");
            assert_eq!(def.target_spec, TargetSpec::None, "{name}");
            match (def.spell_effect)() {
                Some(EffectOp::MoveObject {
                    object: ObjectRef::ThisSource,
                    to_zone: Zone::Battlefield,
                }) => {}
                other => panic!("{name}: expected MoveObject to Battlefield, got {other:?}"),
            }
        }
    }

    #[test]
    fn non_burn_deck_card_has_no_effect_this_increment() {
        // Annul (Mono-Blue Faeries/Terror) is in the pool but out of scope
        // for this increment: present, not castable.
        let id = card_id_by_name("Annul").expect("Annul in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(!def.is_castable());
    }

    #[test]
    fn still_deferred_burn_cards_are_out_of_scope_this_increment() {
        // Relic of Progenitus needs graveyard-card targeting, which doesn't
        // fit any `TargetSpec` shape built so far, and it's sideboard-only
        // -- present in `CARD_DEFS` with correct metadata, not castable,
        // per the kernel's fail-closed invariant.
        let name = "Relic of Progenitus";
        let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
        assert!(!CARD_DEFS[id as usize].is_castable(), "{name}");
    }

    #[test]
    fn highway_robbery_is_castable_and_plottable() {
        let id = card_id_by_name("Highway Robbery").expect("Highway Robbery in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_castable());
        assert_eq!(def.target_spec, TargetSpec::None);
        assert!(matches!(
            (def.spell_effect)(),
            Some(EffectOp::MayPayCostThen {
                discard: 1,
                sacrifice_lands: 1,
                ..
            })
        ));
        let plot_cost = def
            .plot_cost
            .expect("Highway Robbery should have Plot {1}{R}");
        assert_eq!(plot_cost.generic, 1);
        assert_eq!(plot_cost.pips, &[Pip::Colored(ManaColor::R)]);
    }

    #[test]
    fn fiery_temper_has_madness_r() {
        let id = card_id_by_name("Fiery Temper").expect("Fiery Temper in pool");
        let def = &CARD_DEFS[id as usize];
        let madness_cost = def
            .madness_cost
            .expect("Fiery Temper should have Madness {R}");
        assert_eq!(madness_cost.generic, 0);
        assert_eq!(madness_cost.pips, &[Pip::Colored(ManaColor::R)]);
    }

    #[test]
    fn searing_blaze_targets_player_then_their_creature() {
        let id = card_id_by_name("Searing Blaze").expect("Searing Blaze in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_castable());
        assert_eq!(def.target_spec, TargetSpec::PlayerThenTheirCreature);
    }

    #[test]
    fn counterspell_and_dispel_share_the_counter_program_with_distinct_filters() {
        let counterspell = &CARD_DEFS[card_id_by_name("Counterspell").unwrap() as usize];
        let dispel = &CARD_DEFS[card_id_by_name("Dispel").unwrap() as usize];

        assert!(counterspell.has_full_support());
        assert!(dispel.has_full_support());
        assert_eq!(counterspell.target_spec, TargetSpec::AnySpellOnStack);
        assert_eq!(dispel.target_spec, TargetSpec::InstantSpellOnStack);
        assert_eq!((counterspell.spell_effect)(), (dispel.spell_effect)());
        assert!(matches!(
            (counterspell.spell_effect)(),
            Some(EffectOp::Conditional {
                cond: EffectCond::TargetInZone(0, Zone::Stack),
                ..
            })
        ));
    }

    #[test]
    fn pyroblast_and_red_elemental_blast_are_modal_and_castable() {
        let pyroblast = &CARD_DEFS[card_id_by_name("Pyroblast").unwrap() as usize];
        assert!(pyroblast.is_castable());
        assert_eq!(pyroblast.target_spec, TargetSpec::AnySpellOnStack);
        assert_eq!(
            pyroblast.mode2.as_ref().map(|m| m.target_spec),
            Some(TargetSpec::AnyPermanent)
        );

        let reb = &CARD_DEFS[card_id_by_name("Red Elemental Blast").unwrap() as usize];
        assert!(reb.is_castable());
        assert_eq!(reb.target_spec, TargetSpec::BlueSpellOnStack);
        assert_eq!(
            reb.mode2.as_ref().map(|m| m.target_spec),
            Some(TargetSpec::BluePermanent)
        );
    }

    #[test]
    fn card_colors_are_populated_from_the_json_pool() {
        let bolt = &CARD_DEFS[card_id_by_name("Lightning Bolt").unwrap() as usize];
        assert_eq!(bolt.colors, &[ManaColor::R]);
        let relic = &CARD_DEFS[card_id_by_name("Relic of Progenitus").unwrap() as usize];
        assert!(relic.colors.is_empty(), "Relic of Progenitus is colorless");
    }

    #[test]
    fn grab_the_prize_is_castable_with_a_mandatory_discard_additional_cost() {
        let id = card_id_by_name("Grab the Prize").expect("Grab the Prize in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_castable());
        assert_eq!(
            def.additional_cost,
            Some([CostComponent::DiscardCards(1)].as_slice())
        );
    }

    #[test]
    fn fireblast_has_a_sacrifice_two_mountains_alt_cost() {
        let id = card_id_by_name("Fireblast").expect("Fireblast in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(
            def.alt_cost,
            Some([CostComponent::SacrificeLands(2)].as_slice())
        );
    }

    #[test]
    fn faithless_looting_and_lava_dart_have_flashback() {
        let looting = &CARD_DEFS[card_id_by_name("Faithless Looting").unwrap() as usize];
        assert!(matches!(
            looting.flashback,
            Some(FlashbackDef {
                cost: FlashbackCost::Mana(_)
            })
        ));
        let lava_dart = &CARD_DEFS[card_id_by_name("Lava Dart").unwrap() as usize];
        assert!(matches!(
            lava_dart.flashback,
            Some(FlashbackDef {
                cost: FlashbackCost::SacrificeLands(1)
            })
        ));
    }

    #[test]
    fn masked_meower_has_haste_and_a_draw_activated_ability() {
        let id = card_id_by_name("Masked Meower").expect("Masked Meower in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.keywords.has(Keywords::HASTE));
        assert_eq!(def.activated_abilities.len(), 1);
        assert_eq!(
            def.activated_abilities[0].cost,
            [CostComponent::DiscardCards(1), CostComponent::SacrificeSelf].as_slice()
        );
    }

    #[test]
    fn sneaky_snacker_has_flying() {
        let id = card_id_by_name("Sneaky Snacker").expect("Sneaky Snacker in pool");
        assert!(CARD_DEFS[id as usize].keywords.has(Keywords::FLYING));
    }

    #[test]
    fn blood_token_exists_with_its_draw_a_card_ability() {
        let id =
            card_id_by_name("Blood Token").expect("Blood Token should be codegen'd as a token");
        let def = &CARD_DEFS[id as usize];
        assert!(!def.is_castable(), "tokens are never cast");
        assert_eq!(def.activated_abilities.len(), 1);
    }

    // ---- Rally at the Hornburg increment -----------------------------

    #[test]
    fn great_furnace_is_a_second_mountain_that_is_also_an_artifact() {
        let id = card_id_by_name("Great Furnace").expect("Great Furnace in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_land);
        assert!(def.has_type(CardType::Artifact));
        assert_eq!(def.produces_mana, &[ManaColor::R]);
        assert!(!def.is_castable());
        assert!((def.mana_ability)().is_some());
    }

    #[test]
    fn burning_tree_emissary_has_hybrid_cost_and_no_spell_effect_of_its_own() {
        let id = card_id_by_name("Burning-Tree Emissary").expect("Burning-Tree Emissary in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(
            def.cost.pips,
            &[
                Pip::Hybrid(ManaColor::R, ManaColor::G),
                Pip::Hybrid(ManaColor::R, ManaColor::G)
            ]
        );
        assert_eq!(def.subtypes, &[Subtype::Human, Subtype::Shaman]);
        match (def.spell_effect)() {
            Some(EffectOp::MoveObject {
                object: ObjectRef::ThisSource,
                to_zone: Zone::Battlefield,
            }) => {}
            other => panic!("expected MoveObject to Battlefield, got {other:?}"),
        }
    }

    #[test]
    fn chain_lightning_deals_3_damage_then_offers_the_copy_cost() {
        let id = card_id_by_name("Chain Lightning").expect("Chain Lightning in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.target_spec, TargetSpec::AnyTarget);
        match (def.spell_effect)() {
            Some(EffectOp::Sequence(ops)) => {
                assert_eq!(ops.len(), 2);
                assert_eq!(
                    ops[0],
                    EffectOp::DealDamage {
                        target: TargetRef::Target(0),
                        amount: 3
                    }
                );
                assert_eq!(
                    ops[1],
                    EffectOp::OfferAffectedPlayerSpellCopy {
                        affected: TargetRef::Target(0)
                    }
                );
            }
            other => {
                panic!("expected a 2-op Sequence (damage, then the copy offer), got {other:?}")
            }
        }
    }

    #[test]
    fn cast_into_the_fire_remains_deferred_this_increment() {
        // Sideboard-only, modal with a 0-2 variable-count target mode this
        // kernel's `TargetSpec` shape doesn't support -- see the module doc.
        let id = card_id_by_name("Cast into the Fire").expect("Cast into the Fire in pool");
        assert!(!CARD_DEFS[id as usize].is_castable());
    }

    #[test]
    fn goblin_bushwhacker_has_kicker_r_and_no_static_haste() {
        let id = card_id_by_name("Goblin Bushwhacker").expect("Goblin Bushwhacker in pool");
        let def = &CARD_DEFS[id as usize];
        let kicker = def
            .kicker_cost
            .expect("Goblin Bushwhacker should have Kicker {R}");
        assert_eq!(kicker.generic, 0);
        assert_eq!(kicker.pips, &[Pip::Colored(ManaColor::R)]);
        assert!(
            !def.keywords.has(Keywords::HASTE),
            "haste is conditional on Kicker, not a static keyword"
        );
    }

    #[test]
    fn goblin_tomb_raider_has_no_static_haste_either() {
        // "As long as you control an artifact, gets +1/+0 and has haste" is
        // a conditional static ability (`engine::static_self_boost_for`),
        // not an unconditional `Keywords` bit.
        let id = card_id_by_name("Goblin Tomb Raider").expect("Goblin Tomb Raider in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(!def.keywords.has(Keywords::HASTE));
        assert_eq!(def.power, Some(1));
        assert_eq!(def.toughness, Some(2));
    }

    #[test]
    fn clockwork_percussionist_has_haste() {
        let id =
            card_id_by_name("Clockwork Percussionist").expect("Clockwork Percussionist in pool");
        assert!(CARD_DEFS[id as usize].keywords.has(Keywords::HASTE));
    }

    #[test]
    fn experimental_synthesizer_has_a_sorcery_speed_only_sacrifice_ability() {
        let id =
            card_id_by_name("Experimental Synthesizer").expect("Experimental Synthesizer in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.activated_abilities.len(), 1);
        let ability = &def.activated_abilities[0];
        assert!(ability.sorcery_speed_only);
        assert_eq!(
            ability.cost,
            [
                CostComponent::Mana(Cost {
                    pips: &[Pip::Colored(ManaColor::R)],
                    generic: 2,
                    x_count: 0
                }),
                CostComponent::SacrificeSelf
            ]
            .as_slice()
        );
    }

    #[test]
    fn galvanic_blast_is_conditional_on_metalcraft() {
        let id = card_id_by_name("Galvanic Blast").expect("Galvanic Blast in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.target_spec, TargetSpec::AnyTarget);
        assert!(matches!(
            (def.spell_effect)(),
            Some(EffectOp::Conditional {
                cond: EffectCond::ControlsArtifactCount(3),
                ..
            })
        ));
    }

    #[test]
    fn end_the_festivities_hits_the_opponent_and_their_creatures() {
        let id = card_id_by_name("End the Festivities").expect("End the Festivities in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.target_spec, TargetSpec::None);
        assert_eq!(
            (def.spell_effect)(),
            Some(EffectOp::DamageOpponentAndTheirCreatures { amount: 1 })
        );
    }

    #[test]
    fn reckless_impulse_exiles_two_cards_until_owners_next_turn() {
        let id = card_id_by_name("Reckless Impulse").expect("Reckless Impulse in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(
            (def.spell_effect)(),
            Some(EffectOp::ImpulseDraw {
                count: 2,
                duration: crate::effect::ImpulseDuration::UntilOwnersNextTurn
            })
        );
    }

    #[test]
    fn rally_at_the_hornburg_creates_two_tokens_and_pumps_humans() {
        let id = card_id_by_name("Rally at the Hornburg").expect("Rally at the Hornburg in pool");
        let def = &CARD_DEFS[id as usize];
        match (def.spell_effect)() {
            Some(EffectOp::Sequence(ops)) => {
                assert_eq!(ops.len(), 3);
                assert!(matches!(ops[0], EffectOp::CreateToken { .. }));
                assert!(matches!(ops[1], EffectOp::CreateToken { .. }));
                assert!(matches!(
                    ops[2],
                    EffectOp::PumpControlled {
                        filter: crate::effect::CreatureFilter::ControlledWithSubtype(
                            Subtype::Human
                        ),
                        grant_haste: true,
                        ..
                    }
                ));
            }
            other => panic!("expected a 3-op Sequence, got {other:?}"),
        }
    }

    #[test]
    fn human_soldier_and_samurai_tokens_exist() {
        let hst = card_id_by_name("Human Soldier Token")
            .expect("Human Soldier Token should be codegen'd as a token");
        let def = &CARD_DEFS[hst as usize];
        assert!(!def.is_castable());
        assert_eq!(def.power, Some(1));
        assert_eq!(def.toughness, Some(1));
        assert_eq!(def.subtypes, &[Subtype::Human, Subtype::Soldier]);

        let samurai =
            card_id_by_name("Samurai Token").expect("Samurai Token should be codegen'd as a token");
        let sdef = &CARD_DEFS[samurai as usize];
        assert_eq!(sdef.power, Some(2));
        assert_eq!(sdef.toughness, Some(2));
        assert!(sdef.keywords.has(Keywords::VIGILANCE));
    }
}
