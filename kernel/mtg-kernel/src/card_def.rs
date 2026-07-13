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
//! Every non-Burn card in the 132-card pool gets `no_effect` for both --
//! present in the table with correct name/cost/types (so ids are stable and
//! the table is complete), but not castable, per the kernel's fail-closed
//! invariant (see `lib.rs`).

use crate::effect::{EffectCond, EffectOp, ObjectRef, PlayerRef, TargetRef};
use crate::mana::{Cost, ManaColor, Pip};
use crate::state::Zone;
use serde::{Deserialize, Serialize};

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
    /// color (Pyroblast's counter mode -- color is checked at resolution,
    /// not at targeting; see `EffectCond::TargetIsColor`).
    AnySpellOnStack,
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
pub struct Keywords(pub u16);

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

    pub const fn has(self, other: Keywords) -> bool {
        self.0 & other.0 != 0
    }
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
    pub cost: Cost,
    pub types: &'static [CardType],
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
}

impl CardDef {
    pub fn has_type(&self, t: CardType) -> bool {
        self.types.contains(&t)
    }

    pub fn is_castable(&self) -> bool {
        (self.spell_effect)().is_some()
    }
}

pub fn no_effect() -> Option<EffectOp> {
    None
}

include!(concat!(env!("OUT_DIR"), "/card_defs.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::{ObjectRef, PlayerRef, TargetRef};
    use crate::state::Zone;

    #[test]
    fn card_defs_len_matches_pool() {
        // 132 real pool cards + 1 token (Blood, created by Voldaren
        // Epicure's ETB trigger -- see `trigger.rs`).
        assert_eq!(CARD_DEFS.len(), 133);
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
            assert_eq!(card_id_by_name(def.name), Some(i as u16), "name={}", def.name);
        }
        assert_eq!(card_id_by_name("Not A Real Card"), None);
    }

    #[test]
    fn mountain_has_mana_ability_and_no_spell_effect() {
        let id = card_id_by_name("Mountain").expect("Mountain in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.is_land);
        assert_eq!(def.produces_mana, &[ManaColor::R]);
        assert!(!def.is_castable(), "lands aren't cast");
        match (def.mana_ability)() {
            Some(EffectOp::Sequence(ops)) => {
                assert_eq!(ops, vec![
                    EffectOp::TapObject { object: ObjectRef::ThisSource },
                    EffectOp::AddMana { player: PlayerRef::Controller, colors: vec![ManaColor::R] },
                ]);
            }
            other => panic!("expected tap+add-mana sequence, got {other:?}"),
        }
    }

    #[test]
    fn the_four_burn_spells_deal_exactly_their_printed_damage_to_any_target() {
        let expected = [("Lightning Bolt", 3), ("Fiery Temper", 3), ("Fireblast", 4), ("Lava Dart", 1)];
        for (name, amount) in expected {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            let def = &CARD_DEFS[id as usize];
            assert_eq!(def.target_spec, TargetSpec::AnyTarget, "{name}");
            match (def.spell_effect)() {
                Some(EffectOp::DealDamage { target: TargetRef::Target(0), amount: a }) => {
                    assert_eq!(a, amount, "{name}");
                }
                other => panic!("{name}: expected DealDamage to Target(0), got {other:?}"),
            }
        }
    }

    #[test]
    fn vanilla_creatures_resolve_straight_to_battlefield() {
        for name in ["Guttersnipe", "Masked Meower", "Voldaren Epicure", "Sneaky Snacker"] {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            let def = &CARD_DEFS[id as usize];
            assert!(def.has_type(CardType::Creature), "{name}");
            assert_eq!(def.target_spec, TargetSpec::None, "{name}");
            match (def.spell_effect)() {
                Some(EffectOp::MoveObject { object: ObjectRef::ThisSource, to_zone: Zone::Battlefield }) => {}
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
        assert!(matches!((def.spell_effect)(), Some(EffectOp::MayPayCostThen { discard: 1, sacrifice_lands: 1, .. })));
        let plot_cost = def.plot_cost.expect("Highway Robbery should have Plot {1}{R}");
        assert_eq!(plot_cost.generic, 1);
        assert_eq!(plot_cost.pips, &[Pip::Colored(ManaColor::R)]);
    }

    #[test]
    fn fiery_temper_has_madness_r() {
        let id = card_id_by_name("Fiery Temper").expect("Fiery Temper in pool");
        let def = &CARD_DEFS[id as usize];
        let madness_cost = def.madness_cost.expect("Fiery Temper should have Madness {R}");
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
    fn pyroblast_and_red_elemental_blast_are_modal_and_castable() {
        let pyroblast = &CARD_DEFS[card_id_by_name("Pyroblast").unwrap() as usize];
        assert!(pyroblast.is_castable());
        assert_eq!(pyroblast.target_spec, TargetSpec::AnySpellOnStack);
        assert_eq!(pyroblast.mode2.as_ref().map(|m| m.target_spec), Some(TargetSpec::AnyPermanent));

        let reb = &CARD_DEFS[card_id_by_name("Red Elemental Blast").unwrap() as usize];
        assert!(reb.is_castable());
        assert_eq!(reb.target_spec, TargetSpec::BlueSpellOnStack);
        assert_eq!(reb.mode2.as_ref().map(|m| m.target_spec), Some(TargetSpec::BluePermanent));
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
        assert_eq!(def.additional_cost, Some([CostComponent::DiscardCards(1)].as_slice()));
    }

    #[test]
    fn fireblast_has_a_sacrifice_two_mountains_alt_cost() {
        let id = card_id_by_name("Fireblast").expect("Fireblast in pool");
        let def = &CARD_DEFS[id as usize];
        assert_eq!(def.alt_cost, Some([CostComponent::SacrificeLands(2)].as_slice()));
    }

    #[test]
    fn faithless_looting_and_lava_dart_have_flashback() {
        let looting = &CARD_DEFS[card_id_by_name("Faithless Looting").unwrap() as usize];
        assert!(matches!(looting.flashback, Some(FlashbackDef { cost: FlashbackCost::Mana(_) })));
        let lava_dart = &CARD_DEFS[card_id_by_name("Lava Dart").unwrap() as usize];
        assert!(matches!(lava_dart.flashback, Some(FlashbackDef { cost: FlashbackCost::SacrificeLands(1) })));
    }

    #[test]
    fn masked_meower_has_haste_and_a_draw_activated_ability() {
        let id = card_id_by_name("Masked Meower").expect("Masked Meower in pool");
        let def = &CARD_DEFS[id as usize];
        assert!(def.keywords.has(Keywords::HASTE));
        assert_eq!(def.activated_abilities.len(), 1);
        assert_eq!(def.activated_abilities[0].cost, [CostComponent::DiscardCards(1), CostComponent::SacrificeSelf].as_slice());
    }

    #[test]
    fn sneaky_snacker_has_flying() {
        let id = card_id_by_name("Sneaky Snacker").expect("Sneaky Snacker in pool");
        assert!(CARD_DEFS[id as usize].keywords.has(Keywords::FLYING));
    }

    #[test]
    fn blood_token_exists_with_its_draw_a_card_ability() {
        let id = card_id_by_name("Blood Token").expect("Blood Token should be codegen'd as a token");
        let def = &CARD_DEFS[id as usize];
        assert!(!def.is_castable(), "tokens are never cast");
        assert_eq!(def.activated_abilities.len(), 1);
    }
}
