//! Static card database. `CARD_DEFS` (and `card_id_by_name`,
//! `KERNEL_CARDDB_HASH`) are generated at build time by `build.rs` from
//! `kernel/data/cards_v1.json` -- see that file for the codegen and its
//! validation (duplicate names / empty deck coverage / schema-version
//! mismatch all fail the build).
//!
//! Only Mono-Red Burn's 16 cards carry a real `spell_effect` /
//! `mana_ability` program this increment: basic Mountain, the 4 "N damage
//! to any target" burn spells (Lightning Bolt, Fiery Temper, Fireblast,
//! Lava Dart -- alternate/additional costs like madness and the
//! sacrifice-Mountains cost are ignored this increment, both are always
//! hard-cast for their normal mana cost), and the 4 creatures as vanilla
//! bodies (Guttersnipe, Masked Meower, Voldaren Epicure, Sneaky Snacker --
//! keyword abilities/triggers ignored, they're just a castable body).
//! Every other card (including the other 7 Mono-Red Burn cards whose
//! effects don't fit the "any target" burn shape: Faithless Looting, Grab
//! the Prize, Highway Robbery, Pyroblast, Red Elemental Blast, Relic of
//! Progenitus, Searing Blaze) gets `no_effect` for both -- present in the
//! table with correct name/cost/types (so ids are stable and the table is
//! complete), but not castable, per the kernel's fail-closed invariant
//! (see `lib.rs`).

use crate::effect::{EffectOp, ObjectRef, PlayerRef, TargetRef};
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

/// What a spell needs targeted at cast time. Only the shapes Mono-Red Burn
/// needs this increment; modal/variable-count targeting is a future
/// increment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetSpec {
    None,
    /// Exactly 1 target: any creature on either battlefield, or either
    /// player.
    AnyTarget,
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
    pub target_spec: TargetSpec,
    /// Program run when the spell resolves off the stack. `None` = not
    /// implemented this increment (present in the table, not castable).
    pub spell_effect: fn() -> Option<EffectOp>,
    /// Program run when the card's mana ability is activated. `None` = no
    /// mana ability (or not implemented).
    pub mana_ability: fn() -> Option<EffectOp>,
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
        assert_eq!(CARD_DEFS.len(), 132);
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
    fn searing_blaze_and_grab_the_prize_are_out_of_scope_this_increment() {
        // Both are Mono-Red Burn cards but don't fit the "N damage to any
        // target/player-only" shape (2-target conditional landfall burn;
        // conditional discard-draw-burn), so per the design brief they are
        // NOT among the 6 implemented burn/creature effects this
        // increment.
        for name in ["Searing Blaze", "Grab the Prize"] {
            let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} in pool"));
            assert!(!CARD_DEFS[id as usize].is_castable(), "{name}");
        }
    }
}
