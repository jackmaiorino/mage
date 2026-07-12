//! Codegen: `kernel/data/cards_v1.json` -> `$OUT_DIR/card_defs.rs`, included
//! verbatim by `src/card_def.rs`.
//!
//! Fails the build on:
//! - schema version mismatch (`cards_v1.json`'s `"version"` != what this
//!   codegen understands)
//! - duplicate card names
//! - a card with empty deck coverage (`"decks": []`)
//!
//! `u16` ids are assigned in JSON array order (stable: the file is a
//! checked-in fixed pool, not regenerated per build). Only Mono-Red Burn's
//! basic Mountain, its 4 "N damage to any target" burn spells, and its 4
//! creatures get a real `spell_effect/mana_ability` program; see
//! `special_for` and the module doc in `src/card_def.rs`.

use serde::Deserialize;
use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const EXPECTED_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
struct CardJson {
    name: String,
    mana_cost: String,
    #[serde(default)]
    types: Vec<String>,
    #[serde(default)]
    supertypes: Vec<String>,
    power: Option<i32>,
    toughness: Option<i32>,
    is_land: bool,
    #[serde(default)]
    produces_mana: Vec<String>,
    decks: Vec<String>,
    /// A permanent token (e.g. Blood), not itself a deck card: exempt from
    /// the empty-deck-coverage check (see `main`'s validation loop) and
    /// never castable (its `Special` falls through to `Special::None`,
    /// giving it `no_effect`/`no_effect`/`TargetSpec::None` -- correct,
    /// since tokens are never cast, only created by another card's effect).
    #[serde(default)]
    is_token: bool,
}

#[derive(Debug, Deserialize)]
struct CardsFile {
    version: u32,
    cards: Vec<CardJson>,
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let json_path = Path::new(&manifest_dir).join("..").join("data").join("cards_v1.json");
    println!("cargo:rerun-if-changed={}", json_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let text =
        fs::read_to_string(&json_path).unwrap_or_else(|e| panic!("failed to read {}: {e}", json_path.display()));
    let data: CardsFile =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("failed to parse {}: {e}", json_path.display()));

    if data.version != EXPECTED_SCHEMA_VERSION {
        panic!(
            "cards_v1.json schema version mismatch: codegen expects version {EXPECTED_SCHEMA_VERSION}, \
             file has version {}. Update build.rs's CardJson/codegen for the new schema before building.",
            data.version
        );
    }

    let mut seen_names = HashSet::new();
    for c in &data.cards {
        if !seen_names.insert(c.name.clone()) {
            panic!("cards_v1.json: duplicate card name {:?}", c.name);
        }
        if c.decks.is_empty() && !c.is_token {
            panic!("cards_v1.json: card {:?} has empty deck coverage", c.name);
        }
    }

    let out = codegen(&data.cards);

    let out_dir = env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let dest = Path::new(&out_dir).join("card_defs.rs");
    fs::write(&dest, out).unwrap_or_else(|e| panic!("failed to write {}: {e}", dest.display()));
}

/// The Mono-Red Burn cards that get a real effect program this increment.
/// Still deferred -- present in `CARD_DEFS` with correct metadata, not
/// castable; see the module doc in `src/card_def.rs` and the increment-3
/// report for exactly why each one is deferred:
///
/// - Highway Robbery: optional discard-or-sacrifice-land cost + Plot
///   alternate casting, neither modeled.
/// - Pyroblast / Red Elemental Blast: modal cast-time mode choice, plus
///   spell-on-stack targeting, countering, and card-color tracking, none
///   of which exist yet.
/// - Relic of Progenitus: graveyard-card targeting; sideboard-only so
///   lower priority.
/// - Searing Blaze: 2-related-targets shape + landfall watcher;
///   sideboard-only.
enum Special {
    None,
    Mountain,
    /// Deals `amount` damage to any target (Lightning Bolt, Fiery Temper,
    /// Fireblast, Lava Dart). Fiery Temper's madness alternate cost is
    /// NOT modeled this increment (see `card_def.rs` module doc) -- it's
    /// always hard-cast for `{1}{R}{R}`. Fireblast's/Lava Dart's real alt
    /// cost / flashback ARE modeled, via the separate `alt_cost_for`/
    /// `flashback_for` tables below (independent of `Special`, since a
    /// card's targeting/damage shape and its cost shape are orthogonal).
    BurnAnyTarget(i32),
    /// Resolves straight onto the battlefield; any keyword/triggered
    /// ability is layered on separately (`keywords_for` for
    /// static keywords, `src/trigger.rs`'s hand-written `triggers_for`
    /// table for triggered abilities -- see that module for Guttersnipe/
    /// Voldaren Epicure/Sneaky Snacker's real texts).
    VanillaCreature,
    /// "Draw `draw` cards, then discard `discard` cards" (Faithless
    /// Looting: 2 and 2). The discard is a resolution effect (not a cost),
    /// so it's `EffectOp::DiscardCards`, staged via
    /// `engine::EngineState::pending_discard` same as cleanup.
    DrawThenDiscard { draw: i32, discard: i32 },
    /// Grab the Prize: draw two cards, then (if the card discarded to pay
    /// the mandatory additional cost -- see `additional_cost_for` --
    /// wasn't a land) deal 2 damage to the opponent.
    GrabThePrize,
}

fn special_for(name: &str) -> Special {
    match name {
        "Mountain" => Special::Mountain,
        "Lightning Bolt" => Special::BurnAnyTarget(3),
        "Fiery Temper" => Special::BurnAnyTarget(3),
        "Fireblast" => Special::BurnAnyTarget(4),
        "Lava Dart" => Special::BurnAnyTarget(1),
        "Guttersnipe" | "Masked Meower" | "Voldaren Epicure" | "Sneaky Snacker" => Special::VanillaCreature,
        "Faithless Looting" => Special::DrawThenDiscard { draw: 2, discard: 2 },
        "Grab the Prize" => Special::GrabThePrize,
        _ => Special::None,
    }
}

/// Static combat/summoning-sickness keywords, verified against each card's
/// Java source (see the increment-3 report for the exact files read).
/// Only Masked Meower (haste) and Sneaky Snacker (flying) carry one in
/// this pool.
fn keywords_for(name: &str) -> &'static str {
    match name {
        "Masked Meower" => "Keywords::HASTE",
        "Sneaky Snacker" => "Keywords::FLYING",
        _ => "Keywords::NONE",
    }
}

/// `Some` alternative cost source text (`CardDef::alt_cost`), verified
/// against Java. Only Fireblast has one this increment ("You may
/// sacrifice two Mountains rather than pay Fireblast's mana cost.").
fn alt_cost_for(name: &str) -> &'static str {
    match name {
        "Fireblast" => "Some(&[CostComponent::SacrificeLands(2)])",
        _ => "None",
    }
}

/// `Some` mandatory additional cost text (`CardDef::additional_cost`).
/// Only Grab the Prize has one this increment ("As an additional cost to
/// cast this spell, discard a card.").
fn additional_cost_for(name: &str) -> &'static str {
    match name {
        "Grab the Prize" => "Some(&[CostComponent::DiscardCards(1)])",
        _ => "None",
    }
}

/// `Some` flashback definition, verified against Java. Faithless Looting
/// ("Flashback {2}{R}") pays mana; Lava Dart ("Flashback -- Sacrifice a
/// Mountain.") sacrifices a land instead.
fn flashback_for(name: &str) -> String {
    match name {
        "Faithless Looting" => {
            let (pips, generic, x_count) = parse_cost("{2}{R}");
            format!(
                "Some(FlashbackDef {{ cost: FlashbackCost::Mana(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }}) }})",
                pips.join(", ")
            )
        }
        "Lava Dart" => "Some(FlashbackDef { cost: FlashbackCost::SacrificeLands(1) })".to_string(),
        _ => "None".to_string(),
    }
}

/// Non-mana activated abilities, verified against Java. Masked Meower
/// ("Discard a card, Sacrifice this creature: Draw a card.") and the Blood
/// token ("{1}, {T}, Discard a card, Sacrifice this artifact: Draw a
/// card.") both reduce to "discard/sacrifice/[cost]: draw a card", so both
/// share `ability_effect_draw_one`.
fn activated_abilities_for(name: &str) -> &'static str {
    match name {
        "Masked Meower" => {
            "&[ActivatedAbilityDef { \
                cost: &[CostComponent::DiscardCards(1), CostComponent::SacrificeSelf], \
                target_spec: TargetSpec::None, \
                effect: ability_effect_draw_one, \
            }]"
        }
        "Blood Token" => {
            "&[ActivatedAbilityDef { \
                cost: &[CostComponent::Mana(Cost { pips: &[], generic: 1, x_count: 0 }), CostComponent::Tap, \
                        CostComponent::DiscardCards(1), CostComponent::SacrificeSelf], \
                target_spec: TargetSpec::None, \
                effect: ability_effect_draw_one, \
            }]"
        }
        _ => "&[]",
    }
}

fn codegen(cards: &[CardJson]) -> String {
    let mut out = String::new();
    writeln!(out, "// GENERATED by build.rs from kernel/data/cards_v1.json. Do not edit by hand.").unwrap();
    writeln!(out).unwrap();

    // Shared/one-off effect-program functions. Function *pointers* (not
    // owned EffectOp values) are what make a `static [CardDef; N]` array
    // possible: EffectOp contains Vec/Box and can't live in a const
    // initializer directly, but a `fn() -> Option<EffectOp>` can, and it
    // builds the (small) tree fresh each call.
    writeln!(out, "fn mana_ability_mountain() -> Option<EffectOp> {{").unwrap();
    writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
    writeln!(out, "        EffectOp::TapObject {{ object: ObjectRef::ThisSource }},").unwrap();
    writeln!(out, "        EffectOp::AddMana {{ player: PlayerRef::Controller, colors: vec![ManaColor::R] }},").unwrap();
    writeln!(out, "    ]))").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "fn spell_effect_vanilla_creature() -> Option<EffectOp> {{").unwrap();
    writeln!(out, "    Some(EffectOp::MoveObject {{ object: ObjectRef::ThisSource, to_zone: Zone::Battlefield }})").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Masked Meower's / Blood's activated ability both resolve to "draw a
    // card" once their (differing) costs are paid.
    writeln!(out, "fn ability_effect_draw_one() -> EffectOp {{").unwrap();
    writeln!(out, "    EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 1 }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    let draw_then_discard_shapes: Vec<(i32, i32)> = cards
        .iter()
        .filter_map(|c| match special_for(&c.name) {
            Special::DrawThenDiscard { draw, discard } => Some((draw, discard)),
            _ => None,
        })
        .collect();
    for (draw, discard) in draw_then_discard_shapes {
        writeln!(out, "fn spell_effect_draw_then_discard_{draw}_{discard}() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(out, "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},").unwrap();
        writeln!(out, "        EffectOp::DiscardCards {{ player: PlayerRef::Controller, count: {discard} }},").unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards.iter().any(|c| matches!(special_for(&c.name), Special::GrabThePrize)) {
        writeln!(out, "fn spell_effect_grab_the_prize() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(out, "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 2 }},").unwrap();
        writeln!(out, "        EffectOp::Conditional {{").unwrap();
        writeln!(out, "            cond: EffectCond::DiscardedNonLandForCost,").unwrap();
        writeln!(out, "            then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Opponent, amount: 2 }}),").unwrap();
        writeln!(out, "            else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "        }},").unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let burn_amounts: BTreeSetLike = cards
        .iter()
        .filter_map(|c| match special_for(&c.name) {
            Special::BurnAnyTarget(a) => Some(a),
            _ => None,
        })
        .collect();
    for amount in burn_amounts.values() {
        writeln!(out, "fn spell_effect_burn_any_target_{amount}() -> Option<EffectOp> {{").unwrap();
        writeln!(out, "    Some(EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: {amount} }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    // ---- CARD_DEFS -------------------------------------------------
    writeln!(out, "pub static CARD_DEFS: [CardDef; {}] = [", cards.len()).unwrap();
    for c in cards {
        let (pips, generic, x_count) = parse_cost(&c.mana_cost);
        let special = special_for(&c.name);

        let types_src = c.types.iter().map(|t| format!("CardType::{}", card_type_variant(t))).collect::<Vec<_>>().join(", ");
        let supertypes_src = c
            .supertypes
            .iter()
            .map(|t| format!("Supertype::{}", supertype_variant(t)))
            .collect::<Vec<_>>()
            .join(", ");
        let produces_src = c
            .produces_mana
            .iter()
            .map(|m| format!("ManaColor::{}", color_variant(m)))
            .collect::<Vec<_>>()
            .join(", ");
        let pips_src = pips.join(", ");
        let power_src = match c.power {
            Some(p) => format!("Some({p})"),
            None => "None".to_string(),
        };
        let toughness_src = match c.toughness {
            Some(t) => format!("Some({t})"),
            None => "None".to_string(),
        };

        let (target_spec_src, spell_effect_src, mana_ability_src) = match special {
            Special::None => ("TargetSpec::None", "no_effect".to_string(), "no_effect".to_string()),
            Special::Mountain => ("TargetSpec::None", "no_effect".to_string(), "mana_ability_mountain".to_string()),
            Special::BurnAnyTarget(amount) => {
                ("TargetSpec::AnyTarget", format!("spell_effect_burn_any_target_{amount}"), "no_effect".to_string())
            }
            Special::VanillaCreature => {
                ("TargetSpec::None", "spell_effect_vanilla_creature".to_string(), "no_effect".to_string())
            }
            Special::DrawThenDiscard { draw, discard } => {
                ("TargetSpec::None", format!("spell_effect_draw_then_discard_{draw}_{discard}"), "no_effect".to_string())
            }
            Special::GrabThePrize => ("TargetSpec::None", "spell_effect_grab_the_prize".to_string(), "no_effect".to_string()),
        };

        writeln!(out, "    CardDef {{").unwrap();
        writeln!(out, "        name: {:?},", c.name).unwrap();
        writeln!(
            out,
            "        cost: Cost {{ pips: &[{pips_src}], generic: {generic}, x_count: {x_count} }},"
        )
        .unwrap();
        writeln!(out, "        types: &[{types_src}],").unwrap();
        writeln!(out, "        supertypes: &[{supertypes_src}],").unwrap();
        writeln!(out, "        power: {power_src},").unwrap();
        writeln!(out, "        toughness: {toughness_src},").unwrap();
        writeln!(out, "        is_land: {},", c.is_land).unwrap();
        writeln!(out, "        produces_mana: &[{produces_src}],").unwrap();
        writeln!(out, "        target_spec: {target_spec_src},").unwrap();
        writeln!(out, "        keywords: {},", keywords_for(&c.name)).unwrap();
        writeln!(out, "        spell_effect: {spell_effect_src},").unwrap();
        writeln!(out, "        mana_ability: {mana_ability_src},").unwrap();
        writeln!(out, "        alt_cost: {},", alt_cost_for(&c.name)).unwrap();
        writeln!(out, "        additional_cost: {},", additional_cost_for(&c.name)).unwrap();
        writeln!(out, "        flashback: {},", flashback_for(&c.name)).unwrap();
        writeln!(out, "        activated_abilities: {},", activated_abilities_for(&c.name)).unwrap();
        writeln!(out, "    }},").unwrap();
    }
    writeln!(out, "];").unwrap();
    writeln!(out).unwrap();

    // ---- name -> id --------------------------------------------------
    writeln!(out, "pub fn card_id_by_name(name: &str) -> Option<u16> {{").unwrap();
    writeln!(out, "    match name {{").unwrap();
    for (i, c) in cards.iter().enumerate() {
        writeln!(out, "        {:?} => Some({i}),", c.name).unwrap();
    }
    writeln!(out, "        _ => None,").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // ---- content hash --------------------------------------------------
    // Hashes gameplay-relevant fields only (name/cost/types/power/
    // toughness/is_land/produces_mana/decks), in array order, so
    // metadata-only regenerations of cards_v1.json (timestamps, java_file
    // paths, complexity tags) don't churn the constant.
    let mut canon = String::new();
    for c in cards {
        canon.push_str(&c.name);
        canon.push('|');
        canon.push_str(&c.mana_cost);
        canon.push('|');
        canon.push_str(&c.types.join(","));
        canon.push('|');
        canon.push_str(&c.power.map(|p| p.to_string()).unwrap_or_default());
        canon.push('|');
        canon.push_str(&c.toughness.map(|t| t.to_string()).unwrap_or_default());
        canon.push('|');
        canon.push_str(if c.is_land { "L" } else { "-" });
        canon.push('|');
        canon.push_str(&c.produces_mana.join(","));
        canon.push('|');
        canon.push_str(&c.decks.join(","));
        canon.push('\n');
    }
    let hash = fnv1a64(canon.as_bytes());
    writeln!(out, "pub const KERNEL_CARDDB_HASH: u64 = 0x{hash:016x};").unwrap();

    out
}

/// Minimal ordered-unique-values collection so this file doesn't need a
/// `BTreeSet` import just for one dedup+sort of a handful of i32s.
struct BTreeSetLike(Vec<i32>);
impl FromIterator<i32> for BTreeSetLike {
    fn from_iter<T: IntoIterator<Item = i32>>(iter: T) -> Self {
        let mut v: Vec<i32> = iter.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        BTreeSetLike(v)
    }
}
impl BTreeSetLike {
    fn values(&self) -> &[i32] {
        &self.0
    }
}

fn card_type_variant(t: &str) -> &'static str {
    match t {
        "Land" => "Land",
        "Creature" => "Creature",
        "Instant" => "Instant",
        "Sorcery" => "Sorcery",
        "Artifact" => "Artifact",
        "Enchantment" => "Enchantment",
        other => panic!("cards_v1.json: unknown card type {other:?}"),
    }
}

fn supertype_variant(t: &str) -> &'static str {
    match t {
        "Basic" => "Basic",
        "Snow" => "Snow",
        other => panic!("cards_v1.json: unknown supertype {other:?}"),
    }
}

fn color_variant(c: &str) -> &'static str {
    match c {
        "W" => "W",
        "U" => "U",
        "B" => "B",
        "R" => "R",
        "G" => "G",
        "C" => "C",
        other => panic!("cards_v1.json: unknown mana color {other:?}"),
    }
}

/// Parses a mana cost string like `"{1}{R}{R}"` into (pip expressions,
/// generic amount, X count). Pip expressions are emitted as literal Rust
/// source (e.g. `"Pip::Colored(ManaColor::R)"`) ready to splice into a
/// `&[...]` slice literal.
fn parse_cost(mana_cost: &str) -> (Vec<String>, u8, u8) {
    let mut pips = Vec::new();
    let mut generic: u32 = 0;
    let mut x_count: u8 = 0;

    let mut chars = mana_cost.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '{' {
            continue;
        }
        let mut tok = String::new();
        for c2 in chars.by_ref() {
            if c2 == '}' {
                break;
            }
            tok.push(c2);
        }

        if let Ok(n) = tok.parse::<u32>() {
            generic += n;
            continue;
        }
        if tok.eq_ignore_ascii_case("X") {
            x_count += 1;
            continue;
        }
        if let Some((a, b)) = tok.split_once('/') {
            if b.eq_ignore_ascii_case("P") {
                let color = color_variant(a);
                pips.push(format!("Pip::Phyrexian(ManaColor::{color})"));
            } else if a.chars().all(|ch| ch.is_ascii_digit()) {
                // Twobrid ({2/R}): not in the current pool. Approximate as
                // a colored pip; a future increment that needs twobrid
                // costs should give this its own Pip variant instead.
                let color = color_variant(b);
                pips.push(format!("Pip::Colored(ManaColor::{color})"));
            } else {
                let ca = color_variant(a);
                let cb = color_variant(b);
                pips.push(format!("Pip::Hybrid(ManaColor::{ca}, ManaColor::{cb})"));
            }
            continue;
        }
        let color = color_variant(tok.as_str());
        pips.push(format!("Pip::Colored(ManaColor::{color})"));
    }

    if generic > u8::MAX as u32 {
        panic!("cards_v1.json: generic cost {generic} overflows u8 in {mana_cost:?}");
    }
    (pips, generic as u8, x_count)
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
