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
    subtypes: Vec<String>,
    #[serde(default)]
    supertypes: Vec<String>,
    power: Option<i32>,
    toughness: Option<i32>,
    is_land: bool,
    #[serde(default)]
    produces_mana: Vec<String>,
    #[serde(default)]
    colors: Vec<String>,
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
    let json_path = Path::new(&manifest_dir)
        .join("..")
        .join("data")
        .join("cards_v1.json");
    println!("cargo:rerun-if-changed={}", json_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    let text = fs::read_to_string(&json_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", json_path.display()));
    let data: CardsFile = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("failed to parse {}: {e}", json_path.display()));

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

/// The Mono-Red Burn cards that get a real effect program. Relic of
/// Progenitus is the sole remaining deferred card -- present in `CARD_DEFS`
/// with correct metadata, not castable, per the kernel's fail-closed
/// invariant -- graveyard-card targeting doesn't fit any `TargetSpec` shape
/// built so far and it's sideboard-only, so it's lower priority than the 5
/// cards this increment adds.
enum Special {
    None,
    Mountain,
    /// Deals `amount` damage to any target (Lightning Bolt, Fiery Temper,
    /// Fireblast, Lava Dart). Fireblast's/Lava Dart's real alt cost /
    /// flashback, and Fiery Temper's Madness, are modeled via the separate
    /// `alt_cost_for`/`flashback_for`/`madness_cost_for` tables below
    /// (independent of `Special`, since a card's targeting/damage shape and
    /// its cost shape are orthogonal).
    BurnAnyTarget(i32),
    /// "Deals 3 damage to any target. Then that player or that permanent's
    /// controller may pay {R}{R}. If the player does, they may copy this
    /// spell and may choose a new target for that copy." (Chain Lightning,
    /// Rally-only). The mandatory damage is byte-for-byte
    /// `BurnAnyTarget(3)`'s shape; the optional-copy continuation is a
    /// conditional fail-closed (`effect::EffectOp::HaltIfAffectedCanPayCopyCost`)
    /// rather than either a silent "always decline" or an unconditional
    /// defer -- see that leaf's doc, and the increment report for the
    /// external-review citation (corpus non-occurrence doesn't justify
    /// skipping the check; off-trace/search play can reach an affordable
    /// board state even if no recorded game ever does).
    ChainLightning,
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
    DrawThenDiscard {
        draw: i32,
        discard: i32,
    },
    /// Grab the Prize: draw two cards, then (if the card discarded to pay
    /// the mandatory additional cost -- see `additional_cost_for` --
    /// wasn't a land) deal 2 damage to the opponent.
    GrabThePrize,
    /// "You may discard a card or sacrifice a land. If you do, draw two
    /// cards." (Highway Robbery's resolution effect; its Plot ability is
    /// modeled separately via `plot_cost_for`, since Plot is a cast-time
    /// alternative, not a resolution effect).
    HighwayRobbery,
    /// 1 damage to target player/planeswalker's controller and 1 damage to
    /// a creature that player controls; 3/3 instead with landfall this
    /// turn (Searing Blaze).
    SearingBlaze,
    /// "Choose one -- Counter target spell if it's blue; or destroy target
    /// permanent if it's blue." (Pyroblast: unfiltered targeting, checked
    /// at resolution).
    Pyroblast,
    /// "Choose one -- Counter target blue spell; or destroy target blue
    /// permanent." (Red Elemental Blast: filtered targeting).
    RedElementalBlast,
    /// "Deals 1 damage to each opponent and each creature and planeswalker
    /// they control." (End the Festivities, Rally-only). No planeswalker
    /// card exists in this 132-card pool, so the planeswalker half of the
    /// text is vacuously satisfied by construction, not modeled separately
    /// -- see `EffectOp::DamageOpponentAndTheirCreatures`.
    EndTheFestivities,
    /// "Deals 2 damage to any target. Metalcraft -- 4 instead if you
    /// control three or more artifacts." (Galvanic Blast, Rally-only).
    GalvanicBlast,
    /// "Create two 1/1 white Human Soldier creature tokens. Humans you
    /// control gain haste until end of turn." (Rally at the Hornburg,
    /// Rally-only -- the card the deck is named for).
    RallyAtTheHornburg,
    /// "Exile the top two cards of your library. Until the end of your next
    /// turn, you may play those cards." (Reckless Impulse, Rally-only).
    RecklessImpulse,
}

fn special_for(name: &str) -> Special {
    match name {
        // Great Furnace is a second "tap for {R}" land, textually identical
        // to Mountain (its only difference -- also being an Artifact, for
        // Metalcraft/affinity synergy -- is carried by `CardJson::types`,
        // not by this shape).
        "Mountain" | "Great Furnace" => Special::Mountain,
        "Lightning Bolt" => Special::BurnAnyTarget(3),
        "Fiery Temper" => Special::BurnAnyTarget(3),
        "Fireblast" => Special::BurnAnyTarget(4),
        "Lava Dart" => Special::BurnAnyTarget(1),
        "Chain Lightning" => Special::ChainLightning,
        // Every one of these resolves onto the battlefield with no other
        // cast-time effect: Burning-Tree Emissary's ETB mana add, Clockwork
        // Percussionist's dies-trigger impulse draw, Experimental
        // Synthesizer's enters-or-leaves impulse draw + its own sac
        // activated ability, Goblin Bushwhacker's Kicker/ETB pump, and
        // Goblin Tomb Raider's static self-boost are all layered on
        // separately (`keywords_for`, `kicker_cost_for`,
        // `activated_abilities_for`, `src/trigger.rs`'s `triggers_for`
        // table, and `engine::static_self_boost_for`) -- exactly the same
        // "spell/ability shape and triggered/static shape are orthogonal"
        // split `VanillaCreature`'s own doc already establishes for
        // Guttersnipe/Voldaren Epicure/Sneaky Snacker.
        "Guttersnipe"
        | "Masked Meower"
        | "Voldaren Epicure"
        | "Sneaky Snacker"
        | "Burning-Tree Emissary"
        | "Clockwork Percussionist"
        | "Experimental Synthesizer"
        | "Goblin Bushwhacker"
        | "Goblin Tomb Raider" => Special::VanillaCreature,
        "Faithless Looting" => Special::DrawThenDiscard {
            draw: 2,
            discard: 2,
        },
        "Grab the Prize" => Special::GrabThePrize,
        "Highway Robbery" => Special::HighwayRobbery,
        "Searing Blaze" => Special::SearingBlaze,
        "Pyroblast" => Special::Pyroblast,
        "Red Elemental Blast" => Special::RedElementalBlast,
        "End the Festivities" => Special::EndTheFestivities,
        "Galvanic Blast" => Special::GalvanicBlast,
        "Rally at the Hornburg" => Special::RallyAtTheHornburg,
        "Reckless Impulse" => Special::RecklessImpulse,
        _ => Special::None,
    }
}

/// Static combat/summoning-sickness keywords, verified against each card's
/// Java source (see the increment-3 report for the exact files read).
/// Masked Meower/Sneaky Snacker (Burn) and Clockwork Percussionist/Samurai
/// Token (Rally) carry an *unconditional* static keyword this way. Goblin
/// Bushwhacker's/Goblin Tomb Raider's haste is conditional (Kicker-gated,
/// or "as long as you control an artifact") and temporary/derived, so it is
/// deliberately NOT here -- see `engine::static_self_boost_for` and
/// `EffectOp::PumpControlled`'s `grant_haste` instead.
fn keywords_for(name: &str) -> &'static str {
    match name {
        "Masked Meower" | "Clockwork Percussionist" => "Keywords::HASTE",
        "Sneaky Snacker" => "Keywords::FLYING",
        "Samurai Token" => "Keywords::VIGILANCE",
        _ => "Keywords::NONE",
    }
}

/// `Some` Kicker cost source text (`CardDef::kicker_cost`), verified against
/// Java (`KickerAbility`). Only Goblin Bushwhacker has one this increment
/// ("Kicker {R}").
fn kicker_cost_for(name: &str) -> String {
    match name {
        "Goblin Bushwhacker" => cost_src("{R}"),
        _ => "None".to_string(),
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
/// share `ability_effect_draw_one`. Experimental Synthesizer's ("{2}{R},
/// Sacrifice Experimental Synthesizer: Create a 2/2 white Samurai creature
/// token with vigilance. Activate only as a sorcery.") is Rally's only
/// activated ability and the only one so far with `sorcery_speed_only:
/// true` -- see that field's doc in `card_def.rs`.
fn activated_abilities_for(name: &str) -> &'static str {
    match name {
        "Masked Meower" => {
            "&[ActivatedAbilityDef { \
                cost: &[CostComponent::DiscardCards(1), CostComponent::SacrificeSelf], \
                target_spec: TargetSpec::None, \
                effect: ability_effect_draw_one, \
                sorcery_speed_only: false, \
            }]"
        }
        "Blood Token" => {
            "&[ActivatedAbilityDef { \
                cost: &[CostComponent::Mana(Cost { pips: &[], generic: 1, x_count: 0 }), CostComponent::Tap, \
                        CostComponent::DiscardCards(1), CostComponent::SacrificeSelf], \
                target_spec: TargetSpec::None, \
                effect: ability_effect_draw_one, \
                sorcery_speed_only: false, \
            }]"
        }
        "Experimental Synthesizer" => {
            "&[ActivatedAbilityDef { \
                cost: &[CostComponent::Mana(Cost { pips: &[Pip::Colored(ManaColor::R)], generic: 2, x_count: 0 }), CostComponent::SacrificeSelf], \
                target_spec: TargetSpec::None, \
                effect: ability_effect_create_samurai_token, \
                sorcery_speed_only: true, \
            }]"
        }
        _ => "&[]",
    }
}

/// `Some` Plot cost source text (`CardDef::plot_cost`), verified against
/// Java (`PlotAbility`). Only Highway Robbery has one this increment
/// ("Plot {1}{R}").
fn plot_cost_for(name: &str) -> String {
    match name {
        "Highway Robbery" => cost_src("{1}{R}"),
        _ => "None".to_string(),
    }
}

/// `Some` Madness cost source text (`CardDef::madness_cost`), verified
/// against Java (`MadnessAbility`). Only Fiery Temper has one this
/// increment ("Madness {R}").
fn madness_cost_for(name: &str) -> String {
    match name {
        "Fiery Temper" => cost_src("{R}"),
        _ => "None".to_string(),
    }
}

/// `Some` second-mode source text (`CardDef::mode2`), verified against
/// Java. Pyroblast's and Red Elemental Blast's destroy modes.
fn mode2_for(name: &str) -> &'static str {
    match name {
        "Pyroblast" => "Some(ModeDef { target_spec: TargetSpec::AnyPermanent, effect: mode2_effect_pyroblast })",
        "Red Elemental Blast" => "Some(ModeDef { target_spec: TargetSpec::BluePermanent, effect: mode2_effect_red_elemental_blast })",
        _ => "None",
    }
}

/// Renders a mana cost string straight to a `Cost { .. }` literal wrapped in
/// `Some(..)`, for the one-off cost tables above (`plot_cost_for`/
/// `madness_cost_for`) that aren't full `CardJson` records.
fn cost_src(mana_cost: &str) -> String {
    let (pips, generic, x_count) = parse_cost(mana_cost);
    format!(
        "Some(Cost {{ pips: &[{}], generic: {generic}, x_count: {x_count} }})",
        pips.join(", ")
    )
}

fn codegen(cards: &[CardJson]) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "// GENERATED by build.rs from kernel/data/cards_v1.json. Do not edit by hand."
    )
    .unwrap();
    writeln!(out).unwrap();

    // Shared/one-off effect-program functions. Function *pointers* (not
    // owned EffectOp values) are what make a `static [CardDef; N]` array
    // possible: EffectOp contains Vec/Box and can't live in a const
    // initializer directly, but a `fn() -> Option<EffectOp>` can, and it
    // builds the (small) tree fresh each call.
    writeln!(out, "fn mana_ability_mountain() -> Option<EffectOp> {{").unwrap();
    writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
    writeln!(
        out,
        "        EffectOp::TapObject {{ object: ObjectRef::ThisSource }},"
    )
    .unwrap();
    writeln!(out, "        EffectOp::AddMana {{ player: PlayerRef::Controller, colors: vec![ManaColor::R] }},").unwrap();
    writeln!(out, "    ]))").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    writeln!(
        out,
        "fn spell_effect_vanilla_creature() -> Option<EffectOp> {{"
    )
    .unwrap();
    writeln!(out, "    Some(EffectOp::MoveObject {{ object: ObjectRef::ThisSource, to_zone: Zone::Battlefield }})").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    // Masked Meower's / Blood's activated ability both resolve to "draw a
    // card" once their (differing) costs are paid.
    writeln!(out, "fn ability_effect_draw_one() -> EffectOp {{").unwrap();
    writeln!(
        out,
        "    EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 1 }}"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    if cards.iter().any(|c| c.name == "Experimental Synthesizer") {
        // Experimental Synthesizer's sac ability: "Create a 2/2 white
        // Samurai creature token with vigilance."
        writeln!(
            out,
            "fn ability_effect_create_samurai_token() -> EffectOp {{"
        )
        .unwrap();
        writeln!(out, "    let samurai = crate::card_def::card_id_by_name(\"Samurai Token\").expect(\"Samurai Token in CARD_DEFS\");").unwrap();
        writeln!(
            out,
            "    EffectOp::CreateToken {{ token_def: samurai, controller: PlayerRef::Controller }}"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::ChainLightning))
    {
        // "Deals 3 damage to any target. Then that player or that
        // permanent's controller may pay {R}{R}. If the player does, they
        // may copy this spell...": mandatory damage first (identical shape
        // to `BurnAnyTarget(3)`), then the conditional halt-gate -- see
        // `EffectOp::HaltIfAffectedCanPayCopyCost`'s doc.
        writeln!(
            out,
            "fn spell_effect_chain_lightning() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 3 }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::HaltIfAffectedCanPayCopyCost {{ affected: TargetRef::Target(0) }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::EndTheFestivities))
    {
        // "Deals 1 damage to each opponent and each creature and
        // planeswalker they control." No planeswalker exists in this pool,
        // so `DamageOpponentAndTheirCreatures` (which only hits the
        // opponent + their creatures) already covers 100% of the reachable
        // text -- see `EffectOp::DamageOpponentAndTheirCreatures`'s doc.
        writeln!(
            out,
            "fn spell_effect_end_the_festivities() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(
            out,
            "    Some(EffectOp::DamageOpponentAndTheirCreatures {{ amount: 1 }})"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::GalvanicBlast))
    {
        // Metalcraft -- 4 damage instead of 2 if you control 3+ artifacts.
        writeln!(
            out,
            "fn spell_effect_galvanic_blast() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::ControlsArtifactCount(3),").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 4 }}),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 2 }}),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::RallyAtTheHornburg))
    {
        // "Create two 1/1 white Human Soldier creature tokens. Humans you
        // control gain haste until end of turn." The two just-created
        // tokens are themselves Human, so they're inside the "Humans you
        // control" set the pump snapshots -- see `EffectOp::PumpControlled`'s
        // doc for why sequencing (create both tokens, *then* pump) is what
        // makes that true.
        writeln!(
            out,
            "fn spell_effect_rally_at_the_hornburg() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    let human_soldier = crate::card_def::card_id_by_name(\"Human Soldier Token\").expect(\"Human Soldier Token in CARD_DEFS\");").unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(out, "        EffectOp::CreateToken {{ token_def: human_soldier, controller: PlayerRef::Controller }},").unwrap();
        writeln!(out, "        EffectOp::CreateToken {{ token_def: human_soldier, controller: PlayerRef::Controller }},").unwrap();
        writeln!(
            out,
            "        EffectOp::PumpControlled {{ filter: CreatureFilter::ControlledWithSubtype(Subtype::Human), power: 0, toughness: 0, grant_haste: true }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::RecklessImpulse))
    {
        // "Exile the top two cards of your library. Until the end of your
        // next turn, you may play those cards."
        writeln!(
            out,
            "fn spell_effect_reckless_impulse() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::ImpulseDraw {{ count: 2, duration: ImpulseDuration::UntilOwnersNextTurn }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    let draw_then_discard_shapes: Vec<(i32, i32)> = cards
        .iter()
        .filter_map(|c| match special_for(&c.name) {
            Special::DrawThenDiscard { draw, discard } => Some((draw, discard)),
            _ => None,
        })
        .collect();
    for (draw, discard) in draw_then_discard_shapes {
        writeln!(
            out,
            "fn spell_effect_draw_then_discard_{draw}_{discard}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: {draw} }},"
        )
        .unwrap();
        writeln!(
            out,
            "        EffectOp::DiscardCards {{ player: PlayerRef::Controller, count: {discard} }},"
        )
        .unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::GrabThePrize))
    {
        writeln!(
            out,
            "fn spell_effect_grab_the_prize() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "        EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 2 }},"
        )
        .unwrap();
        writeln!(out, "        EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "            cond: EffectCond::DiscardedNonLandForCost,"
        )
        .unwrap();
        writeln!(out, "            then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Opponent, amount: 2 }}),").unwrap();
        writeln!(
            out,
            "            else_: Box::new(EffectOp::Sequence(vec![])),"
        )
        .unwrap();
        writeln!(out, "        }},").unwrap();
        writeln!(out, "    ]))").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::HighwayRobbery))
    {
        // "You may discard a card or sacrifice a land. If you do, draw two
        // cards." -- DoIfCostPaid(OrCost(DiscardCardCost, SacrificeTargetCost)).
        writeln!(
            out,
            "fn spell_effect_highway_robbery() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::MayPayCostThen {{").unwrap();
        writeln!(out, "        discard: 1,").unwrap();
        writeln!(out, "        sacrifice_lands: 1,").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::DrawCards {{ player: PlayerRef::Controller, count: 2 }}),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::SearingBlaze))
    {
        // 1 damage to target player + 1 damage to target creature that
        // player controls; landfall bumps both to 3. The creature-damage
        // leaf is individually fizzle-guarded (608.2b: Searing Blaze still
        // hits the player even if the creature target became illegal --
        // the player target can't, so the whole spell can never fully
        // fizzle in this pool).
        writeln!(
            out,
            "fn spell_effect_searing_blaze() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::LandfallThisTurn,").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "            EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 3 }},"
        )
        .unwrap();
        writeln!(out, "            EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "                cond: EffectCond::TargetInZone(1, Zone::Battlefield),"
        )
        .unwrap();
        writeln!(out, "                then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Target(1), amount: 3 }}),").unwrap();
        writeln!(
            out,
            "                else_: Box::new(EffectOp::Sequence(vec![])),"
        )
        .unwrap();
        writeln!(out, "            }},").unwrap();
        writeln!(out, "        ])),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![").unwrap();
        writeln!(
            out,
            "            EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: 1 }},"
        )
        .unwrap();
        writeln!(out, "            EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "                cond: EffectCond::TargetInZone(1, Zone::Battlefield),"
        )
        .unwrap();
        writeln!(out, "                then: Box::new(EffectOp::DealDamage {{ target: TargetRef::Target(1), amount: 1 }}),").unwrap();
        writeln!(
            out,
            "                else_: Box::new(EffectOp::Sequence(vec![])),"
        )
        .unwrap();
        writeln!(out, "            }},").unwrap();
        writeln!(out, "        ])),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::Pyroblast))
    {
        // Primary mode: counter target spell if it's blue (unfiltered
        // targeting -- PyroblastCounterTargetEffect checks color at
        // resolution, not TargetSpell's legality). Same guarded-move shape
        // handles 608.2b fizzle (TargetInZone) and the color check
        // (TargetIsColor) together via EffectCond::And.
        writeln!(
            out,
            "fn spell_effect_pyroblast_mode1() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::And(").unwrap();
        writeln!(
            out,
            "            Box::new(EffectCond::TargetInZone(0, Zone::Stack)),"
        )
        .unwrap();
        writeln!(
            out,
            "            Box::new(EffectCond::TargetIsColor(0, ManaColor::U)),"
        )
        .unwrap();
        writeln!(out, "        ),").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Graveyard }}),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
        // Mode 2: destroy target permanent if it's blue (same shape, on the
        // battlefield instead of the stack).
        writeln!(out, "fn mode2_effect_pyroblast() -> EffectOp {{").unwrap();
        writeln!(out, "    EffectOp::Conditional {{").unwrap();
        writeln!(out, "        cond: EffectCond::And(").unwrap();
        writeln!(
            out,
            "            Box::new(EffectCond::TargetInZone(0, Zone::Battlefield)),"
        )
        .unwrap();
        writeln!(
            out,
            "            Box::new(EffectCond::TargetIsColor(0, ManaColor::U)),"
        )
        .unwrap();
        writeln!(out, "        ),").unwrap();
        writeln!(out, "        then: Box::new(EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Graveyard }}),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    if cards
        .iter()
        .any(|c| matches!(special_for(&c.name), Special::RedElementalBlast))
    {
        // Primary mode: counter target blue spell (color pre-filtered by
        // TargetSpec::BlueSpellOnStack at targeting time -- only the
        // 608.2b fizzle re-check is needed at resolution).
        writeln!(
            out,
            "fn spell_effect_red_elemental_blast_mode1() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(out, "    Some(EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "        cond: EffectCond::TargetInZone(0, Zone::Stack),"
        )
        .unwrap();
        writeln!(out, "        then: Box::new(EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Graveyard }}),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }})").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "fn mode2_effect_red_elemental_blast() -> EffectOp {{").unwrap();
        writeln!(out, "    EffectOp::Conditional {{").unwrap();
        writeln!(
            out,
            "        cond: EffectCond::TargetInZone(0, Zone::Battlefield),"
        )
        .unwrap();
        writeln!(out, "        then: Box::new(EffectOp::MoveObject {{ object: ObjectRef::Target(0), to_zone: Zone::Graveyard }}),").unwrap();
        writeln!(out, "        else_: Box::new(EffectOp::Sequence(vec![])),").unwrap();
        writeln!(out, "    }}").unwrap();
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
        writeln!(
            out,
            "fn spell_effect_burn_any_target_{amount}() -> Option<EffectOp> {{"
        )
        .unwrap();
        writeln!(
            out,
            "    Some(EffectOp::DealDamage {{ target: TargetRef::Target(0), amount: {amount} }})"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    // ---- CARD_DEFS -------------------------------------------------
    writeln!(out, "pub static CARD_DEFS: [CardDef; {}] = [", cards.len()).unwrap();
    for c in cards {
        let (pips, generic, x_count) = parse_cost(&c.mana_cost);
        let special = special_for(&c.name);

        let types_src = c
            .types
            .iter()
            .map(|t| format!("CardType::{}", card_type_variant(t)))
            .collect::<Vec<_>>()
            .join(", ");
        let subtypes_src = c
            .subtypes
            .iter()
            .map(|s| subtype_variant(s))
            .collect::<Vec<_>>()
            .join(", ");
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
        let colors_src = c
            .colors
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
            Special::None => (
                "TargetSpec::None",
                "no_effect".to_string(),
                "no_effect".to_string(),
            ),
            Special::Mountain => (
                "TargetSpec::None",
                "no_effect".to_string(),
                "mana_ability_mountain".to_string(),
            ),
            Special::BurnAnyTarget(amount) => (
                "TargetSpec::AnyTarget",
                format!("spell_effect_burn_any_target_{amount}"),
                "no_effect".to_string(),
            ),
            Special::ChainLightning => (
                "TargetSpec::AnyTarget",
                "spell_effect_chain_lightning".to_string(),
                "no_effect".to_string(),
            ),
            Special::VanillaCreature => (
                "TargetSpec::None",
                "spell_effect_vanilla_creature".to_string(),
                "no_effect".to_string(),
            ),
            Special::DrawThenDiscard { draw, discard } => (
                "TargetSpec::None",
                format!("spell_effect_draw_then_discard_{draw}_{discard}"),
                "no_effect".to_string(),
            ),
            Special::GrabThePrize => (
                "TargetSpec::None",
                "spell_effect_grab_the_prize".to_string(),
                "no_effect".to_string(),
            ),
            Special::HighwayRobbery => (
                "TargetSpec::None",
                "spell_effect_highway_robbery".to_string(),
                "no_effect".to_string(),
            ),
            Special::SearingBlaze => (
                "TargetSpec::PlayerThenTheirCreature",
                "spell_effect_searing_blaze".to_string(),
                "no_effect".to_string(),
            ),
            Special::Pyroblast => (
                "TargetSpec::AnySpellOnStack",
                "spell_effect_pyroblast_mode1".to_string(),
                "no_effect".to_string(),
            ),
            Special::RedElementalBlast => (
                "TargetSpec::BlueSpellOnStack",
                "spell_effect_red_elemental_blast_mode1".to_string(),
                "no_effect".to_string(),
            ),
            Special::EndTheFestivities => (
                "TargetSpec::None",
                "spell_effect_end_the_festivities".to_string(),
                "no_effect".to_string(),
            ),
            Special::GalvanicBlast => (
                "TargetSpec::AnyTarget",
                "spell_effect_galvanic_blast".to_string(),
                "no_effect".to_string(),
            ),
            Special::RallyAtTheHornburg => (
                "TargetSpec::None",
                "spell_effect_rally_at_the_hornburg".to_string(),
                "no_effect".to_string(),
            ),
            Special::RecklessImpulse => (
                "TargetSpec::None",
                "spell_effect_reckless_impulse".to_string(),
                "no_effect".to_string(),
            ),
        };

        writeln!(out, "    CardDef {{").unwrap();
        writeln!(out, "        name: {:?},", c.name).unwrap();
        writeln!(
            out,
            "        cost: Cost {{ pips: &[{pips_src}], generic: {generic}, x_count: {x_count} }},"
        )
        .unwrap();
        writeln!(out, "        types: &[{types_src}],").unwrap();
        writeln!(out, "        subtypes: &[{subtypes_src}],").unwrap();
        writeln!(out, "        supertypes: &[{supertypes_src}],").unwrap();
        writeln!(out, "        power: {power_src},").unwrap();
        writeln!(out, "        toughness: {toughness_src},").unwrap();
        writeln!(out, "        is_land: {},", c.is_land).unwrap();
        writeln!(out, "        produces_mana: &[{produces_src}],").unwrap();
        writeln!(out, "        colors: &[{colors_src}],").unwrap();
        writeln!(out, "        target_spec: {target_spec_src},").unwrap();
        writeln!(out, "        keywords: {},", keywords_for(&c.name)).unwrap();
        writeln!(out, "        spell_effect: {spell_effect_src},").unwrap();
        writeln!(out, "        mana_ability: {mana_ability_src},").unwrap();
        writeln!(out, "        alt_cost: {},", alt_cost_for(&c.name)).unwrap();
        writeln!(out, "        kicker_cost: {},", kicker_cost_for(&c.name)).unwrap();
        writeln!(
            out,
            "        additional_cost: {},",
            additional_cost_for(&c.name)
        )
        .unwrap();
        writeln!(out, "        flashback: {},", flashback_for(&c.name)).unwrap();
        writeln!(
            out,
            "        activated_abilities: {},",
            activated_abilities_for(&c.name)
        )
        .unwrap();
        writeln!(out, "        plot_cost: {},", plot_cost_for(&c.name)).unwrap();
        writeln!(out, "        madness_cost: {},", madness_cost_for(&c.name)).unwrap();
        writeln!(out, "        mode2: {},", mode2_for(&c.name)).unwrap();
        writeln!(out, "        is_token: {},", c.is_token).unwrap();
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
    // Hashes gameplay-relevant fields only (name/cost/types/subtypes/power/
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
        canon.push_str(&c.subtypes.join(","));
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

/// Maps a `cards_v1.json` subtype string to a `card_def::Subtype` variant --
/// a fully closed set (one variant per distinct string across the whole
/// 135-card pool this increment's data covers), so this panics on an
/// unrecognized value same as `card_type_variant`/`supertype_variant`/
/// `color_variant` -- see `Subtype`'s own doc for why it's closed rather
/// than named-variants-plus-string-fallback (a `&'static str` payload can't
/// derive `Deserialize`, and `Subtype` is embedded in `effect::EffectOp`,
/// which needs to).
fn subtype_variant(t: &str) -> &'static str {
    match t {
        "Ape" => "Subtype::Ape",
        "Aura" => "Subtype::Aura",
        "BIRD" => "Subtype::BirdAllCaps",
        "Bird" => "Subtype::Bird",
        "Blood" => "Subtype::Blood",
        "Cat" => "Subtype::Cat",
        "Detective" => "Subtype::Detective",
        "Dragon" => "Subtype::Dragon",
        "Drone" => "Subtype::Drone",
        "Druid" => "Subtype::Druid",
        "Eldrazi" => "Subtype::Eldrazi",
        "Elf" => "Subtype::Elf",
        "Equipment" => "Subtype::Equipment",
        "FAERIE" => "Subtype::FaerieAllCaps",
        "Faerie" => "Subtype::Faerie",
        "Food" => "Subtype::Food",
        "Forest" => "Subtype::Forest",
        "Gate" => "Subtype::Gate",
        "Goblin" => "Subtype::Goblin",
        "HUMAN" => "Subtype::HumanAllCaps",
        "Hero" => "Subtype::Hero",
        "Human" => "Subtype::Human",
        "Hydra" => "Subtype::Hydra",
        "Island" => "Subtype::Island",
        "Knight" => "Subtype::Knight",
        "MONK" => "Subtype::Monk",
        "MOONFOLK" => "Subtype::Moonfolk",
        "Monkey" => "Subtype::Monkey",
        "Mountain" => "Subtype::Mountain",
        "Myr" => "Subtype::Myr",
        "NINJA" => "Subtype::NinjaAllCaps",
        "Ninja" => "Subtype::Ninja",
        "Ouphe" => "Subtype::Ouphe",
        "Pirate" => "Subtype::Pirate",
        "Plains" => "Subtype::Plains",
        "ROGUE" => "Subtype::RogueAllCaps",
        "Ranger" => "Subtype::Ranger",
        "Rat" => "Subtype::Rat",
        "Rogue" => "Subtype::Rogue",
        "SERPENT" => "Subtype::Serpent",
        "Saga" => "Subtype::Saga",
        "Samurai" => "Subtype::Samurai",
        "Shaman" => "Subtype::Shaman",
        "Shapeshifter" => "Subtype::Shapeshifter",
        "Soldier" => "Subtype::Soldier",
        "Spider" => "Subtype::Spider",
        "Spirit" => "Subtype::Spirit",
        "Swamp" => "Subtype::Swamp",
        "Toy" => "Subtype::Toy",
        "Treefolk" => "Subtype::Treefolk",
        "Vampire" => "Subtype::Vampire",
        "WIZARD" => "Subtype::WizardAllCaps",
        "Warrior" => "Subtype::Warrior",
        "Wizard" => "Subtype::Wizard",
        "Zombie" => "Subtype::Zombie",
        other => panic!("cards_v1.json: unknown subtype {other:?}"),
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
