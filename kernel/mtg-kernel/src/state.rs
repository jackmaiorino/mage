//! Core game state. Every collection here is a `Vec` (or a fixed array) with
//! caller-controlled order; nothing is ever iterated via a `HashMap`, so two
//! states built from the same inputs serialize and hash identically (see
//! `state_hash` and the determinism test below).

use crate::ids::{Arena, ObjectId, PlayerId};
use crate::mana::ManaColor;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

pub const STARTING_LIFE: i32 = 20;

/// Exact diagnostic full-state hash contract written into privileged audit
/// artifacts. The algorithm is FNV-1a-64 over the compact UTF-8 JSON bytes of
/// `DiagnosticStateHashEnvelopeV1` below.
///
/// Changing the envelope, JSON representation, or digest algorithm requires a
/// new constant value and an audit-artifact schema bump. Policy artifacts do
/// not contain this privileged full-state diagnostic.
pub const DIAGNOSTIC_STATE_HASH_ALGORITHM: &str = "fnv1a64-serde-json-game-state-envelope-v1";
pub const DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Zone {
    Library,
    Hand,
    Battlefield,
    Graveyard,
    Stack,
    Exile,
    /// Emblems, suspended cards, etc. Unused by the 132-card pool today but
    /// cheap to carry so the enum doesn't need to change later.
    Command,
}

/// Counter families required by the Pauper pool. Signed storage is deliberate:
/// effect validation may reject an underflow without first converting between
/// unrelated integer shapes, while i16 leaves ample headroom for copied and
/// doubled counter effects.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Counters {
    pub plus1_plus1: i16,
    pub minus1_minus1: i16,
    pub minus0_minus1: i16,
    pub stun: i16,
    pub lore: i16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectLinkV4 {
    pub object: ObjectId,
    pub zone_change_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbilityKindV4 {
    Mana,
    Activated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AbilityUseV4 {
    pub ability_kind: AbilityKindV4,
    pub ability_index: u16,
    pub uses: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GoadStateV4 {
    pub player: PlayerId,
    /// First turn number at whose appropriate untap boundary this goad no
    /// longer applies. Duration stays explicit rather than being inferred
    /// from presence in the vector.
    pub expires_at_turn: u32,
}

/// Schema-v4 dynamic object substrate. Base colors, subtypes, and token
/// identity are materialized from the registry at object creation, rather
/// than assigning zero a meaning that would change when mechanics arrive.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObjectStateV4 {
    pub is_token: bool,
    pub face_index: u8,
    pub effective_color_mask: u8,
    /// Sorted, unique `card_def::Subtype::stable_id()` values.
    pub effective_subtype_ids: Vec<u16>,
    pub chosen_color: Option<ManaColor>,
    pub entered_battlefield_turn: Option<u32>,
    /// Sorted by `(ability_kind, ability_index)`; absent abilities have zero uses.
    pub ability_uses_this_turn: Vec<AbilityUseV4>,
    pub skip_next_untap: bool,
    /// Sorted by `(player, expires_at_turn)`.
    pub goaded_by: Vec<GoadStateV4>,
    pub attached_to: Option<ObjectLinkV4>,
    pub exiled_by: Option<ObjectLinkV4>,
    pub ward_generic: u16,
    /// `None` means derive the ordinary one-blocker baseline (or Menace)
    /// from effective keywords; `Some` is a rules-effect override.
    pub minimum_blockers_override: Option<u8>,
    /// W/U/B/R/G/C bits naming the land types this object can landwalk.
    pub landwalk_mask: u8,
}

impl ObjectStateV4 {
    pub fn from_card_def(card_def: u16) -> ObjectStateV4 {
        let def = &crate::card_def::CARD_DEFS[card_def as usize];
        let mut subtype_ids: Vec<u16> = def
            .subtypes
            .iter()
            .map(|subtype| subtype.stable_id())
            .collect();
        subtype_ids.sort_unstable();
        subtype_ids.dedup();
        ObjectStateV4 {
            is_token: def.is_token,
            face_index: 0,
            effective_color_mask: crate::card_def::mana_colors_mask(def.colors),
            effective_subtype_ids: subtype_ids,
            chosen_color: None,
            entered_battlefield_turn: None,
            ability_uses_this_turn: Vec::new(),
            skip_next_untap: false,
            goaded_by: Vec::new(),
            attached_to: None,
            exiled_by: None,
            ward_generic: 0,
            minimum_blockers_override: None,
            landwalk_mask: 0,
        }
    }

    pub fn reset_for_zone_change(&mut self, card_def: u16, to_zone: Zone, turn: u32) {
        let base = ObjectStateV4::from_card_def(card_def);
        *self = base;
        if to_zone == Zone::Battlefield {
            self.entered_battlefield_turn = Some(turn);
        }
    }

    pub fn note_ability_use(&mut self, ability_kind: AbilityKindV4, ability_index: u16) {
        let key = (ability_kind, ability_index);
        match self
            .ability_uses_this_turn
            .binary_search_by_key(&key, |entry| (entry.ability_kind, entry.ability_index))
        {
            Ok(index) => {
                self.ability_uses_this_turn[index].uses =
                    self.ability_uses_this_turn[index].uses.saturating_add(1);
            }
            Err(index) => self.ability_uses_this_turn.insert(
                index,
                AbilityUseV4 {
                    ability_kind,
                    ability_index,
                    uses: 1,
                },
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameObject {
    /// Index into the (not-yet-built) card database.
    pub card_def: u16,
    /// Debug-only display name; not used for gameplay logic.
    pub name: String,
    pub owner: PlayerId,
    pub controller: PlayerId,
    pub zone: Zone,
    pub tapped: bool,
    pub summoning_sick: bool,
    pub damage: u16,
    pub counters: Counters,
    pub attachments: Vec<ObjectId>,
    pub v4: ObjectStateV4,
    /// `Some(turn)` iff this card was Plotted (`PlotAbility`) on kernel
    /// round `turn` -- set by `engine::plot_spell`, read by
    /// `engine::is_plotted_castable_now` (castable from exile for free at
    /// sorcery speed, but never the same turn it was plotted). `None` for
    /// every card that has never been Plotted. Only Highway Robbery in this
    /// pool has `CardDef::plot_cost`, so this is `None` for every other
    /// card for the whole game.
    pub plotted_turn: Option<u32>,
    /// How many times this object has ever changed zones (CR 400.7's own
    /// `zoneChangeCounter` concept, ported deliberately -- see `engine::
    /// legal_blockers_for`'s sibling doc mentioning the reference engine's
    /// version). Bumped once per `event::commit_zone_change` call for this
    /// id, regardless of which zones. Read by `engine::PlayPermission::
    /// zone_change_generation`: a permission snapshots this value the
    /// instant it's granted, and is only ever honored while the object's
    /// *current* count still matches -- any further zone change (playing
    /// the card through the permission, or anything else) silently voids
    /// it, structurally, without this module needing to remember to remove
    /// the stale entry.
    pub zone_change_count: u32,
}

impl GameObject {
    fn new_in_library(card_def: u16, name: String, owner: PlayerId) -> GameObject {
        GameObject {
            card_def,
            name,
            owner,
            controller: owner,
            zone: Zone::Library,
            tapped: false,
            summoning_sick: false,
            damage: 0,
            counters: Counters::default(),
            attachments: Vec::new(),
            v4: ObjectStateV4::from_card_def(card_def),
            plotted_turn: None,
            zone_change_count: 0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DungeonStateV4 {
    pub dungeon_id: Option<u16>,
    pub room_id: Option<u16>,
    /// Sorted, unique stable dungeon ids.
    pub completed_dungeons: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayerState {
    pub life: i32,
    /// Index 0 = TOP of library. Drawing removes index 0.
    pub library: Vec<ObjectId>,
    /// Insertion order, oldest first; order is player-visible info in traces.
    pub hand: Vec<ObjectId>,
    /// Insertion order (order a permanent entered), not board position.
    pub battlefield: Vec<ObjectId>,
    /// Order matters (top of graveyard, Karmic Guide-style effects, etc).
    /// Last element is the most-recently-added (top).
    pub graveyard: Vec<ObjectId>,
    /// [W, U, B, R, G, C].
    pub mana_pool: [u8; 6],
    pub has_lost: bool,
    pub lands_played_this_turn: u8,
    /// Set by `event::commit` when a `Draw` was attempted against an empty
    /// library. Checked (and turned into `has_lost`) by
    /// `trigger::sba_fixed_point` (rule 704.5c).
    pub drew_from_empty: bool,
    /// Cards successfully drawn since the current turn began (both
    /// players' counters reset together at every `Step::Untap`, matching
    /// the reference engine's `DrawNthCardWatcher`, which is a
    /// whole-game-scoped watcher whose backing map is cleared once per
    /// turn boundary). Used by `trigger::TriggerCondition::DrawNth`
    /// (Sneaky Snacker: "whenever you draw your third card in a turn").
    pub draws_this_turn: u32,
    pub spells_cast_this_turn: u16,
    pub dungeon: DungeonStateV4,
}

impl PlayerState {
    fn new(life: i32) -> PlayerState {
        PlayerState {
            life,
            library: Vec::new(),
            hand: Vec::new(),
            battlefield: Vec::new(),
            graveyard: Vec::new(),
            mana_pool: [0; 6],
            has_lost: false,
            lands_played_this_turn: 0,
            drew_from_empty: false,
            draws_this_turn: 0,
            spells_cast_this_turn: 0,
            dungeon: DungeonStateV4::default(),
        }
    }
}

/// Steps the RL decision stream actually visits (see golden-trace `phase`
/// field). Untap/Cleanup are included even though the reference engine
/// rarely stops for priority there, since the kernel still transitions
/// through them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Step {
    Untap,
    Upkeep,
    Draw,
    Main1,
    BeginCombat,
    DeclareAttackers,
    DeclareBlockers,
    CombatDamage,
    EndCombat,
    Main2,
    End,
    Cleanup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Target {
    Object(ObjectId),
    Player(PlayerId),
}

/// Publicly distinguishable origin of a stack item. This is stamped by the
/// creation path, not inferred from card text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StackItemKind {
    Spell,
    ActivatedAbility,
    TriggeredAbility,
    MadnessOffer,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CastMethodV4 {
    #[default]
    Normal,
    Alternative,
    Flashback,
    Madness,
    Plotted,
    Escape,
    Bestow,
    Omen,
}

/// Historical identity of one object used to pay a cost. It belongs to the
/// stack incarnation and must not follow the arena object through later zone
/// changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PaidCostRefV4 {
    pub object: ObjectId,
    pub card_def: u16,
    pub owner: PlayerId,
    pub controller: PlayerId,
    /// Zone and generation immediately after the cost finished moving the
    /// object. These are historical provenance, not a live object lookup.
    pub zone: Zone,
    pub zone_change_count: u32,
    /// P0/P1 bitmask frozen at payment time. Public payment destinations are
    /// visible to both seats; hidden destinations retain only observers who
    /// actually knew that exact incarnation.
    pub visible_to_mask: u8,
}

/// Cast/payment provenance that belongs to the stack incarnation, not the
/// underlying card object. Abilities use `cast_method: None`; spells always
/// carry an explicit method (ordinary casts are `Some(Normal)`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackStateV4 {
    pub cast_method: Option<CastMethodV4>,
    pub face_index: u8,
    pub x_value: u16,
    pub paid_cost_refs: Vec<PaidCostRefV4>,
}

impl StackStateV4 {
    pub fn spell(cast_method: CastMethodV4) -> StackStateV4 {
        StackStateV4 {
            cast_method: Some(cast_method),
            ..StackStateV4::default()
        }
    }
}

/// Minimal stack entry: enough to represent "something is on the stack with
/// these targets." Resolution/effect semantics belong to the step layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackItem {
    pub kind: StackItemKind,
    pub source: ObjectId,
    pub controller: PlayerId,
    pub targets: Vec<Target>,
    /// True iff this spell is a virtual copy rather than a physical card.
    /// Copies receive their own stable arena object so stack targeting never
    /// aliases the original spell, but they cease to exist instead of moving
    /// to a graveyard/exile when they resolve, fizzle, or are countered.
    /// Always false for abilities and ordinary cast spells.
    pub is_copy: bool,
    /// `Some` for a triggered ability or a non-mana activated ability
    /// (Masked Meower's, the Blood token's -- see `card_def::
    /// ActivatedAbilityDef`): an inline effect program that isn't looked
    /// up from a `CARD_DEFS` entry at resolution time. `None` for a spell,
    /// whose program is looked up from
    /// `card_def::CARD_DEFS[objects[source].card_def].spell_effect`
    /// instead.
    pub inline_effect: Option<crate::effect::EffectOp>,
    /// Cards discarded to pay this cast's mandatory additional cost (Grab
    /// the Prize), threaded through to `effect::ExecCtx::discarded` at
    /// resolution time. Empty for everything else.
    pub discarded: Vec<ObjectId>,
    /// True iff this spell was cast via flashback: on resolution, an
    /// instant/sorcery goes to exile instead of the graveyard (702.10e).
    pub is_flashback: bool,
    /// Which mode this cast chose, for a modal spell (`card_def::CardDef::
    /// mode2`): `0` = the card's primary `target_spec`/`spell_effect`, `1`
    /// = `mode2`. Always `0` for a non-modal card (`mode2 == None`), which
    /// is every card in this pool except Pyroblast/Red Elemental Blast.
    pub mode_chosen: u8,
    /// True iff this item is a Madness triggered-ability offer (`card_def::
    /// CardDef::madness_cost`), not a normal spell/ability -- pushed by
    /// `engine::push_trigger_onto_stack` from a `trigger::PendingTrigger`
    /// whose own `is_madness_offer` is set (see that field's doc). Resolving
    /// this item (both players pass priority with it on top, same as any
    /// other stack object -- 117.5) is a real player decision
    /// (`engine::Decision::ChooseMadnessCast`: cast `source` for its madness
    /// cost, or let it go to the graveyard), not a fixed `EffectOp` program,
    /// so `inline_effect` is always `None` here and `engine::
    /// advance_until_decision`'s stack-resolution check special-cases this
    /// flag before ever calling `resolve_top_of_stack`. `false` for every
    /// other stack item (a spell, a normal triggered ability, or a non-mana
    /// activated ability).
    pub madness_offer: bool,
    /// True iff this stack item's own cast paid `card_def::CardDef::
    /// kicker_cost` (Goblin Bushwhacker). Cast-time metadata (CR 702.33/
    /// 601.2f), not a durable fact stored anywhere keyed by stable object
    /// id: `engine::finalize_cast` stamps it on the spell's own item;
    /// `engine::resolve_top_of_stack` copies it into that resolution's
    /// `effect::ExecCtx::kicked` and (via `EngineState::
    /// pending_kicked_source`) into the ETB trigger's own `trigger::
    /// PendingTrigger`, whose `engine::push_trigger_onto_stack` copies it
    /// again onto *that* trigger's stack item -- so by the time the
    /// trigger itself resolves, its own `ExecCtx::kicked` is correctly set,
    /// with nothing left over anywhere once both items have resolved.
    /// `false` for every other stack item (no other card in this pool has
    /// Kicker).
    pub kicked: bool,
    pub v4: StackStateV4,
}

/// One card identity an observer is entitled to know at a specific library
/// position. Knowledge is stored separately from the omniscient library order
/// so observations can project only the acting player's information without
/// leaking the opponent's private look/reorder choices.
///
/// `zone_change_count` binds the fact to this exact object incarnation. A card
/// that leaves and later returns to a library cannot accidentally resurrect an
/// older knowledge entry merely because the arena id is stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LibraryKnowledgeEntry {
    pub position: u32,
    pub object: ObjectId,
    pub zone_change_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HandKnowledgeEntry {
    pub object: ObjectId,
    pub zone_change_count: u32,
}

/// Counter-based, seedable, serializable PRNG (SplitMix64). Deterministic:
/// same seed and same call sequence always produce the same stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn seed(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameState {
    pub objects: Arena<GameObject>,
    pub players: [PlayerState; 2],
    pub turn: u32,
    pub active_player: PlayerId,
    pub priority_player: PlayerId,
    pub step: Step,
    pub stack: Vec<StackItem>,
    pub exile: Vec<ObjectId>,
    pub command: Vec<ObjectId>,
    /// The player who currently holds the initiative, if any.
    pub initiative: Option<PlayerId>,
    /// Observer x library-owner knowledge. Each inner vector is sorted by
    /// `position` and contains no duplicate positions or object incarnations.
    /// This is full engine state (and therefore snapshot/hash state), but RL
    /// observations expose only the row for the acting observer.
    pub library_knowledge: [[Vec<LibraryKnowledgeEntry>; 2]; 2],
    /// Observer x hand-owner identity knowledge. As with library knowledge,
    /// observation code may project only the acting observer's row.
    pub hand_knowledge: [[Vec<HandKnowledgeEntry>; 2]; 2],
    pub rng: SplitMix64,
    /// Priority/stack/turn-structure bookkeeping and the propose-commit
    /// event log, all owned by the `engine`/`event`/`trigger` modules. See
    /// `engine::EngineState`.
    pub engine: crate::engine::EngineState,
}

impl PaidCostRefV4 {
    pub fn capture(state: &GameState, object_id: ObjectId) -> PaidCostRefV4 {
        let object = state.objects.get(object_id);
        let visible_to_mask = match object.zone {
            Zone::Battlefield | Zone::Graveyard | Zone::Stack | Zone::Exile | Zone::Command => 0b11,
            Zone::Hand => [PlayerId::P0, PlayerId::P1]
                .into_iter()
                .filter(|&observer| {
                    observer == object.owner
                        || state
                            .known_hand_cards(observer, object.owner)
                            .iter()
                            .any(|entry| {
                                entry.object == object_id
                                    && entry.zone_change_count == object.zone_change_count
                            })
                })
                .fold(0, |mask, observer| mask | (1 << observer.index())),
            Zone::Library => [PlayerId::P0, PlayerId::P1]
                .into_iter()
                .filter(|&observer| {
                    state
                        .known_library_cards(observer, object.owner)
                        .iter()
                        .any(|entry| {
                            entry.object == object_id
                                && entry.zone_change_count == object.zone_change_count
                        })
                })
                .fold(0, |mask, observer| mask | (1 << observer.index())),
        };
        PaidCostRefV4 {
            object: object_id,
            card_def: object.card_def,
            owner: object.owner,
            controller: object.controller,
            zone: object.zone,
            zone_change_count: object.zone_change_count,
            visible_to_mask,
        }
    }

    pub fn visible_to(self, observer: PlayerId) -> bool {
        self.visible_to_mask & (1 << observer.index()) != 0
    }
}

impl GameState {
    /// Builds a fresh pre-game state from two post-shuffle library orders
    /// (index 0 = top, matching `GoldenTrace::opening_library`). Arena ids
    /// are assigned contiguously in library order, player 0 first, so the
    /// id assignment is fully determined by the two input vecs.
    pub fn new_from_libraries(
        lib0: &[u16],
        lib1: &[u16],
        names: impl Fn(u16) -> String,
        seed: u64,
    ) -> GameState {
        let mut objects = Arena::with_capacity(lib0.len() + lib1.len());
        let mut library0 = Vec::with_capacity(lib0.len());
        let mut library1 = Vec::with_capacity(lib1.len());

        for &card_def in lib0 {
            let id = objects.push(GameObject::new_in_library(
                card_def,
                names(card_def),
                PlayerId::P0,
            ));
            library0.push(id);
        }
        for &card_def in lib1 {
            let id = objects.push(GameObject::new_in_library(
                card_def,
                names(card_def),
                PlayerId::P1,
            ));
            library1.push(id);
        }

        let mut player0 = PlayerState::new(STARTING_LIFE);
        player0.library = library0;
        let mut player1 = PlayerState::new(STARTING_LIFE);
        player1.library = library1;

        GameState {
            objects,
            players: [player0, player1],
            turn: 1,
            active_player: PlayerId::P0,
            priority_player: PlayerId::P0,
            step: Step::Untap,
            stack: Vec::new(),
            exile: Vec::new(),
            command: Vec::new(),
            initiative: None,
            library_knowledge: std::array::from_fn(|_| {
                std::array::from_fn(|_| Vec::<LibraryKnowledgeEntry>::new())
            }),
            hand_knowledge: std::array::from_fn(|_| {
                std::array::from_fn(|_| Vec::<HandKnowledgeEntry>::new())
            }),
            rng: SplitMix64::seed(seed),
            engine: crate::engine::EngineState::default(),
        }
    }

    /// Removes the top card of `player`'s library and puts it in hand.
    /// Returns `None` (no state change) if the library is empty.
    pub fn draw_card(&mut self, player: PlayerId) -> Option<ObjectId> {
        let id = {
            let ps = &mut self.players[player.index()];
            if ps.library.is_empty() {
                return None;
            }
            let id = ps.library.remove(0);
            ps.hand.push(id);
            id
        };
        self.transfer_library_knowledge_to_hand(player, 0, id);
        self.note_library_removal(player, 0);
        self.clear_object_relations(id);
        let turn = self.turn;
        let new_generation = {
            let object = self.objects.get_mut(id);
            object.zone = Zone::Hand;
            object.zone_change_count += 1;
            object
                .v4
                .reset_for_zone_change(object.card_def, Zone::Hand, turn);
            object.zone_change_count
        };
        for observer in [PlayerId::P0, PlayerId::P1] {
            for entry in &mut self.hand_knowledge[observer.index()][player.index()] {
                if entry.object == id {
                    entry.zone_change_count = new_generation;
                }
            }
        }
        Some(id)
    }

    pub fn known_hand_cards(&self, observer: PlayerId, owner: PlayerId) -> &[HandKnowledgeEntry] {
        &self.hand_knowledge[observer.index()][owner.index()]
    }

    /// Records a revealed hand identity for exactly one observer. The entry
    /// is incarnation-bound and stored in arena-id order for deterministic
    /// snapshots and observation hashes.
    pub fn reveal_hand_card(
        &mut self,
        observer: PlayerId,
        owner: PlayerId,
        object: ObjectId,
    ) -> Result<(), String> {
        let live = self
            .players
            .get(owner.index())
            .is_some_and(|player| player.hand.contains(&object));
        let Some(card) = self.objects.try_get(object) else {
            return Err(format!("cannot reveal missing hand object {object}"));
        };
        if !live || card.owner != owner || card.zone != Zone::Hand {
            return Err(format!("{object} is not a live card in {owner:?}'s hand"));
        }
        if observer == owner {
            // `own_hand` already carries the complete private hand. Keeping a
            // second copy would add no information and creates two facts that
            // future hand mutation would have to reconcile.
            return Ok(());
        }
        let entries = &mut self.hand_knowledge[observer.index()][owner.index()];
        if entries.iter().any(|entry| entry.object == object) {
            return Ok(());
        }
        entries.push(HandKnowledgeEntry {
            object,
            zone_change_count: card.zone_change_count,
        });
        entries.sort_by_key(|entry| entry.object);
        Ok(())
    }

    fn transfer_library_knowledge_to_hand(
        &mut self,
        owner: PlayerId,
        position: usize,
        object: ObjectId,
    ) {
        let generation = self.objects.get(object).zone_change_count;
        for observer in [PlayerId::P0, PlayerId::P1] {
            if observer == owner {
                continue;
            }
            if self.library_knowledge[observer.index()][owner.index()]
                .iter()
                .any(|entry| {
                    entry.position as usize == position
                        && entry.object == object
                        && entry.zone_change_count == generation
                })
            {
                let entries = &mut self.hand_knowledge[observer.index()][owner.index()];
                if !entries.iter().any(|entry| entry.object == object) {
                    entries.push(HandKnowledgeEntry {
                        object,
                        zone_change_count: generation,
                    });
                    entries.sort_by_key(|entry| entry.object);
                }
            }
        }
    }

    pub(crate) fn forget_hand_object(&mut self, object: ObjectId) {
        for observer in [PlayerId::P0, PlayerId::P1] {
            for owner in [PlayerId::P0, PlayerId::P1] {
                self.hand_knowledge[observer.index()][owner.index()]
                    .retain(|entry| entry.object != object);
            }
        }
    }

    /// Clears zone-incarnation relations from the moving object and all
    /// reverse references to it. This is called before/while zone-change
    /// generation advances so an attachment or exile provenance link can
    /// never silently reconnect to a later incarnation.
    pub(crate) fn clear_object_relations(&mut self, object: ObjectId) {
        for (_, candidate) in self.objects.iter_mut() {
            candidate.attachments.retain(|&attached| attached != object);
            if candidate
                .v4
                .attached_to
                .is_some_and(|link| link.object == object)
            {
                candidate.v4.attached_to = None;
            }
            if candidate
                .v4
                .exiled_by
                .is_some_and(|link| link.object == object)
            {
                candidate.v4.exiled_by = None;
            }
        }
        let moving = self.objects.get_mut(object);
        moving.attachments.clear();
        moving.v4.attached_to = None;
        moving.v4.exiled_by = None;
    }

    /// Returns the acting observer's currently valid, position-sorted facts
    /// about `owner`'s library. Callers should not inspect a different
    /// observer's row while producing a perspective-limited observation.
    pub fn known_library_cards(
        &self,
        observer: PlayerId,
        owner: PlayerId,
    ) -> &[LibraryKnowledgeEntry] {
        &self.library_knowledge[observer.index()][owner.index()]
    }

    /// Records that `observer` looked at the first `count` cards of `owner`'s
    /// library in their current order. Existing knowledge below that prefix is
    /// retained because revealing a prefix does not randomize the rest.
    pub fn reveal_library_top(&mut self, observer: PlayerId, owner: PlayerId, count: usize) {
        let count = count.min(self.players[owner.index()].library.len());
        let mut entries = self.library_knowledge[observer.index()][owner.index()].clone();
        entries.retain(|entry| entry.position as usize >= count);
        for position in 0..count {
            let object = self.players[owner.index()].library[position];
            entries.push(LibraryKnowledgeEntry {
                position: position as u32,
                object,
                zone_change_count: self.objects.get(object).zone_change_count,
            });
        }
        entries.sort_by_key(|entry| entry.position);
        self.library_knowledge[observer.index()][owner.index()] = entries;
    }

    /// Reorders exactly the top `ordered.len()` cards. The supplied ids must
    /// be a permutation of the current prefix. Observers in `revealed_to`
    /// learn the resulting order; everyone else loses facts inside the
    /// changed prefix while retaining facts below it.
    pub fn reorder_library_top(
        &mut self,
        owner: PlayerId,
        ordered: &[ObjectId],
        revealed_to: &[PlayerId],
    ) -> Result<(), String> {
        let count = ordered.len();
        let library = &self.players[owner.index()].library;
        if count > library.len() {
            return Err(format!(
                "cannot reorder {count} cards in a library of {}",
                library.len()
            ));
        }
        let mut expected = library[..count].to_vec();
        let mut actual = ordered.to_vec();
        expected.sort_unstable();
        actual.sort_unstable();
        if actual != expected {
            return Err("reordered library prefix is not an exact permutation".to_string());
        }

        self.players[owner.index()].library[..count].copy_from_slice(ordered);
        for observer in [PlayerId::P0, PlayerId::P1] {
            let knows_result = revealed_to.contains(&observer);
            let entries = &mut self.library_knowledge[observer.index()][owner.index()];
            entries.retain(|entry| entry.position as usize >= count);
            if knows_result {
                for (position, &object) in ordered.iter().enumerate() {
                    entries.push(LibraryKnowledgeEntry {
                        position: position as u32,
                        object,
                        zone_change_count: self.objects.get(object).zone_change_count,
                    });
                }
                entries.sort_by_key(|entry| entry.position);
            }
        }
        Ok(())
    }

    /// Deterministically shuffles one library with the game RNG and clears
    /// every observer's facts about it before any later effect reveals a new
    /// prefix.
    pub fn shuffle_library(&mut self, owner: PlayerId) {
        let len = self.players[owner.index()].library.len();
        for i in (1..len).rev() {
            let j = (self.rng.next_u64() % (i as u64 + 1)) as usize;
            self.players[owner.index()].library.swap(i, j);
        }
        self.clear_library_knowledge(owner);
    }

    /// Clears all observers' facts about one library. Used for shuffles and
    /// conservative invalidation when a generic zone change cannot preserve
    /// what each observer knows about the mutation.
    pub(crate) fn clear_library_knowledge(&mut self, owner: PlayerId) {
        for observer in [PlayerId::P0, PlayerId::P1] {
            self.library_knowledge[observer.index()][owner.index()].clear();
        }
    }

    /// Updates position facts after a card at a publicly determined library
    /// position leaves. Top-card draws, mills, and impulse exile all use this
    /// exact shift operation.
    pub(crate) fn note_library_removal(&mut self, owner: PlayerId, position: usize) {
        for observer in [PlayerId::P0, PlayerId::P1] {
            let entries = &mut self.library_knowledge[observer.index()][owner.index()];
            entries.retain(|entry| entry.position as usize != position);
            for entry in entries {
                if entry.position as usize > position {
                    entry.position -= 1;
                }
            }
        }
    }

    /// Placeholder zone transition (hand -> battlefield) used by tests and
    /// future step logic; no mana cost / land-drop accounting here.
    pub fn move_hand_to_battlefield(&mut self, player: PlayerId, id: ObjectId) -> bool {
        let ps = &mut self.players[player.index()];
        let Some(pos) = ps.hand.iter().position(|&h| h == id) else {
            return false;
        };
        ps.hand.remove(pos);
        ps.battlefield.push(id);
        self.forget_hand_object(id);
        self.clear_object_relations(id);
        let turn = self.turn;
        let obj = self.objects.get_mut(id);
        obj.zone = Zone::Battlefield;
        obj.summoning_sick = true;
        obj.zone_change_count += 1;
        obj.v4
            .reset_for_zone_change(obj.card_def, Zone::Battlefield, turn);
        true
    }

    /// Fast in-process FNV-1a over Rust's derived `Hash`. This is used only for
    /// hot-path mutation/rollback checks and deliberately is not an artifact
    /// interchange contract; it may depend on target width and Rust hashing
    /// details.
    pub fn state_hash(&self) -> u64 {
        let mut hasher = Fnv1a64::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Cross-platform privileged audit hash: FNV-1a-64 over deterministic
    /// compact JSON bytes for a versioned, full-state envelope. Struct field
    /// order, enum spellings, and stored `Vec`/array order are part of this
    /// versioned artifact contract.
    pub fn diagnostic_state_hash(&self) -> u64 {
        fnv1a64(&diagnostic_state_hash_bytes(self))
    }
}

struct Fnv1a64 {
    state: u64,
}

impl Fnv1a64 {
    fn new() -> Fnv1a64 {
        Fnv1a64 {
            state: 0xcbf29ce484222325,
        }
    }
}

impl Hasher for Fnv1a64 {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u64::from(byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }
}

#[derive(Serialize)]
struct DiagnosticStateHashEnvelopeV1<'a> {
    schema_version: u32,
    state: &'a GameState,
}

fn diagnostic_state_hash_bytes(state: &GameState) -> Vec<u8> {
    serde_json::to_vec(&DiagnosticStateHashEnvelopeV1 {
        schema_version: DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION,
        state,
    })
    .expect("GameState diagnostic hash envelope must serialize")
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut state = 0xcbf29ce484222325;
    for &byte in bytes {
        state ^= u64::from(byte);
        state = state.wrapping_mul(0x100000001b3);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_card_libraries() -> (Vec<u16>, Vec<u16>) {
        (vec![1, 2, 3], vec![4, 5, 6, 7])
    }

    fn debug_names(card_def: u16) -> String {
        format!("card-{card_def}")
    }

    #[test]
    fn new_from_libraries_assigns_ids_p0_first() {
        let (lib0, lib1) = two_card_libraries();
        let state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 42);

        assert_eq!(state.objects.len(), 7);
        assert_eq!(
            state.players[0].library,
            vec![ObjectId(0), ObjectId(1), ObjectId(2)]
        );
        assert_eq!(
            state.players[1].library,
            vec![ObjectId(3), ObjectId(4), ObjectId(5), ObjectId(6)]
        );
        assert_eq!(state.objects.get(ObjectId(0)).card_def, 1);
        assert_eq!(state.objects.get(ObjectId(3)).card_def, 4);
    }

    #[test]
    fn draw_removes_top_of_library_and_appends_to_hand() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 1);

        let drawn = state.draw_card(PlayerId::P0).unwrap();
        assert_eq!(drawn, ObjectId(0)); // was index 0 = top
        assert_eq!(state.players[0].library, vec![ObjectId(1), ObjectId(2)]);
        assert_eq!(state.players[0].hand, vec![ObjectId(0)]);
        assert_eq!(state.objects.get(ObjectId(0)).zone, Zone::Hand);

        let drawn2 = state.draw_card(PlayerId::P0).unwrap();
        assert_eq!(drawn2, ObjectId(1));
        assert_eq!(state.players[0].hand, vec![ObjectId(0), ObjectId(1)]);
    }

    #[test]
    fn library_knowledge_is_perspective_scoped_and_draw_shifts_positions() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 1);

        state.reveal_library_top(PlayerId::P0, PlayerId::P0, 3);
        assert_eq!(
            state
                .known_library_cards(PlayerId::P0, PlayerId::P0)
                .iter()
                .map(|entry| (entry.position, entry.object))
                .collect::<Vec<_>>(),
            vec![(0, ObjectId(0)), (1, ObjectId(1)), (2, ObjectId(2))]
        );
        assert!(state
            .known_library_cards(PlayerId::P1, PlayerId::P0)
            .is_empty());

        assert_eq!(state.draw_card(PlayerId::P0), Some(ObjectId(0)));
        assert_eq!(
            state
                .known_library_cards(PlayerId::P0, PlayerId::P0)
                .iter()
                .map(|entry| (entry.position, entry.object))
                .collect::<Vec<_>>(),
            vec![(0, ObjectId(1)), (1, ObjectId(2))]
        );
    }

    #[test]
    fn reorder_reveals_only_to_named_observers_and_shuffle_clears_everyone() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 7);
        state.reveal_library_top(PlayerId::P0, PlayerId::P0, 3);
        state.reveal_library_top(PlayerId::P1, PlayerId::P0, 3);

        state
            .reorder_library_top(
                PlayerId::P0,
                &[ObjectId(2), ObjectId(0), ObjectId(1)],
                &[PlayerId::P0],
            )
            .unwrap();
        assert_eq!(
            state.players[0].library,
            vec![ObjectId(2), ObjectId(0), ObjectId(1)]
        );
        assert_eq!(
            state
                .known_library_cards(PlayerId::P0, PlayerId::P0)
                .iter()
                .map(|entry| entry.object)
                .collect::<Vec<_>>(),
            vec![ObjectId(2), ObjectId(0), ObjectId(1)]
        );
        assert!(state
            .known_library_cards(PlayerId::P1, PlayerId::P0)
            .is_empty());

        state.shuffle_library(PlayerId::P0);
        assert!(state
            .known_library_cards(PlayerId::P0, PlayerId::P0)
            .is_empty());
        assert!(state
            .known_library_cards(PlayerId::P1, PlayerId::P0)
            .is_empty());
    }

    #[test]
    fn invalid_library_reorder_is_rejected_without_mutation() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 1);
        let before = state.clone();
        assert!(state
            .reorder_library_top(PlayerId::P0, &[ObjectId(0), ObjectId(0)], &[PlayerId::P0])
            .is_err());
        assert_eq!(state, before);
    }

    #[test]
    fn draw_from_empty_library_is_none_and_noop() {
        let mut state = GameState::new_from_libraries(&[], &[1], debug_names, 1);
        assert_eq!(state.draw_card(PlayerId::P0), None);
        assert!(state.players[0].hand.is_empty());
    }

    #[test]
    fn object_id_stable_across_multiple_zone_moves() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 7);

        let id = state.draw_card(PlayerId::P0).unwrap();
        assert_eq!(id, ObjectId(0));
        assert_eq!(state.objects.get(id).zone, Zone::Hand);

        let moved = state.move_hand_to_battlefield(PlayerId::P0, id);
        assert!(moved);
        assert_eq!(id, ObjectId(0)); // same id throughout: library -> hand -> battlefield
        assert_eq!(state.objects.get(id).zone, Zone::Battlefield);
        assert!(state.players[0].battlefield.contains(&id));
        assert!(!state.players[0].hand.contains(&id));
        assert!(!state.players[0].library.contains(&id));
    }

    #[test]
    fn state_hash_is_deterministic_for_identical_sequences() {
        let (lib0, lib1) = two_card_libraries();
        let mut a = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        let mut b = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);

        a.draw_card(PlayerId::P0);
        a.draw_card(PlayerId::P1);
        b.draw_card(PlayerId::P0);
        b.draw_card(PlayerId::P1);

        assert_eq!(a, b);
        assert_eq!(a.state_hash(), b.state_hash());
        assert_eq!(a.diagnostic_state_hash(), b.diagnostic_state_hash());
    }

    #[test]
    fn diagnostic_state_hash_contract_and_golden_value_are_frozen() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        state.draw_card(PlayerId::P0);
        state.draw_card(PlayerId::P1);

        assert_eq!(
            DIAGNOSTIC_STATE_HASH_ALGORITHM,
            "fnv1a64-serde-json-game-state-envelope-v1"
        );
        assert_eq!(DIAGNOSTIC_STATE_HASH_ENVELOPE_SCHEMA_VERSION, 1);
        assert!(
            diagnostic_state_hash_bytes(&state).starts_with(b"{\"schema_version\":1,\"state\":{")
        );
        assert_eq!(state.diagnostic_state_hash(), 0xbd23_32bb_d985_bae9);
    }

    /// Draws to different players don't interact, so interleaving order
    /// across players is unobservable in the resulting state. This is the
    /// flip side of the "no unordered-map iteration" invariant: the only
    /// state each draw touches is `players[p].{library,hand}` and the
    /// touched object, so two draws to distinct players commute.
    #[test]
    fn state_hash_is_order_independent_across_distinct_players() {
        let (lib0, lib1) = two_card_libraries();
        let mut a = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        let mut b = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);

        a.draw_card(PlayerId::P0);
        a.draw_card(PlayerId::P1);
        b.draw_card(PlayerId::P1);
        b.draw_card(PlayerId::P0);

        assert_eq!(a, b);
        assert_eq!(a.state_hash(), b.state_hash());
    }

    #[test]
    fn state_hash_detects_a_genuine_state_difference() {
        let (lib0, lib1) = two_card_libraries();
        let mut a = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        let mut b = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);

        a.draw_card(PlayerId::P0);
        b.draw_card(PlayerId::P0);
        b.draw_card(PlayerId::P0); // b has one extra card drawn

        assert_ne!(a, b);
        assert_ne!(a.state_hash(), b.state_hash());
        assert_ne!(a.diagnostic_state_hash(), b.diagnostic_state_hash());
    }

    #[test]
    fn diagnostic_state_hash_includes_rng_and_pending_cost_override() {
        let (lib0, lib1) = two_card_libraries();
        let mut state = GameState::new_from_libraries(&lib0, &lib1, debug_names, 99);
        let initial = state.diagnostic_state_hash();
        state.rng.next_u64();
        assert_ne!(
            state.diagnostic_state_hash(),
            initial,
            "RNG state is privileged full state"
        );

        let spell = state.players[0].library[0];
        state.engine.pending_cast = Some(crate::engine::PendingCast {
            spell,
            controller: PlayerId::P0,
            target_spec: crate::card_def::TargetSpec::None,
            targets_chosen: Vec::new(),
            is_flashback: false,
            cast_mode: Some(crate::engine::CastMode::Normal),
            additional_cost_discarded: Some(Vec::new()),
            cost_override: None,
            mode_chosen: Some(0),
            origin_zone: Zone::Hand,
            sacrifice_chosen: Vec::new(),
            kicked: Some(false),
        });
        let ordinary_cost = state.diagnostic_state_hash();
        state.engine.pending_cast.as_mut().unwrap().cost_override = Some(crate::mana::Cost::zero());
        assert_ne!(
            state.diagnostic_state_hash(),
            ordinary_cost,
            "the diagnostic full-state envelope must not skip cost_override"
        );
    }

    #[test]
    fn rng_stream_is_deterministic_per_seed() {
        let mut r1 = SplitMix64::seed(12345);
        let mut r2 = SplitMix64::seed(12345);
        for _ in 0..10 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }

        let mut r3 = SplitMix64::seed(6789);
        assert_ne!(r1.next_u64(), r3.next_u64());
    }
}
