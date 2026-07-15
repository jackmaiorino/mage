//! Core game state. Every collection here is a `Vec` (or a fixed array) with
//! caller-controlled order; nothing is ever iterated via a `HashMap`, so two
//! states built from the same inputs always compare and hash identically
//! (see `state_hash` and the determinism test below).

use crate::ids::{Arena, ObjectId, PlayerId};
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

pub const STARTING_LIFE: i32 = 20;

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

/// +1/+1 counters only for now; add fields here as the card pool needs them
/// (e.g. `minus1_minus1`, `charge`, `loyalty`) rather than a generic map, so
/// this stays hashable and iteration-order-free.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Counters {
    pub plus1_plus1: i8,
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
            plotted_turn: None,
            zone_change_count: 0,
        }
    }
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

/// Minimal stack entry: enough to represent "something is on the stack with
/// these targets." Resolution/effect semantics belong to the step layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackItem {
    pub kind: StackItemKind,
    pub source: ObjectId,
    pub controller: PlayerId,
    pub targets: Vec<Target>,
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
    pub rng: SplitMix64,
    /// Priority/stack/turn-structure bookkeeping and the propose-commit
    /// event log, all owned by the `engine`/`event`/`trigger` modules. See
    /// `engine::EngineState`.
    pub engine: crate::engine::EngineState,
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
            rng: SplitMix64::seed(seed),
            engine: crate::engine::EngineState::default(),
        }
    }

    /// Removes the top card of `player`'s library and puts it in hand.
    /// Returns `None` (no state change) if the library is empty.
    pub fn draw_card(&mut self, player: PlayerId) -> Option<ObjectId> {
        let ps = &mut self.players[player.index()];
        if ps.library.is_empty() {
            return None;
        }
        let id = ps.library.remove(0);
        ps.hand.push(id);
        self.objects.get_mut(id).zone = Zone::Hand;
        Some(id)
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
        let obj = self.objects.get_mut(id);
        obj.zone = Zone::Battlefield;
        obj.summoning_sick = true;
        obj.zone_change_count += 1;
        true
    }

    /// FNV-1a over the derived `Hash` impl, which visits every field in
    /// declaration order and every `Vec`/array element in stored order.
    /// There are no floats and no `HashMap`s in the state model, so this is
    /// bit-for-bit reproducible for identical states.
    pub fn state_hash(&self) -> u64 {
        let mut hasher = Fnv1a64::new();
        self.hash(&mut hasher);
        hasher.finish()
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
        for &b in bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }
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
