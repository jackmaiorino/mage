//! Stable RL-facing contracts for the kernel-owned trainer/runner boundary.
//!
//! This module is intentionally data-shaped. It exposes a perspective-safe
//! observation projection, structured legal-action ids, versioned JSONL episode
//! records, and a deterministic Burn-mirror rollout helper. It does not make
//! any learning or strength claim.

use crate::card_def::{card_id_by_name, CARD_DEFS, KERNEL_CARDDB_HASH};
use crate::engine::{Action, CastMode, CostKind, Decision, OptionalCostChoice};
use crate::event::{self, ProposedEvent};
use crate::ids::{ObjectId, PlayerId};
use crate::state::{GameObject, GameState, SplitMix64, StackItem, Target, Zone};
use crate::surface_v2::{HarnessSurfaceV2, SurfaceAction, SurfaceDecision, H2_PREDICATE_VERSION};
use crate::KERNEL_VERSION;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub const OBSERVATION_SCHEMA_VERSION: u32 = 1;
pub const LEGAL_ACTION_SCHEMA_VERSION: u32 = 1;
pub const EPISODE_SCHEMA_VERSION: u32 = 1;
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_MAX_DECISIONS: u64 = 200_000;
pub const BURN_MIRROR_MATCHUP: &str = "burn_mirror";
pub const EPISODE_JSONL_FILENAME: &str = "episodes.jsonl";
pub const MANIFEST_FILENAME: &str = "manifest.json";

const MAX_SUBSET_OBJECTS: usize = 12;
const MAX_TRIGGER_ORDER_OBJECTS: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RlContractError(pub String);

impl fmt::Display for RlContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RlContractError {}

impl From<std::io::Error> for RlContractError {
    fn from(value: std::io::Error) -> Self {
        RlContractError(value.to_string())
    }
}

impl From<serde_json::Error> for RlContractError {
    fn from(value: serde_json::Error) -> Self {
        RlContractError(value.to_string())
    }
}

type Result<T> = std::result::Result<T, RlContractError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlayerSeatV1 {
    P0,
    P1,
}

impl From<PlayerId> for PlayerSeatV1 {
    fn from(value: PlayerId) -> Self {
        match value {
            PlayerId::P0 => PlayerSeatV1::P0,
            PlayerId::P1 => PlayerSeatV1::P1,
            _ => panic!("unsupported player id {}", value.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CountersV1 {
    pub plus1_plus1: i8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CardStableRefV1 {
    pub arena_id: u32,
    pub card_db_id: u16,
    pub owner: PlayerSeatV1,
    pub controller: PlayerSeatV1,
    pub zone: Zone,
    pub zone_change_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CardPublicV1 {
    pub stable: CardStableRefV1,
    pub card_name: String,
    pub tapped: bool,
    pub summoning_sick: bool,
    pub damage: u16,
    pub counters: CountersV1,
    pub attachments: Vec<u32>,
    pub plotted_turn: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CardPrivateV1 {
    pub stable: CardStableRefV1,
    pub card_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "target_kind", rename_all = "snake_case")]
pub enum TargetRefV1 {
    Player { player: PlayerSeatV1 },
    Object { object: CardStableRefV1 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StackItemPublicV1 {
    pub stack_index: u32,
    pub source: CardStableRefV1,
    pub controller: PlayerSeatV1,
    pub targets: Vec<TargetRefV1>,
    pub is_trigger_or_ability: bool,
    pub is_flashback: bool,
    pub mode_chosen: u8,
    pub madness_offer: bool,
    pub kicked: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlayerStatusV1 {
    pub has_lost: bool,
    pub lands_played_this_turn: u8,
    pub drew_from_empty: bool,
    pub draws_this_turn: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PublicObservationProjectionV1 {
    pub turn: u32,
    pub phase: ZoneIndependentStepV1,
    pub active_player: PlayerSeatV1,
    pub priority_player: PlayerSeatV1,
    pub life_totals: [i32; 2],
    pub mana_pools: [[u8; 6]; 2],
    pub hand_counts: [usize; 2],
    pub library_counts: [usize; 2],
    pub player_status: [PlayerStatusV1; 2],
    pub battlefield: [Vec<CardPublicV1>; 2],
    pub graveyards: [Vec<CardPublicV1>; 2],
    pub exile: Vec<CardPublicV1>,
    pub stack: Vec<StackItemPublicV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZoneIndependentStepV1 {
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

impl From<crate::state::Step> for ZoneIndependentStepV1 {
    fn from(value: crate::state::Step) -> Self {
        match value {
            crate::state::Step::Untap => ZoneIndependentStepV1::Untap,
            crate::state::Step::Upkeep => ZoneIndependentStepV1::Upkeep,
            crate::state::Step::Draw => ZoneIndependentStepV1::Draw,
            crate::state::Step::Main1 => ZoneIndependentStepV1::Main1,
            crate::state::Step::BeginCombat => ZoneIndependentStepV1::BeginCombat,
            crate::state::Step::DeclareAttackers => ZoneIndependentStepV1::DeclareAttackers,
            crate::state::Step::DeclareBlockers => ZoneIndependentStepV1::DeclareBlockers,
            crate::state::Step::CombatDamage => ZoneIndependentStepV1::CombatDamage,
            crate::state::Step::EndCombat => ZoneIndependentStepV1::EndCombat,
            crate::state::Step::Main2 => ZoneIndependentStepV1::Main2,
            crate::state::Step::End => ZoneIndependentStepV1::End,
            crate::state::Step::Cleanup => ZoneIndependentStepV1::Cleanup,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationV1 {
    pub schema_version: u32,
    pub kernel_version: String,
    pub surface_version: u32,
    pub card_db_hash: u64,
    pub acting_player: PlayerSeatV1,
    pub step_index: u64,
    pub projection: PublicObservationProjectionV1,
    pub own_hand: Vec<CardPrivateV1>,
    pub diagnostic_state_hash_includes_hidden_state: bool,
    pub diagnostic_state_hash: u64,
    pub visible_projection_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "action_kind", rename_all = "snake_case")]
pub enum ActionSemanticV1 {
    Pass {
        actor: PlayerSeatV1,
    },
    PlayLand {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
    },
    CastSpell {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
    },
    ActivateManaAbility {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
    },
    ActivateAbility {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
        ability_index: u8,
    },
    PlotSpell {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
    },
    ChooseTarget {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
        remaining: u8,
        target: TargetRefV1,
    },
    ChooseCostTarget {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
        cost_kind: CostKind,
        remaining: u8,
        candidate: CardStableRefV1,
    },
    ChooseCastMode {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
        mode: CastMode,
    },
    ChooseKicker {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
        pay: bool,
    },
    ChooseSpellMode {
        actor: PlayerSeatV1,
        source: CardStableRefV1,
        mode_index: u8,
        mode_count: u8,
    },
    ChooseOptionalCostUse {
        actor: PlayerSeatV1,
        use_cost: bool,
    },
    ChooseOptionalCostWhich {
        actor: PlayerSeatV1,
        choice: OptionalCostChoice,
    },
    ChooseMadnessCast {
        actor: PlayerSeatV1,
        card: CardStableRefV1,
        cast_it: bool,
    },
    Discard {
        actor: PlayerSeatV1,
        cards: Vec<CardStableRefV1>,
    },
    DeclareAttackers {
        actor: PlayerSeatV1,
        attackers: Vec<CardStableRefV1>,
    },
    DeclareBlockersForAttacker {
        actor: PlayerSeatV1,
        attacker: CardStableRefV1,
        blockers: Vec<CardStableRefV1>,
    },
    OrderTriggers {
        actor: PlayerSeatV1,
        pending_sources: Vec<CardStableRefV1>,
        order: Vec<usize>,
    },
    Ambiguous {
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegalActionV1 {
    pub schema_version: u32,
    pub selected_index: u32,
    pub stable_id: String,
    pub semantic: ActionSemanticV1,
    pub display_text: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LegalActionCandidateV1 {
    pub record: LegalActionV1,
    pub surface_action: SurfaceAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LibrarySetupV1 {
    pub setup_name: String,
    pub shuffle_algorithm: String,
    pub opening_hand_policy: String,
    pub env_seed: u64,
    pub deck_hashes: [u64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcomeV1 {
    Win,
    Loss,
    Draw,
    Halted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "record_type", rename_all = "snake_case")]
pub enum EpisodeRecordV1 {
    Header {
        schema_version: u32,
        kernel_version: String,
        surface_version: u32,
        card_db_hash: u64,
        matchup: String,
        episode_id: u64,
        game_id: String,
        env_seed: u64,
        policy_seed: u64,
        deck_identifiers: [String; 2],
        library_setup: LibrarySetupV1,
    },
    Decision {
        schema_version: u32,
        episode_id: u64,
        step: u64,
        acting_player: PlayerSeatV1,
        observation: ObservationV1,
        observation_projection_hash: u64,
        diagnostic_state_hash: u64,
        legal_actions: Vec<LegalActionV1>,
        selected_index: u32,
        selected_action_id: String,
        reward: [i32; 2],
    },
    Terminal {
        schema_version: u32,
        episode_id: u64,
        terminal_outcome: TerminalOutcomeV1,
        winner: Option<PlayerSeatV1>,
        terminal_reward: [i32; 2],
        terminal_reason: String,
        decision_count: u64,
        diagnostic_state_hash: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeTerminalSummaryV1 {
    pub episode_id: u64,
    pub outcome: TerminalOutcomeV1,
    pub winner: Option<PlayerSeatV1>,
    pub terminal_reward: [i32; 2],
    pub terminal_reason: String,
    pub decision_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeRunV1 {
    pub records: Vec<EpisodeRecordV1>,
    pub terminal: EpisodeTerminalSummaryV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedManifestV1 {
    pub base_seed: u64,
    pub derivation: String,
    pub episode_ids: Vec<u64>,
    pub env_seeds: Vec<u64>,
    pub policy_seeds: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputFilesV1 {
    pub episode_jsonl: String,
    pub manifest_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeckManifestV1 {
    pub deck_identifiers: [String; 2],
    pub deck_hashes: [u64; 2],
    pub card_count: usize,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GitDirtyFlagV1 {
    Clean,
    Dirty,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitMetadataV1 {
    pub commit: String,
    pub dirty: GitDirtyFlagV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunAggregateV1 {
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub halted: u64,
    pub total_decisions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariableMetadataV1 {
    pub out_dir: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunManifestV1 {
    pub schema_version: u32,
    pub kernel_version: String,
    pub surface_version: u32,
    pub card_db_hash: u64,
    pub matchup: String,
    pub game_count: u64,
    pub max_decisions: u64,
    pub cli_args: Vec<String>,
    pub seed: SeedManifestV1,
    pub output_files: OutputFilesV1,
    pub deck: DeckManifestV1,
    pub git: GitMetadataV1,
    pub aggregate: RunAggregateV1,
    pub variable_metadata: VariableMetadataV1,
}

pub const BURN_MAINBOARD: &[(&str, u32)] = &[
    ("Sneaky Snacker", 4),
    ("Faithless Looting", 2),
    ("Highway Robbery", 4),
    ("Masked Meower", 4),
    ("Lightning Bolt", 4),
    ("Mountain", 18),
    ("Grab the Prize", 4),
    ("Fireblast", 4),
    ("Guttersnipe", 4),
    ("Fiery Temper", 4),
    ("Voldaren Epicure", 4),
    ("Lava Dart", 4),
];

pub fn burn_deck_ids() -> Vec<u16> {
    let mut ids = Vec::with_capacity(60);
    for &(name, qty) in BURN_MAINBOARD {
        let id = card_id_by_name(name).unwrap_or_else(|| panic!("{name} missing from CARD_DEFS"));
        for _ in 0..qty {
            ids.push(id);
        }
    }
    assert_eq!(
        ids.len(),
        60,
        "Mono-Red Burn mainboard should be exactly 60 cards"
    );
    ids
}

pub fn shuffled(ids: &[u16], rng: &mut SplitMix64) -> Vec<u16> {
    let mut v = ids.to_vec();
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

pub fn build_burn_mirror_state(seed: u64) -> GameState {
    let deck = burn_deck_ids();
    let mut shuffle_rng = SplitMix64::seed(seed);
    let lib0 = shuffled(&deck, &mut shuffle_rng);
    let lib1 = shuffled(&deck, &mut shuffle_rng);
    let mut state = GameState::new_from_libraries(&lib0, &lib1, card_name, seed);
    for _ in 0..7 {
        event::propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P0));
        event::propose_and_commit(&mut state, ProposedEvent::draw(PlayerId::P1));
    }
    state
}

pub fn derive_env_seed(base_seed: u64, episode_id: u64) -> u64 {
    derive_seed(base_seed, episode_id, 0x4556_5f52_4c5f_7631)
}

pub fn derive_policy_seed(base_seed: u64, episode_id: u64) -> u64 {
    derive_seed(base_seed, episode_id, 0x504f_4c49_4359_7631)
}

fn derive_seed(base_seed: u64, episode_id: u64, stream: u64) -> u64 {
    let mut rng =
        SplitMix64::seed(base_seed ^ stream ^ episode_id.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    rng.next_u64()
}

pub fn card_name(card_id: u16) -> String {
    CARD_DEFS[card_id as usize].name.to_string()
}

pub fn observe_v1(
    state: &GameState,
    acting_player: PlayerId,
    step_index: u64,
) -> Result<ObservationV1> {
    let projection = PublicObservationProjectionV1 {
        turn: state.turn,
        phase: state.step.into(),
        active_player: state.active_player.into(),
        priority_player: state.priority_player.into(),
        life_totals: [state.players[0].life, state.players[1].life],
        mana_pools: [state.players[0].mana_pool, state.players[1].mana_pool],
        hand_counts: [state.players[0].hand.len(), state.players[1].hand.len()],
        library_counts: [
            state.players[0].library.len(),
            state.players[1].library.len(),
        ],
        player_status: [
            player_status_v1(&state.players[0]),
            player_status_v1(&state.players[1]),
        ],
        battlefield: [
            public_cards(state, &state.players[0].battlefield)?,
            public_cards(state, &state.players[1].battlefield)?,
        ],
        graveyards: [
            public_cards(state, &state.players[0].graveyard)?,
            public_cards(state, &state.players[1].graveyard)?,
        ],
        exile: public_cards(state, &state.exile)?,
        stack: stack_public(state)?,
    };
    let own_hand = state.players[acting_player.index()]
        .hand
        .iter()
        .map(|&id| private_card(state, id))
        .collect::<Result<Vec<_>>>()?;
    let mut obs = ObservationV1 {
        schema_version: OBSERVATION_SCHEMA_VERSION,
        kernel_version: KERNEL_VERSION.to_string(),
        surface_version: H2_PREDICATE_VERSION,
        card_db_hash: KERNEL_CARDDB_HASH,
        acting_player: acting_player.into(),
        step_index,
        projection,
        own_hand,
        diagnostic_state_hash_includes_hidden_state: true,
        diagnostic_state_hash: state.state_hash(),
        visible_projection_hash: 0,
    };
    obs.visible_projection_hash = visible_projection_hash(&obs)?;
    Ok(obs)
}

pub fn make_legal_action_v1(
    selected_index: u32,
    semantic: ActionSemanticV1,
    display_text: Option<String>,
) -> Result<LegalActionV1> {
    if let ActionSemanticV1::Ambiguous { reason } = &semantic {
        return Err(RlContractError(format!(
            "ambiguous legal action representation refused: {reason}"
        )));
    }
    let hash = stable_hash_json(&semantic)?;
    Ok(LegalActionV1 {
        schema_version: LEGAL_ACTION_SCHEMA_VERSION,
        selected_index,
        stable_id: format!("legal-action-v1:{hash:016x}"),
        semantic,
        display_text,
    })
}

pub fn legal_action_candidates_v1(
    decision: &SurfaceDecision,
    state: &GameState,
) -> Result<Vec<LegalActionCandidateV1>> {
    let mut out = Vec::new();
    match decision {
        SurfaceDecision::Decision(decision) => match decision {
            Decision::CastSpellOrPass {
                player,
                castable_spells,
                mana_abilities,
                land_drops,
                activatable_abilities,
                plot_actions,
            } => {
                let actor = (*player).into();
                for &id in castable_spells {
                    push_action(
                        &mut out,
                        ActionSemanticV1::CastSpell {
                            actor,
                            source: card_ref(state, id)?,
                        },
                        SurfaceAction::Action(Action::CastSpell(id)),
                    )?;
                }
                for &id in mana_abilities {
                    push_action(
                        &mut out,
                        ActionSemanticV1::ActivateManaAbility {
                            actor,
                            source: card_ref(state, id)?,
                        },
                        SurfaceAction::Action(Action::ActivateManaAbility(id)),
                    )?;
                }
                for &id in land_drops {
                    push_action(
                        &mut out,
                        ActionSemanticV1::PlayLand {
                            actor,
                            source: card_ref(state, id)?,
                        },
                        SurfaceAction::Action(Action::PlayLand(id)),
                    )?;
                }
                for &(id, ability_index) in activatable_abilities {
                    push_action(
                        &mut out,
                        ActionSemanticV1::ActivateAbility {
                            actor,
                            source: card_ref(state, id)?,
                            ability_index,
                        },
                        SurfaceAction::Action(Action::ActivateAbility(id, ability_index)),
                    )?;
                }
                for &id in plot_actions {
                    push_action(
                        &mut out,
                        ActionSemanticV1::PlotSpell {
                            actor,
                            source: card_ref(state, id)?,
                        },
                        SurfaceAction::Action(Action::PlotSpell(id)),
                    )?;
                }
                push_action(
                    &mut out,
                    ActionSemanticV1::Pass { actor },
                    SurfaceAction::Action(Action::Pass),
                )?;
            }
            Decision::ChooseTargets {
                player,
                spell,
                remaining,
                legal_targets,
            } => {
                let actor = (*player).into();
                let source = card_ref(state, *spell)?;
                for &target in legal_targets {
                    push_action(
                        &mut out,
                        ActionSemanticV1::ChooseTarget {
                            actor,
                            source: source.clone(),
                            remaining: *remaining,
                            target: target_ref(state, target)?,
                        },
                        SurfaceAction::Action(Action::ChooseTarget(target)),
                    )?;
                }
            }
            Decision::ChooseCostTargets {
                player,
                source,
                cost_kind,
                remaining,
                candidates,
            } => {
                let actor = (*player).into();
                let source_ref = card_ref(state, *source)?;
                for &candidate in candidates {
                    push_action(
                        &mut out,
                        ActionSemanticV1::ChooseCostTarget {
                            actor,
                            source: source_ref.clone(),
                            cost_kind: *cost_kind,
                            remaining: *remaining,
                            candidate: card_ref(state, candidate)?,
                        },
                        SurfaceAction::Action(Action::ChooseCostTarget(candidate)),
                    )?;
                }
            }
            Decision::ChooseCastMode {
                player,
                spell,
                options,
            } => {
                let actor = (*player).into();
                let source = card_ref(state, *spell)?;
                for &mode in options {
                    push_action(
                        &mut out,
                        ActionSemanticV1::ChooseCastMode {
                            actor,
                            source: source.clone(),
                            mode,
                        },
                        SurfaceAction::Action(Action::ChooseCastMode(mode)),
                    )?;
                }
            }
            Decision::ChooseKicker { player, spell } => {
                let actor = (*player).into();
                let source = card_ref(state, *spell)?;
                for pay in [false, true] {
                    push_action(
                        &mut out,
                        ActionSemanticV1::ChooseKicker {
                            actor,
                            source: source.clone(),
                            pay,
                        },
                        SurfaceAction::Action(Action::ChooseKicker(pay)),
                    )?;
                }
            }
            Decision::ChooseSpellMode {
                player,
                spell,
                mode_count,
            } => {
                let actor = (*player).into();
                let source = card_ref(state, *spell)?;
                for mode_index in 0..*mode_count {
                    push_action(
                        &mut out,
                        ActionSemanticV1::ChooseSpellMode {
                            actor,
                            source: source.clone(),
                            mode_index,
                            mode_count: *mode_count,
                        },
                        SurfaceAction::Action(Action::ChooseSpellMode(mode_index)),
                    )?;
                }
            }
            Decision::ChooseOptionalCost {
                player,
                discard_payable,
                sacrifice_payable,
            } => {
                let actor = (*player).into();
                match (*discard_payable, *sacrifice_payable) {
                    (false, false) => {
                        for use_cost in [false, true] {
                            push_action(
                                &mut out,
                                ActionSemanticV1::ChooseOptionalCostUse { actor, use_cost },
                                SurfaceAction::Action(Action::ChooseOptionalCostStage(use_cost)),
                            )?;
                        }
                    }
                    (true, true) => {
                        for (choice, use_it) in [
                            (OptionalCostChoice::Discard, true),
                            (OptionalCostChoice::SacrificeLand, false),
                        ] {
                            push_action(
                                &mut out,
                                ActionSemanticV1::ChooseOptionalCostWhich { actor, choice },
                                SurfaceAction::Action(Action::ChooseOptionalCostStage(use_it)),
                            )?;
                        }
                    }
                    other => {
                        return Err(RlContractError(format!(
                            "unsupported surfaced ChooseOptionalCost flags {other:?}; expected H2 use-gate or which-gate sentinel"
                        )));
                    }
                }
            }
            Decision::ChooseMadnessCast { player, card } => {
                let actor = (*player).into();
                let card = card_ref(state, *card)?;
                for cast_it in [false, true] {
                    push_action(
                        &mut out,
                        ActionSemanticV1::ChooseMadnessCast {
                            actor,
                            card: card.clone(),
                            cast_it,
                        },
                        SurfaceAction::Action(Action::ChooseMadnessCast(cast_it)),
                    )?;
                }
            }
            Decision::Discard {
                player,
                count,
                choices,
            } => {
                if *count != 1 {
                    return Err(RlContractError(format!(
                        "surface discard contract expected count=1 after H2 reshape, got count={count}"
                    )));
                }
                let actor = (*player).into();
                for &id in choices {
                    push_action(
                        &mut out,
                        ActionSemanticV1::Discard {
                            actor,
                            cards: vec![card_ref(state, id)?],
                        },
                        SurfaceAction::Action(Action::Discard(vec![id])),
                    )?;
                }
            }
            Decision::DeclareAttackers { player, eligible } => {
                let actor = (*player).into();
                for attackers in subsets(eligible)? {
                    let attacker_refs = attackers
                        .iter()
                        .map(|&id| card_ref(state, id))
                        .collect::<Result<Vec<_>>>()?;
                    push_action(
                        &mut out,
                        ActionSemanticV1::DeclareAttackers {
                            actor,
                            attackers: attacker_refs,
                        },
                        SurfaceAction::Action(Action::DeclareAttackers(attackers)),
                    )?;
                }
            }
            Decision::DeclareBlockers { .. } => {
                return Err(RlContractError(
                    "raw DeclareBlockers is not a HarnessSurfaceV2 decision; expected DeclareBlockersForAttacker reshape".to_string(),
                ));
            }
            Decision::OrderTriggers { player, pending } => {
                let actor = (*player).into();
                let pending_sources = pending
                    .iter()
                    .map(|p| card_ref(state, p.source))
                    .collect::<Result<Vec<_>>>()?;
                for order in permutations(pending.len())? {
                    push_action(
                        &mut out,
                        ActionSemanticV1::OrderTriggers {
                            actor,
                            pending_sources: pending_sources.clone(),
                            order: order.clone(),
                        },
                        SurfaceAction::Action(Action::OrderTriggers(order)),
                    )?;
                }
            }
            Decision::GameOver { .. } | Decision::Halted { .. } => {}
        },
        SurfaceDecision::DeclareBlockersForAttacker {
            attacker,
            legal_blockers,
        } => {
            let actor = state.objects.get(*attacker).controller.opponent().into();
            let attacker_ref = card_ref(state, *attacker)?;
            for blockers in subsets(legal_blockers)? {
                let blocker_refs = blockers
                    .iter()
                    .map(|&id| card_ref(state, id))
                    .collect::<Result<Vec<_>>>()?;
                push_action(
                    &mut out,
                    ActionSemanticV1::DeclareBlockersForAttacker {
                        actor,
                        attacker: attacker_ref.clone(),
                        blockers: blocker_refs,
                    },
                    SurfaceAction::DeclareBlockersForAttacker(blockers),
                )?;
            }
        }
    }
    ensure_unique_action_ids(&out)?;
    Ok(out)
}

pub fn acting_player_for_surface_decision(
    decision: &SurfaceDecision,
    state: &GameState,
) -> Option<PlayerId> {
    match decision {
        SurfaceDecision::Decision(decision) => match decision {
            Decision::CastSpellOrPass { player, .. }
            | Decision::ChooseTargets { player, .. }
            | Decision::ChooseCostTargets { player, .. }
            | Decision::ChooseCastMode { player, .. }
            | Decision::ChooseKicker { player, .. }
            | Decision::ChooseSpellMode { player, .. }
            | Decision::ChooseOptionalCost { player, .. }
            | Decision::ChooseMadnessCast { player, .. }
            | Decision::Discard { player, .. }
            | Decision::DeclareAttackers { player, .. }
            | Decision::DeclareBlockers { player, .. }
            | Decision::OrderTriggers { player, .. } => Some(*player),
            Decision::GameOver { .. } | Decision::Halted { .. } => None,
        },
        SurfaceDecision::DeclareBlockersForAttacker { attacker, .. } => {
            Some(state.objects.get(*attacker).controller.opponent())
        }
    }
}

pub fn record_burn_mirror_episode(
    episode_id: u64,
    env_seed: u64,
    policy_seed: u64,
    max_decisions: u64,
) -> Result<EpisodeRunV1> {
    let mut state = build_burn_mirror_state(env_seed);
    let mut surface = HarnessSurfaceV2::new();
    let mut policy_rng = SplitMix64::seed(policy_seed);
    let deck_hash = burn_deck_hash();
    let mut records = vec![EpisodeRecordV1::Header {
        schema_version: EPISODE_SCHEMA_VERSION,
        kernel_version: KERNEL_VERSION.to_string(),
        surface_version: H2_PREDICATE_VERSION,
        card_db_hash: KERNEL_CARDDB_HASH,
        matchup: BURN_MIRROR_MATCHUP.to_string(),
        episode_id,
        game_id: format!(
            "burn_mirror_env_{env_seed:016x}_policy_{policy_seed:016x}_game_{episode_id:06}"
        ),
        env_seed,
        policy_seed,
        deck_identifiers: deck_identifiers(),
        library_setup: LibrarySetupV1 {
            setup_name: "burn_mirror_splitmix64_v1".to_string(),
            shuffle_algorithm: "splitmix64_fisher_yates_sequential_p0_then_p1".to_string(),
            opening_hand_policy: "seven alternating event draws starting with P0".to_string(),
            env_seed,
            deck_hashes: [deck_hash, deck_hash],
        },
    }];
    let mut decision_count = 0u64;
    loop {
        let surfaced = surface.next_decision(&mut state);
        match &surfaced {
            SurfaceDecision::Decision(Decision::GameOver { winner }) => {
                let summary =
                    terminal_summary(episode_id, *winner, "game_over".to_string(), decision_count);
                push_terminal(&mut records, &summary, state.state_hash());
                return Ok(EpisodeRunV1 {
                    records,
                    terminal: summary,
                });
            }
            SurfaceDecision::Decision(Decision::Halted { mechanic, source }) => {
                let reason = format!("engine_halted:{mechanic:?}:source:{}", source.0);
                let summary = halted_summary(episode_id, reason, decision_count);
                push_terminal(&mut records, &summary, state.state_hash());
                return Ok(EpisodeRunV1 {
                    records,
                    terminal: summary,
                });
            }
            _ => {}
        }
        if decision_count >= max_decisions {
            let summary = halted_summary(
                episode_id,
                format!("decision_cap_reached:{max_decisions}"),
                decision_count,
            );
            push_terminal(&mut records, &summary, state.state_hash());
            return Ok(EpisodeRunV1 {
                records,
                terminal: summary,
            });
        }
        let Some(actor) = acting_player_for_surface_decision(&surfaced, &state) else {
            let summary = halted_summary(
                episode_id,
                "fail_closed:nonterminal decision without acting player".to_string(),
                decision_count,
            );
            push_terminal(&mut records, &summary, state.state_hash());
            return Ok(EpisodeRunV1 {
                records,
                terminal: summary,
            });
        };
        let observation = observe_v1(&state, actor, decision_count)?;
        let candidates = match legal_action_candidates_v1(&surfaced, &state) {
            Ok(candidates) => candidates,
            Err(err) => {
                let summary =
                    halted_summary(episode_id, format!("fail_closed:{err}"), decision_count);
                push_terminal(&mut records, &summary, state.state_hash());
                return Ok(EpisodeRunV1 {
                    records,
                    terminal: summary,
                });
            }
        };
        if candidates.is_empty() {
            let summary = halted_summary(
                episode_id,
                "fail_closed:nonterminal decision produced zero legal actions".to_string(),
                decision_count,
            );
            push_terminal(&mut records, &summary, state.state_hash());
            return Ok(EpisodeRunV1 {
                records,
                terminal: summary,
            });
        }
        let selected_index = rng_below(&mut policy_rng, candidates.len());
        let selected = &candidates[selected_index];
        validate_selected_action(&candidates, selected_index, &selected.record.stable_id)?;
        records.push(EpisodeRecordV1::Decision {
            schema_version: EPISODE_SCHEMA_VERSION,
            episode_id,
            step: decision_count,
            acting_player: actor.into(),
            observation_projection_hash: observation.visible_projection_hash,
            diagnostic_state_hash: observation.diagnostic_state_hash,
            observation,
            legal_actions: candidates.iter().map(|c| c.record.clone()).collect(),
            selected_index: selected_index as u32,
            selected_action_id: selected.record.stable_id.clone(),
            reward: [0, 0],
        });
        if let Err(err) = surface.apply(&mut state, selected.surface_action.clone()) {
            let summary = halted_summary(
                episode_id,
                format!("fail_closed:surface_apply:{err}"),
                decision_count + 1,
            );
            push_terminal(&mut records, &summary, state.state_hash());
            return Ok(EpisodeRunV1 {
                records,
                terminal: summary,
            });
        }
        decision_count += 1;
    }
}

pub fn build_rollout_records(
    games: u64,
    base_seed: u64,
    max_decisions: u64,
) -> Result<(Vec<EpisodeRecordV1>, Vec<EpisodeTerminalSummaryV1>)> {
    let mut all_records = Vec::new();
    let mut summaries = Vec::new();
    for episode_id in 0..games {
        let env_seed = derive_env_seed(base_seed, episode_id);
        let policy_seed = derive_policy_seed(base_seed, episode_id);
        let run = record_burn_mirror_episode(episode_id, env_seed, policy_seed, max_decisions)?;
        all_records.extend(run.records);
        summaries.push(run.terminal);
    }
    Ok((all_records, summaries))
}

pub fn build_run_manifest(
    games: u64,
    base_seed: u64,
    max_decisions: u64,
    cli_args: Vec<String>,
    out_dir: &Path,
    summaries: &[EpisodeTerminalSummaryV1],
    git: GitMetadataV1,
) -> RunManifestV1 {
    let deck_hash = burn_deck_hash();
    RunManifestV1 {
        schema_version: MANIFEST_SCHEMA_VERSION,
        kernel_version: KERNEL_VERSION.to_string(),
        surface_version: H2_PREDICATE_VERSION,
        card_db_hash: KERNEL_CARDDB_HASH,
        matchup: BURN_MIRROR_MATCHUP.to_string(),
        game_count: games,
        max_decisions,
        cli_args,
        seed: SeedManifestV1 {
            base_seed,
            derivation: "env_seed=splitmix64(base_seed ^ ENV_STREAM ^ episode_id*golden_ratio); policy_seed=splitmix64(base_seed ^ POLICY_STREAM ^ episode_id*golden_ratio)".to_string(),
            episode_ids: (0..games).collect(),
            env_seeds: (0..games).map(|episode_id| derive_env_seed(base_seed, episode_id)).collect(),
            policy_seeds: (0..games).map(|episode_id| derive_policy_seed(base_seed, episode_id)).collect(),
        },
        output_files: OutputFilesV1 {
            episode_jsonl: EPISODE_JSONL_FILENAME.to_string(),
            manifest_json: MANIFEST_FILENAME.to_string(),
        },
        deck: DeckManifestV1 {
            deck_identifiers: deck_identifiers(),
            deck_hashes: [deck_hash, deck_hash],
            card_count: 60,
            provenance: "kernel Burn mainboard copied from Deck - Mono-Red Burn.dek sideboard=false entries".to_string(),
        },
        git,
        aggregate: aggregate_summaries(summaries),
        variable_metadata: VariableMetadataV1 {
            out_dir: out_dir.display().to_string(),
        },
    }
}

pub fn git_metadata() -> GitMetadataV1 {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            if s.trim().is_empty() {
                GitDirtyFlagV1::Clean
            } else {
                GitDirtyFlagV1::Dirty
            }
        })
        .unwrap_or(GitDirtyFlagV1::Unknown);
    GitMetadataV1 { commit, dirty }
}

pub fn write_rollout_artifacts(
    out_dir: &Path,
    records: &[EpisodeRecordV1],
    manifest: &RunManifestV1,
) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    write_jsonl_atomic(&out_dir.join(EPISODE_JSONL_FILENAME), records)?;
    write_json_pretty_atomic(&out_dir.join(MANIFEST_FILENAME), manifest)?;
    Ok(())
}

pub fn validate_selected_action(
    actions: &[LegalActionCandidateV1],
    selected_index: usize,
    selected_id: &str,
) -> Result<()> {
    let Some(action) = actions.get(selected_index) else {
        return Err(RlContractError(format!(
            "selected action index {selected_index} out of range {}",
            actions.len()
        )));
    };
    if action.record.selected_index as usize != selected_index {
        return Err(RlContractError(format!(
            "selected action transport index mismatch: vector index {selected_index}, record index {}",
            action.record.selected_index
        )));
    }
    if action.record.stable_id != selected_id {
        return Err(RlContractError(format!(
            "selected action id mismatch: expected {}, got {selected_id}",
            action.record.stable_id
        )));
    }
    Ok(())
}

pub fn burn_deck_hash() -> u64 {
    stable_hash_json(&burn_deck_ids()).expect("burn deck ids serialize")
}

fn deck_identifiers() -> [String; 2] {
    [
        "mono_red_burn_mainboard_v1".to_string(),
        "mono_red_burn_mainboard_v1".to_string(),
    ]
}

fn aggregate_summaries(summaries: &[EpisodeTerminalSummaryV1]) -> RunAggregateV1 {
    let mut aggregate = RunAggregateV1 {
        wins: 0,
        losses: 0,
        draws: 0,
        halted: 0,
        total_decisions: 0,
    };
    for summary in summaries {
        aggregate.total_decisions += summary.decision_count;
        match summary.outcome {
            TerminalOutcomeV1::Win => aggregate.wins += 1,
            TerminalOutcomeV1::Loss => aggregate.losses += 1,
            TerminalOutcomeV1::Draw => aggregate.draws += 1,
            TerminalOutcomeV1::Halted => aggregate.halted += 1,
        }
    }
    aggregate
}

fn terminal_summary(
    episode_id: u64,
    winner: Option<PlayerId>,
    terminal_reason: String,
    decision_count: u64,
) -> EpisodeTerminalSummaryV1 {
    let (outcome, terminal_reward) = match winner {
        Some(PlayerId::P0) => (TerminalOutcomeV1::Win, [1, -1]),
        Some(PlayerId::P1) => (TerminalOutcomeV1::Loss, [-1, 1]),
        None => (TerminalOutcomeV1::Draw, [0, 0]),
        Some(other) => panic!("unsupported winner player id {}", other.0),
    };
    EpisodeTerminalSummaryV1 {
        episode_id,
        outcome,
        winner: winner.map(Into::into),
        terminal_reward,
        terminal_reason,
        decision_count,
    }
}

fn halted_summary(
    episode_id: u64,
    terminal_reason: String,
    decision_count: u64,
) -> EpisodeTerminalSummaryV1 {
    EpisodeTerminalSummaryV1 {
        episode_id,
        outcome: TerminalOutcomeV1::Halted,
        winner: None,
        terminal_reward: [0, 0],
        terminal_reason,
        decision_count,
    }
}

fn push_terminal(
    records: &mut Vec<EpisodeRecordV1>,
    summary: &EpisodeTerminalSummaryV1,
    diagnostic_state_hash: u64,
) {
    records.push(EpisodeRecordV1::Terminal {
        schema_version: EPISODE_SCHEMA_VERSION,
        episode_id: summary.episode_id,
        terminal_outcome: summary.outcome,
        winner: summary.winner,
        terminal_reward: summary.terminal_reward,
        terminal_reason: summary.terminal_reason.clone(),
        decision_count: summary.decision_count,
        diagnostic_state_hash,
    });
}

fn player_status_v1(player: &crate::state::PlayerState) -> PlayerStatusV1 {
    PlayerStatusV1 {
        has_lost: player.has_lost,
        lands_played_this_turn: player.lands_played_this_turn,
        drew_from_empty: player.drew_from_empty,
        draws_this_turn: player.draws_this_turn,
    }
}

fn card_ref(state: &GameState, id: ObjectId) -> Result<CardStableRefV1> {
    let object = state
        .objects
        .try_get(id)
        .ok_or_else(|| RlContractError(format!("object id {} missing", id.0)))?;
    Ok(CardStableRefV1 {
        arena_id: id.0,
        card_db_id: object.card_def,
        owner: object.owner.into(),
        controller: object.controller.into(),
        zone: object.zone,
        zone_change_count: object.zone_change_count,
    })
}

fn public_card(state: &GameState, id: ObjectId) -> Result<CardPublicV1> {
    let object = state
        .objects
        .try_get(id)
        .ok_or_else(|| RlContractError(format!("object id {} missing", id.0)))?;
    Ok(CardPublicV1 {
        stable: card_ref(state, id)?,
        card_name: card_name(object.card_def),
        tapped: object.tapped,
        summoning_sick: object.summoning_sick,
        damage: object.damage,
        counters: CountersV1 {
            plus1_plus1: object.counters.plus1_plus1,
        },
        attachments: object.attachments.iter().map(|id| id.0).collect(),
        plotted_turn: object.plotted_turn,
    })
}

fn public_cards(state: &GameState, ids: &[ObjectId]) -> Result<Vec<CardPublicV1>> {
    ids.iter().map(|&id| public_card(state, id)).collect()
}

fn private_card(state: &GameState, id: ObjectId) -> Result<CardPrivateV1> {
    let object = state
        .objects
        .try_get(id)
        .ok_or_else(|| RlContractError(format!("object id {} missing", id.0)))?;
    Ok(CardPrivateV1 {
        stable: card_ref(state, id)?,
        card_name: card_name(object.card_def),
    })
}

fn stack_public(state: &GameState) -> Result<Vec<StackItemPublicV1>> {
    state
        .stack
        .iter()
        .enumerate()
        .map(|(stack_index, item)| stack_item_public(state, stack_index as u32, item))
        .collect()
}

fn stack_item_public(
    state: &GameState,
    stack_index: u32,
    item: &StackItem,
) -> Result<StackItemPublicV1> {
    Ok(StackItemPublicV1 {
        stack_index,
        source: card_ref(state, item.source)?,
        controller: item.controller.into(),
        targets: item
            .targets
            .iter()
            .map(|&target| target_ref(state, target))
            .collect::<Result<Vec<_>>>()?,
        is_trigger_or_ability: item.inline_effect.is_some(),
        is_flashback: item.is_flashback,
        mode_chosen: item.mode_chosen,
        madness_offer: item.madness_offer,
        kicked: item.kicked,
    })
}

fn target_ref(state: &GameState, target: Target) -> Result<TargetRefV1> {
    match target {
        Target::Player(player) => Ok(TargetRefV1::Player {
            player: player.into(),
        }),
        Target::Object(object) => Ok(TargetRefV1::Object {
            object: card_ref(state, object)?,
        }),
    }
}

fn visible_projection_hash(observation: &ObservationV1) -> Result<u64> {
    #[derive(Serialize)]
    struct ObservationHashInput<'a> {
        schema_version: u32,
        kernel_version: &'a str,
        surface_version: u32,
        card_db_hash: u64,
        acting_player: PlayerSeatV1,
        step_index: u64,
        projection: &'a PublicObservationProjectionV1,
        own_hand: &'a [CardPrivateV1],
    }

    stable_hash_json(&ObservationHashInput {
        schema_version: observation.schema_version,
        kernel_version: &observation.kernel_version,
        surface_version: observation.surface_version,
        card_db_hash: observation.card_db_hash,
        acting_player: observation.acting_player,
        step_index: observation.step_index,
        projection: &observation.projection,
        own_hand: &observation.own_hand,
    })
}

fn push_action(
    out: &mut Vec<LegalActionCandidateV1>,
    semantic: ActionSemanticV1,
    surface_action: SurfaceAction,
) -> Result<()> {
    let record = make_legal_action_v1(out.len() as u32, semantic, None)?;
    out.push(LegalActionCandidateV1 {
        record,
        surface_action,
    });
    Ok(())
}

fn ensure_unique_action_ids(actions: &[LegalActionCandidateV1]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for action in actions {
        if !seen.insert(action.record.stable_id.clone()) {
            return Err(RlContractError(format!(
                "duplicate stable legal action id within one decision: {}",
                action.record.stable_id
            )));
        }
    }
    Ok(())
}

fn subsets(ids: &[ObjectId]) -> Result<Vec<Vec<ObjectId>>> {
    if ids.len() > MAX_SUBSET_OBJECTS {
        return Err(RlContractError(format!(
            "legal subset decision has {} candidates, exceeding fail-closed cap {MAX_SUBSET_OBJECTS}",
            ids.len()
        )));
    }
    let count = 1usize << ids.len();
    let mut out = Vec::with_capacity(count);
    for mask in 0..count {
        let mut picked = Vec::new();
        for (i, &id) in ids.iter().enumerate() {
            if (mask & (1usize << i)) != 0 {
                picked.push(id);
            }
        }
        out.push(picked);
    }
    Ok(out)
}

fn permutations(n: usize) -> Result<Vec<Vec<usize>>> {
    if n > MAX_TRIGGER_ORDER_OBJECTS {
        return Err(RlContractError(format!(
            "trigger order decision has {n} pending triggers, exceeding fail-closed cap {MAX_TRIGGER_ORDER_OBJECTS}"
        )));
    }
    let mut current: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    permute_from(0, &mut current, &mut out);
    Ok(out)
}

fn permute_from(start: usize, current: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
    if start == current.len() {
        out.push(current.clone());
        return;
    }
    for i in start..current.len() {
        current.swap(start, i);
        permute_from(start + 1, current, out);
        current.swap(start, i);
    }
}

fn rng_below(rng: &mut SplitMix64, n: usize) -> usize {
    debug_assert!(n > 0);
    (rng.next_u64() % n as u64) as usize
}

fn stable_hash_json<T: Serialize>(value: &T) -> Result<u64> {
    let bytes = serde_json::to_vec(value)?;
    Ok(fnv1a64(&bytes))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn write_jsonl_atomic(path: &Path, records: &[EpisodeRecordV1]) -> Result<()> {
    write_atomic(path, |writer| {
        for record in records {
            serde_json::to_writer(&mut *writer, record)?;
            writer.write_all(b"\n")?;
        }
        Ok(())
    })
}

fn write_json_pretty_atomic(path: &Path, manifest: &RunManifestV1) -> Result<()> {
    write_atomic(path, |writer| {
        serde_json::to_writer_pretty(&mut *writer, manifest)?;
        writer.write_all(b"\n")?;
        Ok(())
    })
}

fn write_atomic(
    path: &Path,
    write_fn: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    let tmp = tmp_path(path);
    if tmp.exists() {
        fs::remove_file(&tmp)?;
    }
    {
        let file = File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        write_fn(&mut writer)?;
        writer.flush()?;
    }
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

fn tmp_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!("{file_name}.tmp"))
}

#[allow(dead_code)]
fn _assert_game_object_is_visible_data(_: &GameObject) {}
