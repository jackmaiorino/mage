# MTG Kernel Science-Ready Roadmap

Status: active
Roadmap baseline: `a78e9a2f81b0fdd7eec4c692c837e25656794ebb` (`Kernel: Prove evaluator failure boundaries`)
Scope: the nine canonical Pauper decks below in best-of-one play. Sideboarding and best-of-three are a separate, deferred gate.

## Decision summary

The kernel is science-ready when it can train and evaluate policies over all nine pinned BO1 decks without unsupported cards or mechanics, reproduce every accepted run from a clean clone, and publish a complete seat-balanced 9x9 matchup matrix whose primary result uses sampled policy actions. Greedy evaluation remains a required secondary diagnostic, not the promotion result.

The current checkpoint has a strong deterministic RL and artifact foundation, but card coverage is still effectively Burn plus Rally. The generated pool metadata now freezes all nine decks, and `cards_v1.json` declares their exact memberships for the 132 deck cards it contains; eighteen Spy cards are still absent from the registry, and most registered cards still lack executable behavior. Card-pool expansion therefore precedes claims about general Pauper learning.

The implementation order is:

1. Certify Burn and Rally, including Chain Lightning's live copy branch.
2. Build reusable card, target, cost, zone, trigger, combat, and token primitives.
3. Terror.
4. Faeries.
5. Elves.
6. Affinity.
7. Wildfire.
8. CawGates.
9. Spy.
10. Run the final sampled and greedy 9x9 matrices, then complete independent-repository and clean-clone release gates.

## Science-ready completion definition

All of the following are required:

- **Pinned scope:** the exact nine mainboards below load as 60-card decks and match their canonical full-text source hashes under `utf8_text_crlf_v1`. A changed hash creates a new experimental protocol; it does not silently replace this one.
- **Rules completeness:** every mainboard spell is castable, every mana source works, every reachable card branch is modeled, and no BO1 run can reach `no_effect`, an unimplemented decision, or `UnsupportedMechanic`.
- **Reference evidence:** each rules primitive has unit coverage, each deck has a deterministic public-engine golden, and fixed-provenance XMage traces replay without unexplained divergence.
- **Deterministic environment:** deck construction, shuffling, policy sampling, seat swaps, observations, legal-action identities, checkpoints, and artifacts are versioned and reproducible from explicit seeds.
- **Training integrity:** the already-proven constant-work, append-only training/checkpoint path remains crash-consistent and hash-validated for every deck configuration.
- **Evaluation integrity:** sampled-policy paired evaluation is primary; deterministic greedy paired evaluation is secondary. Both reject halted, truncated, inconsistent, or provenance-drifting runs.
- **Pool evidence:** every diagonal mirror and every ordered cross-deck cell passes the gates below, producing the complete 9x9 report with seat strata, draws, paired uncertainty, and exact provenance.
- **Reproducible finish:** a clean clone of the independent kernel repository passes CI and reproduces the published smoke, golden, and evaluation artifact digests without access to an XMage working tree at runtime.

Sideboard and BO3 support are deliberately excluded from this BO1 definition. They have their own promotion gate later in this document.

## Canonical BO1 decks

The hash covers the complete UTF-8 `.dek` source text, including its sideboard, even though the first science-ready protocol uses only the 60-card mainboard. `utf8_text_crlf_v1` strictly decodes UTF-8, normalizes CRLF, LF, and bare-CR line boundaries, then hashes the complete text encoded with CRLF boundaries without adding or removing a final newline. This keeps the pinned hash portable across Git checkout line endings without excluding any XML content.

| Order | Archetype | Canonical source | SHA-256 |
|---:|---|---|---|
| 1 | Wildfire | `Deck - Jund Wildfire.dek` | `cff35798ff724888a9e5a4520dd55e70b0c628a55908697aa116089d8fd980a5` |
| 2 | Rally | `Deck - Mono Red Rally.dek` | `4b5019bd08f9387aeabebdca0d90aaa10dfd75fc75ed3a87c95a2fabf4dba834` |
| 3 | Affinity | `Deck - Grixis Affinity.dek` | `4a41135ac6d14960e75ddce8e9980c0505c0b71a9c08a2e10578a10d2fcf8801` |
| 4 | Elves | `Deck - Elves.dek` | `6b040933c9b3506536e7dc71c94dcaf5f16c7ade43a3d0f7f9b240be6deb0d87` |
| 5 | Spy | `Deck - Spy Combo.dek` | `f08177d5ed133b18312f59649d1155e15b5074ababeaabcdf3f31ded650308ba` |
| 6 | Burn | `Deck - Mono-Red Burn.dek` | `4ebba6b42bb27a0ea55001cee133aada81f0dffd8661b46b012fc5026675aa32` |
| 7 | Terror | `Deck - Mono-Blue Terror.dek` | `8ba22b67b843bc49a421e1c2814c4dd24a04ab2b45131ec7876a8312115a9fda` |
| 8 | CawGates | `Deck - Caw-Gates.dek` | `72c2bbf76a7fd219349a0ad81c44dc6166b4a797a1f66fe9b5a5de79aa6cdc14` |
| 9 | Faeries | `Deck - Mono-Blue Faeries.dek` | `8cb962c4ccee6a5f8c0c70fc27c17d13323d13606c82b9b12b8985aa87e0f344` |

Canonical order is part of the protocol and is reused for manifests, seed derivation, matrix rows, and matrix columns.

## Current checkpoint truth

### What is already implemented

- A deterministic Burn-mirror engine/RL session with versioned observations and stable legal-action identities.
- Append-only, hash-linked training artifacts and checkpoints with constant-work update and recovery behavior.
- Failure-boundary and path-safety proofs for training and evaluation publication.
- Deterministic paired head-versus-update-zero evaluation with seat swaps, natural-terminal enforcement, W/D/L and seat strata, paired bootstrap intervals, an exact paired sign test, and immutable artifact validation.
- Greedy V1 and sampled V2 evaluator lanes. Sampled V2 is currently a Burn-mirror, head-versus-update-zero contract with one actor-local action stream per physical seat, shared across the two pair legs; its stream keys exclude policy role and game/leg. Greedy remains the secondary lane.
- Source-level card behavior for the Burn mainboard and Rally mainboard, except that Chain Lightning explicitly halts if its spell-copy choice becomes live.
- Generated `pauper_pool_v1.json` and `pauper_support_v1.json` metadata that pins all nine normalized 60+15 rosters, exact registry membership, current support blockers, token dependencies, source hashes, and raw pool/registry hashes.

### Card coverage

"Effect-backed" means a card copy has an implemented spell or mana program. Registry metadata alone does not qualify. A land with no mana program can be played but is not a usable mana source; a spell mapped to `no_effect` is uncastable.

| Archetype | Main registered | Main effect-backed | Side registered | Side effect-backed | Current limiting fact |
|---|---:|---:|---:|---:|---|
| Wildfire | 60/60 | 2/60 | 15/15 | 3/15 | Only Mountain is usable in the mainboard |
| Rally | 60/60 | 60/60 | 15/15 | 8/15 | Chain Lightning's affordable copy branch halts |
| Affinity | 60/60 | 7/60 | 15/15 | 5/15 | Great Furnace and Galvanic Blast only |
| Elves | 60/60 | 0/60 | 15/15 | 0/15 | Entire deck is fail-closed |
| Spy | 21/60 | 0/60 | 4/15 | 0/15 | 39 main and 11 side copies are absent from the registry |
| Burn | 60/60 | 60/60 | 15/15 | 11/15 | Mainboard is the current complete baseline |
| Terror | 60/60 | 0/60 | 15/15 | 0/15 | Entire deck is fail-closed |
| CawGates | 60/60 | 0/60 | 15/15 | 3/15 | Entire mainboard is fail-closed |
| Faeries | 60/60 | 0/60 | 15/15 | 0/15 | Entire deck is fail-closed |

The registry currently contains 135 definitions: 132 deck cards and three tokens. `pool_decks` now lists all nine sources in canonical order, and the eight already-present Spy cards declare exact Spy membership. Seven of those are Spy mainboard names shared with other decks; fourteen Spy mainboard names and four additional sideboard-only names still need new records.

Keyword representation is also not rules completeness. For example, defender is represented but not enforced when declaring attackers, trample still uses a no-trample combat path, and deathtouch, lifelink, protection, hexproof, and ward need real rules behavior.

### Reusable implementation clusters

| Cluster | Required capability | Primary consumers |
|---|---|---|
| Permanent and mana substrate | Generic permanent resolution, colored/colorless/choice mana, tapped entry, chosen colors, indestructible | Every remaining deck |
| Targets and costs | Zone-aware filters, min/max target counts, dependent targets, filtered sacrifices/taps, life and graveyard costs, once-per-turn use | Every remaining deck |
| Library and zones | Search, reveal, choose subsets, order top/bottom, shuffle, scry, mill, cycling, linked exile and return | Terror, Faeries, Elves, Wildfire, CawGates, Spy |
| Stack interaction | General counters, filtered counters, counter-unless-pay, ward, modal spells, spell copy | Rally, Terror, Faeries, CawGates |
| Triggers and continuous effects | Generic ETB/LTB/dies/cast/sacrifice/combat triggers, dynamic counts, temporary effects, type changes | All non-Burn decks |
| Combat | Defender, trample, deathtouch, lifelink, blocker-count restrictions, protection and hexproof | Elves, Wildfire, CawGates, Spy |
| Tokens, counters, attachments | Map, Spawn, Clue, Food, Bird, Treasure, Hero, token copies; stun, toughness and lore counters; Aura/equip/bestow | Affinity, Wildfire, Elves, Terror, CawGates, Spy |
| Persistent object/game state | Chosen color, linked cards, transformed face, Saga chapters, initiative/dungeon progress | CawGates, Elves, Spy |

These should be generic effect and decision primitives, not a growing per-card `Special` enum. Card-specific definitions should compose tested primitives and fail closed when a reachable branch has no representation.

## Dependency-ordered delivery

### Phase 0 - Seal the current baseline

- Keep the generated pool and support manifests current, with explicit `full`, `partial`, and `no_effect` states over all 150 deck cards.
- Preserve the assertion that the authoritative deck set equals these nine canonical full-text hashes; `unresolved: []` must not be possible when a source or registry record was never included.
- Finish Chain Lightning's copy, retarget, and repeat-copy decisions.
- Preserve all Burn/Rally goldens, replays, RL contracts, and evaluator proofs.

Exit: Burn and Rally pass every BO1 gate below with no conditional halt.

### Phase 1 - General card substrate

- Generate ordinary permanent resolution and mana behavior from card metadata.
- Replace fixed target shapes with zone/filter/cardinality descriptions.
- Generalize costs, library operations, triggers, continuous effects, tokens, counters, and attachments.
- Make every unsupported reachable branch explicit in the support manifest and runtime diagnostics.

Exit: foundational unit suites pass and an unsupported card cannot be mistaken for a playable one.

### Phase 2 - Terror

Terror is the lowest-novelty route to the reusable blue core: Island mana, draw and ordered library choices, self/target mill, graveyard-count cost reduction, Counterspell/Dispel, cycling, flashback with life, escape, freeze, ward, and spell-cast token triggers.

Exit: Terror mirror plus Burn, Rally, and Terror cross-deck gates pass.

### Phase 3 - Faeries

Reuse Island, Counterspell, and Dispel. Add flash timing, ninjutsu as a combat-window special action, combat-damage triggers, Faerie-count counters, counter-unless-pay, bounce/untap, loot, backup, hexproof, and temporary power reduction.

Exit: Faeries mirror and all cross-deck cells against accepted decks pass.

### Phase 4 - Elves

Add mana creatures, dynamic Elf counts, filtered tap/return costs, creature/land selection modes, Forestcycling, Food, changeling, bestow, and initiative/dungeon state. This also completes five exact Spy mainboard cards and Nyxborn Hydra for Wildfire.

Exit: Elves mirror and all accepted cross-deck cells pass.

### Phase 5 - Affinity

Implement artifact lands and bridges, affinity cost reduction, sacrifice-and-draw costs, ETB/LTB triggers, graveyard hate and recursion, investigate, damage sweeps, equipment, Job select/Hero attachment, subtype grants, and stun counters.

Exit: Affinity mirror and all accepted cross-deck cells pass.

### Phase 6 - Wildfire

Reuse eleven Affinity cards plus the Elves bestow work. Add typed-basic fetching, land destruction with indestructible handling, Map and Spawn tokens, colorless-cast and Eldrazi-sacrifice triggers, scry, graveyard-to-library triggers, and creature/land recursion.

Exit: Wildfire mirror and all accepted cross-deck cells pass.

### Phase 7 - CawGates

Reuse the blue core, then add Gate chosen-color and count state, linked exile, lifelink and dies triggers, Saga chapters and transformation, embalm, multi-mode/multi-target spells, color-based prevention, same-name search, and protection from monocolored.

Exit: CawGates mirror and all accepted cross-deck cells pass.

### Phase 8 - Spy

- Add Spy and its eighteen missing unique card records to the generated registry.
- First implement defender mana, filtered tap/sacrifice costs, Land Grant, basic search, cycling, and Omen casting.
- Then prove the core no-land line: Balustrade Spy mills the library, Dread Return sacrifices three creatures, and Lotleth Giant resolves the graveyard-count kill.
- Finish linked hand exile, defender combat restrictions, Wall of Roots counters, three-blocker restriction, and remaining interaction.

Exit: both the land-hit and landless-combo branches have goldens and replay evidence; Spy mirror and all cross-deck cells pass.

## Promotion gates

Every phase must pass gates in this order. A later gate cannot waive an earlier one.

### 1. Registry gate

- Parse the pinned `.dek` file to exactly 60 mainboard and 15 sideboard cards.
- Resolve every in-scope mainboard card and required token.
- Match the deck hash and card-database hash recorded in the run manifest.
- Report zero `no_effect` or partial mainboard definitions.

### 2. Unit gate

- Test every new effect, cost, target filter, trigger, continuous effect, combat rule, and state field in positive and negative cases.
- Cover payment selection, target loss/fizzle, indestructible, zone-change identity, trigger order, illegal action rejection, and serialization/hash sensitivity.
- Treat checked-in XMage Java card implementations as the card-semantics oracle.

### 3. Golden gate

- Add a deterministic `<deck>_goldfish.rs` driven only through the public engine after setup.
- Assert the exact decision sequence, targets/costs, events, mana, zones, life, battlefield, and terminal result.
- Cover every new nontrivial mechanic in at least one golden or reference replay.

### 4. Replay gate

- Capture fixed-provenance XMage traces for the deck and new mechanic branches.
- Match candidate sets, chosen actions, targets, modes, costs, and checkpointed public state.
- Allow zero unexplained divergence, unexpected suppression, halt, or uncastable reference action.

### 5. Mirror gate

- Run at least 32 fixed paired seeds with the deck in both seats.
- Admit only natural win/draw terminals; any halt, truncation, illegal action, or unsupported branch fails the gate.
- Rerunning from identical inputs must reproduce decision, state, terminal, and artifact digests exactly.

### 6. Cross-deck gate

- Before phase promotion, run at least 16 paired seeds against every previously accepted deck under the separately versioned multi-deck seed contract described below. Current sampled V2 is not that contract.
- Require natural terminals, exact rerun determinism, zero unsupported cards/mechanics, and explicit seat-stratified results.
- After Spy, replace these smoke-sized gates with the final matrix protocol below.

## Evaluation protocol

### Primary: sampled policy

Science claims use actions sampled from the policy logits, with all sampling performed by an explicit, versioned deterministic RNG. The manifest records policy snapshot hashes, deck hashes, environment and policy seed derivation versions, pair count, temperature, and any logit transformation.

#### Current sampled V2 seed contract

The current head-versus-update-zero V2 evaluator is a single-deck Burn mirror. For each pair it uses actor-local action RNG streams keyed by physical seat. The physical-P0 stream and physical-P1 stream are each shared across both pair legs; action-stream derivation intentionally excludes candidate/baseline role and game/leg. This is the frozen V2 common-random-number contract. It does not define a multi-deck shuffle or action schedule.

#### Future multi-deck seed contract

Cross-deck evaluation requires a new, separately versioned schema and seed-derivation contract. That protocol may key shuffle/determinization and action streams by logical deck role so that each deck carries the corresponding common-random-number streams when the decks swap physical seats. The exact mapping must be explicit in manifests and tested for seat-swap, row-order, and rerun invariance. It must not be retrofitted into V2 or described as behavior V2 already provides.

The primary report includes candidate-centric W/D/L, score, candidate-as-P0 and candidate-as-P1 strata, favorable/tied/unfavorable pairs, a deterministic paired-bootstrap 95% interval for draw-inclusive score, and the exact two-sided paired sign test. Halted, truncated, provenance-drifting, or inconsistent terminal rows invalidate the complete cell rather than becoming losses or draws.

### Secondary: greedy policy

Run the existing deterministic argmax policy as a separate artifact over the same snapshots and ordered matchup plan. Greedy results diagnose policy sharpness, tie behavior, and sampling sensitivity. They cannot substitute for the sampled result or independently promote a checkpoint.

### Comparators

- **Integrity anchor:** every candidate is evaluated against the frozen update-zero policy for continuity with the current evaluator.
- **Promotion comparator:** every candidate is also evaluated against the last accepted checkpoint under identical sampled and greedy protocols.
- **Pool matrix:** the accepted checkpoint supplies the policy used for every deck in the final 9x9 environment matrix.

The promotion decision must name which comparator supports it. An update-zero gain alone is not evidence of improvement over the accepted checkpoint.

## Final ordered 9x9 matrix

Rows are the evaluated deck and columns are the opponent. Every cell is seat-balanced and independently materialized in this canonical order, including diagonal mirrors and reciprocal off-diagonal cells. Reciprocal cells must agree after candidate/opponent perspective inversion when backed by the same paired games.

Abbreviations: `Wld` Wildfire, `Rly` Rally, `Aff` Affinity, `Elf` Elves, `Spy` Spy, `Brn` Burn, `Ter` Terror, `Caw` CawGates, `Fae` Faeries.

| Row \ Opponent | Wld | Rly | Aff | Elf | Spy | Brn | Ter | Caw | Fae |
|---|---|---|---|---|---|---|---|---|---|
| **Wld** | Wld/Wld | Wld/Rly | Wld/Aff | Wld/Elf | Wld/Spy | Wld/Brn | Wld/Ter | Wld/Caw | Wld/Fae |
| **Rly** | Rly/Wld | Rly/Rly | Rly/Aff | Rly/Elf | Rly/Spy | Rly/Brn | Rly/Ter | Rly/Caw | Rly/Fae |
| **Aff** | Aff/Wld | Aff/Rly | Aff/Aff | Aff/Elf | Aff/Spy | Aff/Brn | Aff/Ter | Aff/Caw | Aff/Fae |
| **Elf** | Elf/Wld | Elf/Rly | Elf/Aff | Elf/Elf | Elf/Spy | Elf/Brn | Elf/Ter | Elf/Caw | Elf/Fae |
| **Spy** | Spy/Wld | Spy/Rly | Spy/Aff | Spy/Elf | Spy/Spy | Spy/Brn | Spy/Ter | Spy/Caw | Spy/Fae |
| **Brn** | Brn/Wld | Brn/Rly | Brn/Aff | Brn/Elf | Brn/Spy | Brn/Brn | Brn/Ter | Brn/Caw | Brn/Fae |
| **Ter** | Ter/Wld | Ter/Rly | Ter/Aff | Ter/Elf | Ter/Spy | Ter/Brn | Ter/Ter | Ter/Caw | Ter/Fae |
| **Caw** | Caw/Wld | Caw/Rly | Caw/Aff | Caw/Elf | Caw/Spy | Caw/Brn | Caw/Ter | Caw/Caw | Caw/Fae |
| **Fae** | Fae/Wld | Fae/Rly | Fae/Aff | Fae/Elf | Fae/Spy | Fae/Brn | Fae/Ter | Fae/Caw | Fae/Fae |

Initial final-run minimums are 128 paired seeds per cell for sampled-primary evaluation and 32 paired seeds per cell for greedy-secondary evaluation. Counts are protocol fields and may be raised before launch, but may not be reduced after the run begins. The matrix is accepted only if all 81 cells validate and the aggregate contains no halt or truncation.

## Deferred sideboard and BO3 gate

BO1 completion does not imply match-play support. BO3 work starts only after the sampled-primary BO1 matrix is accepted.

BO3 promotion requires:

- all 15 sideboard cards for every pinned deck to be fully supported;
- a versioned, deterministic sideboarding policy and legal 60/15 exchange validation;
- game-to-game match state, play/draw choice, and best-of-three termination;
- post-board goldens and reference replays for graveyard hate, color hosers, artifact/enchantment hate, sweepers, and alternate win lines;
- sampled-primary and greedy-secondary post-board mirror/cross-deck matrices with the same failure rules as BO1.

Until this gate passes, reports must say "canonical-mainboard BO1," not "Pauper match" or "BO3."

## Independent repository, CI, and clean-clone finish

This is the final engineering milestone, after the 9x9 BO1 matrix passes.

1. Extract `kernel/` into an independent repository while preserving a documented commit mapping to this XMage checkpoint.
2. Vendor the generated card database, token definitions, canonical deck manifests, and source hashes needed at build/runtime. XMage Java sources remain provenance and oracle inputs, not runtime dependencies.
3. Pin Rust, Python, dependency-lock, schema, seed-derivation, and artifact-format versions.
4. Add Linux and Windows CI that runs formatting/lint checks, `cargo test --locked`, the Python test suite, deck/hash validation, all unit/golden tests, and a bounded deterministic mirror/cross-deck smoke.
5. From a brand-new clone with no parent XMage checkout, build the environment, run one training update, recover/validate its store, run sampled and greedy paired evaluation, and reproduce checked-in expected artifact digests.
6. Verify generated outputs, caches, model artifacts, databases, and `target/` remain untracked and the clean-clone workflow ends with a clean worktree.
7. Tag the first release only after CI and clean-clone artifacts record the independent repository commit, original Mage checkpoint, nine deck hashes, card-database hash, environment binary hash, and protocol versions.

The project is finished for science-ready BO1 only when that release artifact and the accepted sampled-primary 9x9 matrix can be independently reproduced.
