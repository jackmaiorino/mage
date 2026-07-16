# MTG Kernel Science-Ready Roadmap

Status: active
Roadmap baseline: `a78e9a2f81b0fdd7eec4c692c837e25656794ebb` (`Kernel: Prove evaluator failure boundaries`)
Scope: the nine canonical Pauper decks below in best-of-one play. Sideboarding and best-of-three are a separate, deferred gate.

## Decision summary

The kernel is science-ready when it can train and evaluate policies over all nine pinned BO1 decks without unsupported cards or mechanics, reproduce every accepted run from a clean clone, and publish a complete seat-balanced 9x9 matchup matrix whose primary result uses sampled policy actions. Greedy evaluation remains a required secondary diagnostic, not the promotion result.

These nine decks are the complete Pauper meta pool carried forward from the previous project. A result over Burn/Rally, a convenient subset, or only the newly implemented decks cannot close this roadmap: all nine mainboards must be implemented, trained against, and evaluated in the final full-pool protocol before any concluding science claim.

The current checkpoint has a strong deterministic RL and artifact foundation, but complete-deck coverage is still Burn plus Rally. Metadata-derived basic lands plus the reusable Counterspell/Dispel stack-interaction slice now work across the other decks, without making any incomplete deck runnable. The generated pool metadata freezes all nine decks, and the schema-v2 registry at the legacy `cards_v1.json` path declares exact memberships and fail-closed engine capabilities for the 132 deck cards it contains; eighteen Spy cards are still absent, and most registered cards still lack executable behavior. Card-pool expansion therefore precedes claims about general Pauper learning.

The implementation order is:

1. Preserve the certified Burn/Rally baseline. The two designated local corpora now pass tracked content-lock and provenance gates and each replay 40/40 with zero divergence.
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
- **Training viability:** before the long full-pool run, benchmark end-to-end throughput and the stable sampler over the observed legal-action-width distribution from all nine decks, then record the hardware and capacity estimate. A faster selector changes the sampler version and requires new deterministic vectors; it cannot silently replace the accepted algorithm.
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
- Greedy V1 and sampled V3 evaluator lanes. Sampled V3 is currently a Burn-mirror, head-versus-update-zero contract with one actor-local action stream per physical seat, shared across the two pair legs; its stream keys exclude policy role and game/leg. V3 replaces the frozen Torch-dependent V2 selector with exact binary32-to-Decimal softmax, Hamilton-apportioned `2**64`-unit mass, and one SplitMix64 draw per decision. Greedy remains the secondary lane.
- Source-level card behavior for the complete Burn and Rally mainboards. Chain Lightning now has explicit payment, retarget, target, repeat-copy, counter/fizzle, and copy-departure behavior; the RL observation/session contracts are schema v3.
- Tracked Phase-0 content locks for `burn_mirror_v6` and `rally_mirror_v2`: all 40 trace paths, raw-byte sizes and SHA-256 digests per corpus, each `manifest.json`, and an aggregate digest are embedded into the replay gate. Designated-corpus replay now fails before parsing a trace on non-`LOCKED` status or any missing, extra, or changed replay input.
- Generated `pauper_pool_v1.json` and `pauper_support_v1.json` metadata that pins all nine normalized 60+15 rosters, exact registry membership, current support blockers, token dependencies, source hashes, and raw pool/registry hashes.
- The support manifest currently classifies 32 unique cards as `full`, zero as `partial`, and 118 as `no_effect`. Rally and Burn are both 60/60 `full` at the mainboard card-behavior layer, and both designated content-locked, fixed-provenance reference corpora replay 40/40 with zero divergence. The first Phase-1 substrate slice adds metadata-derived intrinsic mana for every registered basic land and a shared fail-closed engine-capability/preflight contract.
- Counterspell and Dispel now compose one generated counter-target effect with separate reusable any-spell and instant-spell stack filters. The target engine rejects a cast unless some complete mandatory target assignment exists, recursively filters dependent target prefixes, excludes the announcing spell itself, and silently retains the original printed mode index when only one modal branch is viable. These changes preserve schema-v3 observation/action/session identities while preventing targetless dead-end decisions.
- `xmage_counter_reference_windows_v1.json` pins the raw manifest/trace hashes and six exact Counterspell/Dispel target selections from Caw-Gates, Terror, and Faeries XMage games, plus bounded visible graveyard deltas. The source run did not record its Java commit, so this evidence is explicitly a source-hash-backed micro-reference gate, not fixed-provenance whole-deck parity; the ignored materialization test rehashes and reparses the local source logs when available.

### Card coverage

"Effect-backed" means a card copy has explicit `full` engine capability plus its required spell, permanent, or mana program. Registry rules metadata alone does not qualify. A land mapped to `no_effect` cannot be played, and a spell mapped to `no_effect` cannot be cast.

| Archetype | Main registered | Main effect-backed | Side registered | Side effect-backed | Current limiting fact |
|---|---:|---:|---:|---:|---|
| Wildfire | 60/60 | 7/60 | 15/15 | 3/15 | Its Mountain, Forest, and Swamp copies are usable; spells/nonbasics remain fail-closed |
| Rally | 60/60 | 60/60 | 15/15 | 8/15 | Locked fixed-provenance replay passes 40/40 |
| Affinity | 60/60 | 8/60 | 15/15 | 5/15 | Great Furnace, Swamp, and Galvanic Blast only |
| Elves | 60/60 | 13/60 | 15/15 | 0/15 | Snow-Covered Forest works; spells remain fail-closed |
| Spy | 21/60 | 4/60 | 4/15 | 1/15 | Forest/Swamp work; 39 main and 11 side copies are absent from the registry |
| Burn | 60/60 | 60/60 | 15/15 | 11/15 | Mainboard is the current complete baseline |
| Terror | 60/60 | 22/60 | 15/15 | 0/15 | Island, Counterspell, and Dispel work; the remaining blue spell core is next |
| CawGates | 60/60 | 8/60 | 15/15 | 5/15 | Island and Counterspell work; Gates and remaining spells stay fail-closed |
| Faeries | 60/60 | 24/60 | 15/15 | 1/15 | Island, Counterspell, and Dispel work; creatures and remaining spells stay fail-closed |

The registry currently contains 135 definitions: 132 deck cards and three tokens. `pool_decks` now lists all nine sources in canonical order, and the eight already-present Spy cards declare exact Spy membership. Seven of those are Spy mainboard names shared with other decks; fourteen Spy mainboard names and four additional sideboard-only names still need new records.

Chain Lightning's implementation checkpoint is backed by unit tests for unpayable and declined payment, copied-stack identity, retargeting, recursive copies, illegal-target fizzles, copy-aware counters and flashback replacement, target-pool filtering, RL serialization/action semantics, and snapshot/restore determinism. The locked `rally_mirror_v2` corpus contains 40 games, 15 logged payment prompts, six accepted payments, six retarget prompts, and three accepted retargets. Its manifest pins ReferenceRules v2 and Java oracle commit `0723fc0c2be922af47b0ef0539f28114cc23b998`; the runtime provenance gate passes. Replay reaches GameOver in all 40 traces, matches all 40 winners, and reports zero divergence and zero halt. Closing the former residuals required mirroring Java's rendered-cast-candidate equivalence in the replay comparator while preserving exact chosen object identity, filtering attackers killed before blocker declaration, and matching XMage's deterministic generic-mana pool spending order.

The formal Burn and Rally evidence is now content-locked locally by tracked metadata; `kernel/CORPUS_CONTENT_LOCKS.md` defines the byte-level algorithm. This makes mutation or substitution detectable, but it does not make the ignored corpus payloads available from a clean clone. Durable content-addressed storage, authenticated retrieval, and post-download digest verification remain part of the independent-repository release gate.

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
- Preserve Chain Lightning's implemented copy, retarget, repeat-copy, counter/fizzle, schema-v3 RL, snapshot, and trace-parser regressions.
- Preserve the content-lock- and provenance-gated, zero-divergence 40/40 replays for `burn_mirror_v6` and `rally_mirror_v2`.
- Preserve all Burn/Rally goldens, replays, RL contracts, and evaluator proofs.

Exit: Burn and Rally pass every BO1 gate below with no conditional halt.

### Phase 1 - General card substrate

- Generate ordinary permanent resolution and mana behavior from card metadata.
- Keep runtime support and generated support reports on the same per-definition capability declaration, and preflight both token-free 60-card decks before any science-facing environment shuffles or constructs state.
- Replace fixed target shapes with zone/filter/cardinality descriptions.
- Generalize costs, library operations, triggers, continuous effects, tokens, counters, and attachments.
- Make every unsupported reachable branch explicit in the support manifest and runtime diagnostics.

Schema-v3-neutral work may add fixed-color mana, exact-one filtered targets, deterministic draw/mill/counter/zone moves, graveyard-count cost reductions, and no-choice triggers. Before the first policy-visible ordered-library choice, variable target count, Escape selection, Ward payment, freeze marker, or other resumable multi-stage effect, make one deliberate schema-v4 migration: add a generic effect continuation/choice machine and perspective-safe known-library state, bump the Rust observation/legal-action/session contracts and every Python feature/model/policy/audit/trainer/checkpoint identity together, freeze v3, and reject cross-schema resume. Do not grow more card-specific `pending_*` state to avoid this boundary.

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

- Before phase promotion, run at least 16 paired seeds against every previously accepted deck under the separately versioned multi-deck seed contract described below. Current sampled V3 is not that contract.
- Require natural terminals, exact rerun determinism, zero unsupported cards/mechanics, and explicit seat-stratified results.
- After Spy, replace these smoke-sized gates with the final matrix protocol below.

## Evaluation protocol

### Primary: sampled policy

Science claims use actions sampled from the policy logits, with all sampling performed by an explicit, versioned deterministic RNG. The manifest records policy snapshot hashes, deck hashes, environment and policy seed derivation versions, pair count, temperature, and any logit transformation.

#### Current sampled V3 seed and selector contract

Sampled V2 remains a frozen legacy identity: it selected with `torch.multinomial`, whose seeded categorical stream is not stable across supported Torch releases. Sampled V3 does not reinterpret that schema. It converts finite CPU binary32 logits exactly to Decimal, subtracts the maximum at 256-digit precision, evaluates exponentials at 80-digit round-half-even precision (with deltas strictly below -128 assigned zero mass), apportions exactly `2**64` units by largest remainder with legal-index tie-breaking, and selects by inverse CDF using the first SplitMix64-v1 output seeded by the action seed. Exact vectors cover the softmax weights, RNG outputs, CDF boundaries, translation invariance, and independence from process-global Python/Torch RNG state and Decimal context.

V3 preserves V2's common-random-number schedule. For each pair it uses actor-local action RNG streams keyed by physical seat. The physical-P0 stream and physical-P1 stream are each shared across both pair legs; action-stream derivation intentionally excludes candidate/baseline role and game/leg. This is still a single-deck Burn-mirror contract, not a multi-deck shuffle or action schedule.

#### Future multi-deck seed contract

Cross-deck evaluation requires a new, separately versioned schema and seed-derivation contract. That protocol may key shuffle/determinization and action streams by logical deck role so that each deck carries the corresponding common-random-number streams when the decks swap physical seats. The exact mapping must be explicit in manifests and tested for seat-swap, row-order, and rerun invariance. It must not be retrofitted into V3 or described as behavior V3 already provides.

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
2. Vendor the generated card database, token definitions, canonical deck manifests, and source hashes needed at build/runtime. Publish the formal replay corpora to durable content-addressed storage and provide an authenticated retrieval command that verifies the tracked lock before use. XMage Java sources remain provenance and oracle inputs, not runtime dependencies.
3. Pin Rust, Python, dependency-lock, schema, seed-derivation, and artifact-format versions.
4. Add Linux and Windows CI that runs formatting/lint checks, `cargo test --locked`, the Python test suite, deck/hash validation, all unit/golden tests, and a bounded deterministic mirror/cross-deck smoke.
5. From a brand-new clone with no parent XMage checkout, build the environment, run one training update, recover/validate its store, run sampled and greedy paired evaluation, and reproduce checked-in expected artifact digests.
6. Verify generated outputs, caches, model artifacts, databases, and `target/` remain untracked and the clean-clone workflow ends with a clean worktree.
7. Tag the first release only after CI and clean-clone artifacts record the independent repository commit, original Mage checkpoint, nine deck hashes, card-database hash, environment binary hash, and protocol versions.

The project is finished for science-ready BO1 only when that release artifact and the accepted sampled-primary 9x9 matrix can be independently reproduced.
