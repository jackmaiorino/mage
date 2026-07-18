package mage.player.ai.rl;

import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.CardScanner;
import mage.cards.repository.CardRepository;
import mage.cards.repository.ExpansionRepository;
import mage.collectors.DataCollectorServices;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerMatch;
import mage.game.match.MatchOptions;
import mage.game.mulligan.MulliganType;
import mage.player.ai.ComputerPlayerRL;
import mage.player.ai.ComputerPlayerUniformMirror;
import mage.players.Player;
import mage.util.RandomUtil;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.RuntimeMXBean;
import java.math.BigDecimal;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.regex.Pattern;

/**
 * Matched seeded-uniform Rally mirror runtime benchmark for XMage.
 *
 * <p>This is deliberately a standalone benchmark process, not a trainer mode.
 * It performs no model inference, persistence, health-monitor kill, timeout,
 * turn cap, or assigned-game-ID accounting. A formal record is valid only when
 * every launched game reaches a strict natural terminal state.</p>
 */
public final class XMageUniformMirrorBenchmark {

    public static final String SCHEMA_VERSION = "xmage_uniform_mirror_benchmark/v2";
    public static final String DECK_RELATIVE_PATH =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono Red Rally.dek";
    public static final String DECK_SHA256 =
            "4b5019bd08f9387aeabebdca0d90aaa10dfd75fc75ed3a87c95a2fabf4dba834";
    private static final long MEASURE_EPISODE_BASE = 1L << 62;
    private static final long EPISODES_PER_PHASE = 1L << 62;
    private static final Pattern COMMIT_PATTERN = Pattern.compile("[0-9a-f]{40}");
    private static final Pattern TRIAL_PATTERN = Pattern.compile("[A-Za-z0-9._-]{1,128}");
    private static final Pattern SECONDS_PATTERN = Pattern.compile("(?:0|[1-9][0-9]*)(?:\\.[0-9]{1,9})?");
    private static final Set<Integer> ACTOR_COUNTS =
            Collections.unmodifiableSet(new HashSet<>(Arrays.asList(1, 4, 8, 16)));
    private static final ThreadLocal<String> FAILURE_STAGE =
            ThreadLocal.withInitial(() -> "process_start");

    private XMageUniformMirrorBenchmark() {
    }

    public static void main(String[] rawArgs) {
        int exitCode;
        try {
            quietLogging();
            if (rawArgs.length == 1 && "--self-test".equals(rawArgs[0])) {
                FAILURE_STAGE.set("self_test");
                runSelfTest();
                exitCode = 0;
            } else {
                FAILURE_STAGE.set("argument_parse");
                exitCode = runBenchmark(Args.parse(rawArgs));
            }
        } catch (Throwable t) {
            Map<String, Object> fatal = new LinkedHashMap<>();
            fatal.put("schema_version", SCHEMA_VERSION);
            fatal.put("record_type", "fatal_error");
            fatal.put("valid", false);
            fatal.put("failure_stage", FAILURE_STAGE.get());
            fatal.put("failure", throwableFingerprint(t));
            System.out.println(toJson(fatal));
            exitCode = 2;
        }
        if (exitCode != 0) {
            System.exit(exitCode);
        }
    }

    private static int runBenchmark(Args args) throws Exception {
        FAILURE_STAGE.set("repository_and_deck_binding");
        Path repoRoot = args.repoRoot.toRealPath();
        if (!repoRoot.isAbsolute()) {
            throw new IllegalArgumentException("--repo-root must resolve to an absolute path");
        }
        Path deckPath = repoRoot.resolve(DECK_RELATIVE_PATH).normalize().toRealPath();
        if (!deckPath.startsWith(repoRoot)) {
            throw new IllegalStateException("canonical deck escaped repository root");
        }
        String observedDeckHash = sha256(deckPath);
        if (!DECK_SHA256.equals(observedDeckHash)) {
            throw new IllegalStateException("canonical Rally deck SHA-256 mismatch: " + observedDeckHash);
        }

        int observedAvailableProcessors = Runtime.getRuntime().availableProcessors();
        FAILURE_STAGE.set("source_binding_before");
        SourceBinding source = SourceBinding.inspect(repoRoot, args.expectedCommit, args.bindingMode);
        FAILURE_STAGE.set("mutable_artifact_binding_before_scanner");
        MutableArtifactGuard artifactGuard = MutableArtifactGuard.captureBeforeScanner(
                repoRoot, args.bindingMode);
        List<String> scannerErrors = new ArrayList<>();
        DataCollectorServices.init(false, false);
        boolean scannerExecuted = "dirty-smoke".equals(args.bindingMode);
        if (scannerExecuted) {
            FAILURE_STAGE.set("card_catalog_scan");
            CardScanner.scan(scannerErrors);
        }
        quietLogging();
        if (!scannerErrors.isEmpty()) {
            throw new IllegalStateException("card scanner errors: " + scannerErrors);
        }
        long observedCardContentVersion = -1L;
        long observedExpansionContentVersion = -1L;
        if (!scannerExecuted) {
            FAILURE_STAGE.set("strict_catalog_version_binding");
            observedCardContentVersion = CardRepository.instance.getContentVersionFromDB();
            observedExpansionContentVersion = ExpansionRepository.instance.getContentVersionFromDB();
        }
        FAILURE_STAGE.set("deck_materialization");
        DeckResources decks = DeckResources.load(deckPath);
        FAILURE_STAGE.set("runtime_binding_before");
        RuntimeBinding runtimeBefore = RuntimeBinding.capture(args);

        FAILURE_STAGE.set("warmup_phase");
        long inferenceBefore = initializeComputerPlayerRlSilently();
        int activationFailuresBefore = ComputerPlayerRL.getRLActivationFailureCount();
        PhaseResult warmup = runPhase("warmup", 0L, args.warmupNanos, args, decks);
        FAILURE_STAGE.set("measurement_phase");
        PhaseResult measurement = runPhase(
                "measurement", MEASURE_EPISODE_BASE, args.measureNanos, args, decks);
        long inferenceAfter = ComputerPlayerRL.REAL_INFERENCE_CALLS.get();
        long inferenceDelta = inferenceAfter - inferenceBefore;
        int activationFailureDelta = ComputerPlayerRL.getRLActivationFailureCount()
                - activationFailuresBefore;
        FAILURE_STAGE.set("runtime_binding_after");
        RuntimeBinding runtimeAfter = RuntimeBinding.capture(args);

        FAILURE_STAGE.set("card_catalog_quiesce_after");
        CardRepository.instance.closeDB(false);
        FAILURE_STAGE.set("source_binding_after");
        source.verifyUnchanged(repoRoot);
        FAILURE_STAGE.set("mutable_artifact_binding_after");
        List<String> artifactInvalidReasons = artifactGuard.verifyUnchanged();

        List<String> invalidReasons = new ArrayList<>();
        invalidReasons.addAll(source.invalidReasons);
        invalidReasons.addAll(artifactGuard.configurationInvalidReasons);
        invalidReasons.addAll(artifactInvalidReasons);
        invalidReasons.addAll(runtimeBefore.configurationInvalidReasons);
        if (!runtimeBefore.sameContractAndBytes(runtimeAfter)) {
            invalidReasons.add("runtime_or_classpath_changed_during_trial");
        }
        if (observedAvailableProcessors != args.expectedAvailableProcessors
                || runtimeBefore.availableProcessors != args.expectedAvailableProcessors
                || runtimeAfter.availableProcessors != args.expectedAvailableProcessors) {
            invalidReasons.add("available_processors_contract_mismatch");
        }
        if ("strict".equals(args.bindingMode) && !artifactGuard.prebuiltCatalogPresent) {
            invalidReasons.add("prebuilt_card_catalog_required");
        }
        if ("strict".equals(args.bindingMode)
                && (observedCardContentVersion != CardRepository.instance.getContentVersionConstant()
                || observedExpansionContentVersion
                != ExpansionRepository.instance.getContentVersionConstant())) {
            invalidReasons.add("prebuilt_card_catalog_content_version_mismatch");
        }
        addPhaseInvalidReasons(invalidReasons, warmup);
        addPhaseInvalidReasons(invalidReasons, measurement);
        if (inferenceDelta != 0L) {
            invalidReasons.add("real_model_inference_calls=" + inferenceDelta);
        }
        if (activationFailureDelta != 0) {
            invalidReasons.add("activation_failures=" + activationFailureDelta);
        }
        if (measurement.naturalCompletions == 0L) {
            invalidReasons.add("no_measured_natural_completions");
        }
        boolean valid = invalidReasons.isEmpty();

        Map<String, Object> record = new LinkedHashMap<>();
        record.put("schema_version", SCHEMA_VERSION);
        record.put("record_type", "xmage_uniform_mirror_benchmark");
        record.put("created_utc", Instant.now().toString());
        record.put("trial_id", args.trialId);
        record.put("valid", valid);
        record.put("claim_scope", valid
                ? "formal_xmage_runtime_trial_candidate" : "diagnostic_only");
        record.put("formal_comparison_claim", false);
        record.put("external_paired_validator_required", true);
        record.put("external_comparison_gate", Arrays.asList(
                "same_host_cpu_topology_power_and_affinity_contract_ids",
                "same_effective_available_processors",
                "paired_XMage_and_Rust_trials",
                "AB_BA_execution_order_matrix",
                "external_validator_attestation"));
        record.put("invalid_reasons", invalidReasons);
        record.put("source_binding", source.toMap());
        record.put("mutable_artifact_guard", artifactGuard.toMap());
        record.put("card_catalog_runtime", cardCatalogRuntimeMap(
                scannerExecuted, observedCardContentVersion, observedExpansionContentVersion));
        record.put("runtime_binding", runtimeBefore.toMap(runtimeAfter));
        record.put("workload", workloadMap(args, observedDeckHash, decks));
        record.put("runtime_contract", runtimeBefore.stableRuntimeContract);
        record.put("warmup", warmup.toMap());
        record.put("measurement", measurement.toMap());
        record.put("rates", ratesMap(measurement));
        record.put("real_model_inference_calls", inferenceDelta);
        record.put("activation_failures", activationFailureDelta);
        FAILURE_STAGE.set("record_serialization");
        System.out.println(toJson(record));

        // Dirty-smoke is explicitly non-claiming and exists only to validate an
        // uncommitted driver. Strict mode must fail the process on any invalidity.
        return "strict".equals(args.bindingMode) && !valid ? 2 : 0;
    }

    private static void addPhaseInvalidReasons(List<String> out, PhaseResult phase) {
        if (phase.actorFailures > 0L) {
            out.add(phase.name + ":actor_failures=" + phase.actorFailures);
        }
        if (phase.gamesWithAnyInvalidity > 0L) {
            out.add(phase.name + ":nonnatural_games=" + phase.gamesWithAnyInvalidity);
        }
        if (phase.policyFailures > 0L) {
            out.add(phase.name + ":policy_failures=" + phase.policyFailures);
        }
        if (phase.modelFailures > 0L) {
            out.add(phase.name + ":model_failures=" + phase.modelFailures);
        }
        if (phase.explicitConcessionAttempts > 0L) {
            out.add(phase.name + ":explicit_concession_attempts="
                    + phase.explicitConcessionAttempts);
        }
        if (phase.canonicalizationFailures > 0L) {
            out.add(phase.name + ":canonicalization_failures="
                    + phase.canonicalizationFailures);
        }
    }

    private static PhaseResult runPhase(String name, long episodeBase, long requestedNanos,
                                        Args args, DeckResources decks) throws InterruptedException {
        CountDownLatch ready = new CountDownLatch(args.actors);
        CountDownLatch start = new CountDownLatch(1);
        List<ActorResult> actorResults = new ArrayList<>(args.actors);
        List<Thread> threads = new ArrayList<>(args.actors);
        for (int actor = 0; actor < args.actors; actor++) {
            ActorResult actorResult = new ActorResult(actor);
            actorResults.add(actorResult);
            final int actorIndex = actor;
            Thread thread = new Thread(() -> runActor(
                    actorResult, ready, start, episodeBase, requestedNanos, args, decks),
                    "GAME-UNIFORM-MIRROR-" + name.toUpperCase(Locale.ROOT) + "-" + actorIndex);
            thread.setDaemon(false);
            threads.add(thread);
            thread.start();
        }
        ready.await();
        long phaseStart = System.nanoTime();
        long deadline = checkedAdd(phaseStart, requestedNanos, "phase deadline");
        for (ActorResult actorResult : actorResults) {
            actorResult.phaseStartNanos = phaseStart;
            actorResult.deadlineNanos = deadline;
        }
        start.countDown();

        boolean interrupted = false;
        for (Thread thread : threads) {
            for (;;) {
                try {
                    thread.join();
                    break;
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
            throw new InterruptedException("interrupted while waiting for natural game completion");
        }
        return PhaseResult.aggregate(name, requestedNanos, phaseStart, deadline, actorResults);
    }

    private static void runActor(ActorResult out, CountDownLatch ready, CountDownLatch start,
                                 long episodeBase, long requestedNanos,
                                 Args args, DeckResources decks) {
        ready.countDown();
        try {
            start.await();
            long localEpisodeIndex = 0L;
            while (System.nanoTime() < out.deadlineNanos) {
                long episodeId = stripedEpisodeId(
                        episodeBase, localEpisodeIndex, out.actorIndex, args.actors);
                out.recordAttempt(episodeId);
                long gameStart = System.nanoTime();
                GameResult result = runNaturalGame(
                        args.baseSeed, episodeId, out.actorIndex, decks, gameStart);
                long gameFinish = result.finishNanos;
                if (gameFinish > out.deadlineNanos) {
                    result.finishedAfterDeadline = true;
                }
                out.record(result);
                localEpisodeIndex++;
            }
        } catch (Throwable t) {
            out.actorFailure = t;
        } finally {
            out.finishNanos = System.nanoTime();
        }
    }

    private static GameResult runNaturalGame(long baseSeed, long episodeId, int actorIndex,
                                             DeckResources decks, long startNanos) {
        Game game = null;
        ComputerPlayerUniformMirror p0 = null;
        ComputerPlayerUniformMirror p1 = null;
        Throwable thrown = null;
        long envSeed = SeededUniformMirrorPolicy.deriveEnvSeed(baseSeed, episodeId);
        try (RandomUtil.RandomIsolation ignored = RandomUtil.isolateThreadLocalRandom(envSeed)) {
            Deck p0Deck = decks.p0Template.copy();
            Deck p1Deck = decks.p1Template.copy();
            p0Deck.getSideboard().clear();
            p1Deck.getSideboard().clear();
            verifyDisjointCardIds(p0Deck, p1Deck, "per-game decks");
            MutableSplitMix64 shuffle = new MutableSplitMix64(envSeed);
            shuffleDeck(p0Deck, shuffle);
            shuffleDeck(p1Deck, shuffle);

            MatchOptions matchOptions = fixedMatchOptions();
            TwoPlayerMatch match = new TwoPlayerMatch(matchOptions);
            match.startGame();
            game = match.getGames().get(0);
            String suffix = "-a" + actorIndex + "-e" + episodeId;
            p0 = new ComputerPlayerUniformMirror(
                    "uniform-p0" + suffix, RangeOfInfluence.ALL, baseSeed, episodeId, "p0");
            p1 = new ComputerPlayerUniformMirror(
                    "uniform-p1" + suffix, RangeOfInfluence.ALL, baseSeed, episodeId, "p1");
            game.addPlayer(p0, p0Deck);
            match.addPlayer(p0, p0Deck);
            game.addPlayer(p1, p1Deck);
            match.addPlayer(p1, p1Deck);
            game.loadCards(p0Deck.getCards(), p0.getId());
            game.loadCards(p1Deck.getCards(), p1.getId());
            forceLibraryOrder(p0, p0Deck, game);
            forceLibraryOrder(p1, p1Deck, game);
            GameOptions gameOptions = new GameOptions();
            gameOptions.skipInitShuffling = true;
            gameOptions.rollbackTurnsAllowed = false;
            game.setGameOptions(gameOptions);
            game.setStartingPlayerId(p0.getId());
            if (!p0.getId().equals(game.getStartingPlayerId())) {
                throw new IllegalStateException("engine rejected fixed P0 starting player");
            }
            game.start(p0.getId());
        } catch (Throwable t) {
            thrown = t;
        }
        return GameResult.classify(
                startNanos, System.nanoTime(), game, p0, p1, thrown);
    }

    private static MatchOptions fixedMatchOptions() {
        MatchOptions options = new MatchOptions(
                "XMageUniformMirrorBenchmark", "TwoPlayerDuel", false);
        options.setWinsNeeded(1);
        options.setFreeMulligans(0);
        options.setCustomStartLifeEnabled(true);
        options.setCustomStartLife(20);
        options.setCustomStartHandSizeEnabled(true);
        options.setCustomStartHandSize(7);
        options.setDeckType("Constructed");
        options.setLimited(false);
        options.setRollbackTurnsAllowed(false);
        options.setSpectatorsAllowed(false);
        options.setRated(false);
        options.setMullgianType(MulliganType.LONDON);
        return options;
    }

    private static void forceLibraryOrder(Player player, Deck deck, Game game) {
        LinkedHashSet<Card> ordered = new LinkedHashSet<>();
        for (Card card : deck.getCards()) {
            if (card != null && !card.isExtraDeckCard()) {
                ordered.add(card);
            }
        }
        if (ordered.size() != 60) {
            throw new IllegalStateException("forced library must contain exactly 60 cards");
        }
        player.getLibrary().clear();
        player.getLibrary().addAll(ordered, game);
        if (player.getLibrary().size() != 60) {
            throw new IllegalStateException("engine library size mismatch after fixed ordering");
        }
    }

    private static void shuffleDeck(Deck deck, MutableSplitMix64 random) {
        List<Card> cards = new ArrayList<>(deck.getCards());
        for (int i = cards.size() - 1; i > 0; i--) {
            int j = SeededUniformMirrorPolicy.unsignedModulo(random.next(), i + 1);
            Card tmp = cards.get(i);
            cards.set(i, cards.get(j));
            cards.set(j, tmp);
        }
        deck.getCards().clear();
        deck.getCards().addAll(cards);
    }

    private static long stripedEpisodeId(long phaseBase, long localIndex,
                                         int actorIndex, int actorCount) {
        if (localIndex < 0L) {
            throw new IllegalArgumentException("negative local episode index");
        }
        long remaining = EPISODES_PER_PHASE - 1L - actorIndex;
        if (localIndex > remaining / actorCount) {
            throw new IllegalStateException("episode schedule exhausted reserved uint63 phase range");
        }
        return phaseBase + localIndex * actorCount + actorIndex;
    }

    private static Map<String, Object> workloadMap(Args args, String deckHash, DeckResources decks) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("workload_id", "rally_mirror_bo1_keep7_p0_starts/v2");
        map.put("policy_id", SeededUniformMirrorPolicy.POLICY_ID);
        map.put("seed_derivation_version", SeededUniformMirrorPolicy.SEED_DERIVATION_VERSION);
        map.put("base_seed", args.baseSeed);
        map.put("deck_relative_path", DECK_RELATIVE_PATH);
        map.put("deck_sha256", deckHash);
        map.put("deck_card_class_count", decks.cardClassCount);
        map.put("deck_card_class_manifest_sha256", decks.cardClassManifestSha256);
        map.put("maindeck_cards_per_seat", decks.mainCount);
        map.put("source_sideboard_cards_per_seat", decks.sideboardCount);
        map.put("game_sideboard_cards_per_seat", 0);
        map.put("format", "BO1");
        map.put("starting_life", 20);
        map.put("opening_hand", 7);
        map.put("mulligan", "fixed_keep_seven_consumes_zero_policy_decisions");
        map.put("starting_player", "p0_fixed_before_Game.start");
        map.put("library_shuffle", "one mutable SplitMix64(env_seed), Fisher-Yates p0 then p1");
        map.put("engine_rng", "RandomUtil thread-local java.util.Random(env_seed) for full game setup/play");
        map.put("combat_policy", "attackers one group and leaf(i)%2; blockers per attacker 35pct gate then leaf1 rank");
        map.put("noncombat_policy", "one physical group per non-forced complete engine menu; canonical ranks; no UUID keys");
        map.put("forced_choice_policy", "empty priority, jointly unaffordable Bushwhacker kicker, and all-mandatory target sets consume zero policy decisions");
        map.put("chain_lightning_copy_policy", "exact prompt/source exposes [pay_and_copy,decline] as two ranks regardless of current affordability");
        map.put("trigger_order_policy", "one n!-rank physical decision in exact Rust swap-permutation order, replayed through XMage callbacks; cap=7");
        map.put("priority_candidate_surface", "PlayerImpl.getPlayable(game,true,Zone.ALL,false), preserving every source-distinct activation, plus PASS only when at least one playable exists; empty playable auto-passes with zero policy decisions");
        map.put("canonical_tie_policy", "record and invalidate any engine-encounter tie break");
        map.put("timeout_policy", "none; external process supervision only");
        map.put("termination_policy", "natural terminal only; no turn cap, watchdog, interrupt, or forced game.end");
        map.put("warmup_episode_range", "[0,2^62)");
        map.put("measurement_episode_range", "[2^62,2^63)");
        map.put("actor_episode_schedule", "episode_base + actor_index + local_index*actor_count");
        map.put("deadline_policy", "stop launching at common deadline, finish every launched game");
        map.put("throughput_denominator", "slowest actor finish offset from common phase start");
        map.put("decision_count_source", "actual SeededUniformMirrorPolicy counters; never assigned IDs");
        return map;
    }

    private static Map<String, Object> cardCatalogRuntimeMap(
            boolean scannerExecuted, long observedCardVersion, long observedExpansionVersion) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("scanner_executed", scannerExecuted);
        map.put("scanner_policy", scannerExecuted
                ? "dirty_smoke_only_and_catalog_mutation_invalidates"
                : "strict_uses_prebuilt_read_only_catalog_without_scanner");
        map.put("observed_card_content_version", observedCardVersion < 0L
                ? null : observedCardVersion);
        map.put("expected_card_content_version",
                CardRepository.instance.getContentVersionConstant());
        map.put("observed_expansion_content_version", observedExpansionVersion < 0L
                ? null : observedExpansionVersion);
        map.put("expected_expansion_content_version",
                ExpansionRepository.instance.getContentVersionConstant());
        return map;
    }

    private static Map<String, Object> ratesMap(PhaseResult measurement) {
        double seconds = nanosToSeconds(measurement.elapsedNanos);
        Map<String, Object> rates = new LinkedHashMap<>();
        rates.put("denominator_seconds", seconds);
        rates.put("natural_games_per_second", measurement.naturalCompletions / seconds);
        rates.put("turns_per_second", measurement.turns / seconds);
        rates.put("physical_decisions_per_second", measurement.physicalDecisions / seconds);
        rates.put("policy_action_selections_per_second", measurement.policyActionSelections / seconds);
        rates.put("policy_leaf_evaluations_per_second", measurement.policyLeafEvaluations / seconds);
        return rates;
    }

    private static Map<String, Object> runtimeContractMap(
            Args args, int observedProcessors) throws Exception {
        RuntimeMXBean runtime = ManagementFactory.getRuntimeMXBean();
        MemoryMXBean memory = ManagementFactory.getMemoryMXBean();
        java.lang.management.OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("actors", args.actors);
        map.put("actor_threads", args.actors);
        map.put("thread_model", "one synchronous GAME-* thread per actor");
        map.put("affinity_contract_id", args.affinityContractId);
        map.put("cpu_contract_id", args.cpuContractId);
        map.put("topology_contract_id", args.topologyContractId);
        map.put("host_contract_id", args.hostContractId);
        map.put("power_contract_id", args.powerContractId);
        map.put("expected_available_processors", args.expectedAvailableProcessors);
        map.put("observed_available_processors", observedProcessors);
        map.put("available_processors_match", observedProcessors == args.expectedAvailableProcessors);
        map.put("hardware_contract_ids_are_external_attestations", true);
        map.put("hardware_or_affinity_actually_verified_by_jvm", false);
        map.put("jvm_name", privacySafeLabel(runtime.getVmName(), "jvm_name"));
        map.put("jvm_vendor", privacySafeLabel(runtime.getVmVendor(), "jvm_vendor"));
        map.put("jvm_version", privacySafeVersion(runtime.getVmVersion(), "jvm_version"));
        map.put("java_version", privacySafeVersion(System.getProperty("java.version"), "java_version"));
        RuntimeConfiguration configuration = RuntimeConfiguration.capture(runtime.getInputArguments());
        map.put("jvm_argument_manifest_sha256", configuration.jvmArgumentManifestSha256);
        map.put("jvm_arguments", configuration.jvmArguments);
        map.put("system_property_count", configuration.systemPropertyCount);
        map.put("system_property_manifest_sha256", configuration.systemPropertyManifestSha256);
        map.put("relevant_environment_manifest_sha256",
                configuration.relevantEnvironmentManifestSha256);
        map.put("relevant_environment", configuration.relevantEnvironment);
        map.put("configuration_invalid_reasons", configuration.invalidReasons);
        map.put("os_name", privacySafeLabel(os.getName(), "os_name"));
        map.put("os_arch", privacySafeLabel(os.getArch(), "os_arch"));
        map.put("heap_initial_mib", bytesToMiB(memory.getHeapMemoryUsage().getInit()));
        map.put("heap_max_mib", bytesToMiB(memory.getHeapMemoryUsage().getMax()));
        List<String> collectors = new ArrayList<>();
        for (GarbageCollectorMXBean bean : ManagementFactory.getGarbageCollectorMXBeans()) {
            collectors.add(privacySafeLabel(bean.getName(), "garbage_collector"));
        }
        map.put("garbage_collectors", collectors);
        map.put("file_encoding", privacySafeLabel(System.getProperty("file.encoding"), "file_encoding"));
        return map;
    }

    private static long bytesToMiB(long bytes) {
        return bytes < 0L ? -1L : bytes / (1024L * 1024L);
    }

    private static String privacySafeLabel(String value, String field) {
        if (value == null || value.isEmpty()
                || !value.matches("[A-Za-z0-9 ._()+-]{1,128}")) {
            throw new IllegalStateException(field + " is not a privacy-safe normalized label");
        }
        return value;
    }

    private static String privacySafeVersion(String value, String field) {
        if (value == null || value.isEmpty()
                || !value.matches("[A-Za-z0-9 ._()+,;:+-]{1,160}")) {
            throw new IllegalStateException(field + " is not a privacy-safe normalized version");
        }
        return value;
    }


    private static void quietLogging() {
        LogManager.getRootLogger().setLevel(Level.OFF);
        Enumeration<?> current = LogManager.getCurrentLoggers();
        while (current.hasMoreElements()) {
            Object next = current.nextElement();
            if (next instanceof Logger) {
                ((Logger) next).setLevel(Level.OFF);
            }
        }
    }

    /** ComputerPlayerRL has an unconditional diagnostic println in static init. */
    private static long initializeComputerPlayerRlSilently() {
        PrintStream original = System.out;
        PrintStream sink = new PrintStream(new OutputStream() {
            @Override
            public void write(int ignored) {
            }
        });
        try {
            System.setOut(sink);
            return ComputerPlayerRL.REAL_INFERENCE_CALLS.get();
        } finally {
            System.setOut(original);
            sink.close();
        }
    }

    private static void runSelfTest() throws Exception {
        SeededUniformMirrorPolicy.runFixedGoldenSelfTest();
        initializeComputerPlayerRlSilently();
        ComputerPlayerUniformMirror.runUniformMirrorSelfTest();
        assertActivationFailureSourceContract();
        if (stripedEpisodeId(0L, 0L, 0, 4) != 0L
                || stripedEpisodeId(0L, 3L, 2, 4) != 14L
                || stripedEpisodeId(MEASURE_EPISODE_BASE, 2L, 3, 4)
                != MEASURE_EPISODE_BASE + 11L) {
            throw new IllegalStateException("striped seed schedule self-test failed");
        }
        boolean rejected = false;
        try {
            stripedEpisodeId(MEASURE_EPISODE_BASE, EPISODES_PER_PHASE, 0, 1);
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        if (!rejected) {
            throw new IllegalStateException("seed schedule overflow did not fail closed");
        }
        MutableSplitMix64 a = new MutableSplitMix64(71_501L);
        MutableSplitMix64 b = new MutableSplitMix64(71_501L);
        for (int i = 0; i < 32; i++) {
            if (a.next() != b.next()) {
                throw new IllegalStateException("mutable SplitMix64 identity failed");
            }
        }
        SeededUniformMirrorPolicy original = new SeededUniformMirrorPolicy(71_501L, 9L, "p0");
        SeededUniformMirrorPolicy copied = original.copy();
        copied.chooseAttackers(3);
        if (original.getPhysicalDecisionCount() != 0L
                || copied.getPhysicalDecisionCount() != 1L
                || copied.getPolicyActionSelections() != 3L
                || copied.getPolicyLeafEvaluations() != 3L) {
            throw new IllegalStateException("policy copy/counter isolation failed");
        }
        expectFailure(() -> Args.parse(new String[]{"--actors", "2"}), "strict CLI arity");
        Args privacyArgs = Args.parse(selfTestArgs("host_contract"));
        expectFailure(() -> Args.parse(selfTestArgs("C:\\Users\\private\\host")),
                "privacy-safe contract identifiers");
        String privatePath = "C:\\Users\\private\\secret.txt";
        String fingerprint = toJson(throwableFingerprint(
                new IllegalStateException("failed at " + privatePath)));
        if (fingerprint.contains("private") || fingerprint.contains("secret")
                || fingerprint.contains("C:\\")) {
            throw new IllegalStateException("throwable fingerprint leaked an exception message");
        }
        String runtimeJson = toJson(runtimeContractMap(
                privacyArgs, Runtime.getRuntime().availableProcessors()));
        for (String forbidden : Arrays.asList(
                "java_home", "java_class_path", "jvm_input_arguments",
                "processor_identifier", "repo_root")) {
            if (runtimeJson.contains(forbidden)) {
                throw new IllegalStateException("runtime privacy self-test found forbidden key");
            }
        }
        if (RuntimeConfiguration.classifyJvmArgument("-Xint").allowed
                || RuntimeConfiguration.classifyJvmArgument("-javaagent:C:\\private.jar").allowed
                || RuntimeConfiguration.classifyJvmArgument("-Dunknown.setting=true").allowed
                || !RuntimeConfiguration.classifyJvmArgument("-Xmx2g").allowed
                || !RuntimeConfiguration.classifyJvmArgument("-XX:+UseG1GC").allowed
                || RuntimeConfiguration.classifyEnvironment("RL_ACTIVATION_DIAG", "1").allowed
                || !RuntimeConfiguration.classifyEnvironment("RL_ACTIVATION_DIAG", "0").allowed
                || RuntimeConfiguration.classifyEnvironment("RL_UNKNOWN_SETTING", "1").allowed) {
            throw new IllegalStateException("runtime configuration allowlist self-test failed");
        }
        assertSourceAuditedEnvironmentInventory();
        expectFailure(() -> {
            Map<String, Object> invalidJson = new LinkedHashMap<>();
            invalidJson.put("value", Double.NaN);
            toJson(invalidJson);
        }, "non-finite JSON");
        expectFailure(() -> checkedAdd(Long.MAX_VALUE, 1L, "self-test"), "deadline overflow");
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("schema_version", SCHEMA_VERSION);
        result.put("record_type", "self_test");
        result.put("valid", true);
        result.put("checks", Arrays.asList(
                "authoritative_seed_goldens", "uint63_phase_partition",
                "striped_actor_schedule", "schedule_overflow_fail_closed",
                "mutable_splitmix_identity", "policy_copy_counter_isolation",
                "uniform_player_trigger_forced_target_chain_and_source_fixtures",
                "activation_failure_diagnostics_source_order",
                "strict_cli_fail_closed", "nonfinite_json_fail_closed",
                "deadline_overflow_fail_closed", "privacy_contract_id_fail_closed",
                "throwable_message_redaction", "runtime_output_allowlist",
                "complete_runtime_configuration_allowlist",
                "source_audited_environment_inventory"));
        System.out.println(toJson(result));
    }

    private static void assertSourceAuditedEnvironmentInventory() throws Exception {
        List<String> mandatory = Arrays.asList(
                "EVAL_RANDOM_UTIL_DIRECT_TRACE_JSON",
                "EVAL_RANDOM_UTIL_DIRECT_TRACE_FILE",
                "EVAL_RANDOM_UTIL_DIRECT_TRACE",
                "EVAL_RANDOM_UTIL_WRAPPER_TRACE_JSON",
                "EVAL_RANDOM_UTIL_WRAPPER_TRACE_FILE",
                "EVAL_RANDOM_UTIL_WRAPPER_TRACE",
                "MTG_AI_LOG_LEVEL", "MTG_AI_LOG_FILE");
        if (!RuntimeConfiguration.SOURCE_AUDITED_FORBIDDEN_ENV.containsAll(mandatory)) {
            throw new IllegalStateException(
                    "source-audited environment inventory lost a mandatory entry");
        }
        Map<String, String> synthetic = new LinkedHashMap<>();
        for (String name : RuntimeConfiguration.SOURCE_AUDITED_FORBIDDEN_ENV) {
            synthetic.put(name, name.contains("FILE") || name.contains("PATH")
                    || name.endsWith("_DIR")
                    ? "C:\\uniform-self-test\\forbidden" : "1");
        }
        synthetic.put("EVAL_UNAUDITED_FUTURE_SWITCH", "1");
        EnvironmentInventory inventory = RuntimeConfiguration.captureEnvironment(synthetic);
        Map<String, Map<String, Object>> records = new LinkedHashMap<>();
        for (Map<String, Object> record : inventory.records) {
            records.put(String.valueOf(record.get("name")), record);
        }
        for (String name : RuntimeConfiguration.SOURCE_AUDITED_FORBIDDEN_ENV) {
            Map<String, Object> record = records.get(name);
            if (record == null || !Boolean.TRUE.equals(record.get("present"))
                    || !Boolean.FALSE.equals(record.get("allowed"))
                    || record.get("value_sha256") == null
                    || !"source_audited_benchmark_environment".equals(
                    record.get("category"))) {
                throw new IllegalStateException(
                        "source-audited environment was not rejected and digested: " + name);
            }
        }
        Map<String, Object> failSafe = records.get("EVAL_UNAUDITED_FUTURE_SWITCH");
        if (failSafe == null || !Boolean.FALSE.equals(failSafe.get("allowed"))
                || inventory.invalidReasons.isEmpty()
                || inventory.manifestSha256 == null
                || inventory.manifestSha256.length() != 64
                || toJson(inventory.records).contains("uniform-self-test")) {
            throw new IllegalStateException(
                    "fail-safe environment rejection or sanitization self-test failed");
        }
    }

    private static void assertActivationFailureSourceContract() throws Exception {
        Path workingDirectory = Paths.get(System.getProperty("user.dir")).toRealPath();
        Path source = workingDirectory.resolve(
                "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/ComputerPlayerRL.java")
                .normalize().toRealPath();
        if (!source.startsWith(workingDirectory)) {
            throw new IllegalStateException("activation source contract escaped working directory");
        }
        String text = new String(Files.readAllBytes(source), StandardCharsets.UTF_8);
        int method = text.indexOf("protected void act(Game game, ActivatedAbility ability)");
        int hook = text.indexOf(
                "boolean failClosedActivation = failClosedOnActivationFailure();", method);
        int diagnostics = text.indexOf("boolean activationDiagnostics = ACTIVATION_DIAG", hook);
        int activationCall = text.indexOf(
                "activationResult = super.activateAbility(freshAbility, game);", diagnostics);
        int catchBlock = text.indexOf("} catch (Exception e) {", activationCall);
        int diagnosticGate = text.indexOf(
                "if (activationFailureDiagnosticsAllowed(failClosedActivation))", catchBlock);
        int stackTrace = text.indexOf("e.printStackTrace();", catchBlock);
        int finallyBlock = text.indexOf("} finally {", catchBlock);
        int failCloseGuard = text.indexOf(
                "if (!activationResult && failClosedActivation)", finallyBlock);
        int postFailureDiagnostics = text.indexOf(
                "if (activationDiagnostics && !activationResult)", failCloseGuard);
        if (!(method >= 0 && method < hook && hook < diagnostics
                && diagnostics < activationCall && activationCall < catchBlock
                && catchBlock < diagnosticGate && diagnosticGate < stackTrace
                && stackTrace < finallyBlock && finallyBlock < failCloseGuard
                && failCloseGuard < postFailureDiagnostics)) {
            throw new IllegalStateException(
                    "activation failure diagnostics no longer follow the fail-close source contract");
        }
        String catchPrefix = text.substring(catchBlock, diagnosticGate);
        if (catchPrefix.contains("threadLocalLogger")
                || catchPrefix.contains("printStackTrace")
                || catchPrefix.contains("System.err")) {
            throw new IllegalStateException(
                    "activation exception diagnostic precedes the fail-close gate");
        }
    }

    private static String[] selfTestArgs(String hostContractId) {
        return new String[]{
                "--repo-root", "C:\\benchmark-root",
                "--expected-commit", "0000000000000000000000000000000000000000",
                "--actors", "1",
                "--base-seed", "71501",
                "--warmup-seconds", "1",
                "--measure-seconds", "1",
                "--trial-id", "self_test",
                "--binding-mode", "dirty-smoke",
                "--affinity-contract-id", "affinity_contract",
                "--expected-available-processors", "1",
                "--cpu-contract-id", "cpu_contract",
                "--topology-contract-id", "topology_contract",
                "--host-contract-id", hostContractId,
                "--power-contract-id", "power_contract"
        };
    }

    private static void expectFailure(Runnable action, String label) {
        try {
            action.run();
        } catch (RuntimeException expected) {
            return;
        }
        throw new IllegalStateException(label + " did not fail closed");
    }

    private static long checkedAdd(long left, long right, String label) {
        if (right < 0L || left > Long.MAX_VALUE - right) {
            throw new IllegalArgumentException(label + " overflows nanoTime domain");
        }
        return left + right;
    }

    private static double nanosToSeconds(long nanos) {
        if (nanos <= 0L) {
            throw new IllegalStateException("elapsed nanoseconds must be positive");
        }
        return nanos / 1_000_000_000.0;
    }

    private static void verifyDisjointCardIds(Deck left, Deck right, String label) {
        Set<UUID> ids = new HashSet<>();
        addIds(ids, left.getCards(), label + " left main");
        addIds(ids, left.getSideboard(), label + " left sideboard");
        int leftCount = ids.size();
        addIds(ids, right.getCards(), label + " right main");
        addIds(ids, right.getSideboard(), label + " right sideboard");
        int expected = leftCount + right.getCards().size() + right.getSideboard().size();
        if (ids.size() != expected) {
            throw new IllegalStateException(label + " share card UUIDs between seats");
        }
    }

    private static void addIds(Set<UUID> ids, Collection<Card> cards, String label) {
        for (Card card : cards) {
            if (card == null || card.getId() == null || !ids.add(card.getId())) {
                throw new IllegalStateException(label + " contains null or duplicate card UUID");
            }
        }
    }

    private static String sha256(Path path) throws Exception {
        try (InputStream in = Files.newInputStream(path)) {
            return sha256(in);
        }
    }

    private static String sha256ClassResource(Class<?> cls) throws Exception {
        String resource = "/" + cls.getName().replace('.', '/') + ".class";
        try (InputStream in = cls.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IllegalStateException("resolved deck card class resource is absent");
            }
            return sha256(in);
        }
    }

    private static String sha256(InputStream in) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[64 * 1024];
        int read;
        while ((read = in.read(buffer)) >= 0) {
            if (read > 0) {
                digest.update(buffer, 0, read);
            }
        }
        return hex(digest.digest());
    }

    private static String hex(byte[] bytes) {
        StringBuilder out = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            out.append(String.format(Locale.ROOT, "%02x", b & 0xff));
        }
        return out.toString();
    }

    private static String digestUtf8(String value) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        digest.update(value.getBytes(StandardCharsets.UTF_8));
        return hex(digest.digest());
    }

    private static String digestStrings(List<String> values) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        for (String value : values) {
            updateFramed(digest, value);
        }
        return hex(digest.digest());
    }

    private static void updateFramed(MessageDigest digest, String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        digest.update((byte) ((bytes.length >>> 24) & 0xff));
        digest.update((byte) ((bytes.length >>> 16) & 0xff));
        digest.update((byte) ((bytes.length >>> 8) & 0xff));
        digest.update((byte) (bytes.length & 0xff));
        digest.update(bytes);
    }

    private static Map<String, Object> throwableFingerprint(Throwable t) {
        Map<String, Object> map = new LinkedHashMap<>();
        String label = throwableClassLabel(t);
        map.put("class_label", label);
        String message = t.getMessage() == null ? "" : t.getMessage();
        try {
            map.put("message_sha256", digestUtf8(message));
        } catch (Exception impossible) {
            throw new IllegalStateException("SHA-256 unavailable", impossible);
        }
        return map;
    }

    private static String throwableClassLabel(Throwable t) {
        String label = t.getClass().getSimpleName();
        if (label == null || label.isEmpty() || !label.matches("[A-Za-z0-9_$]{1,128}")) {
            label = "Throwable";
        }
        return label;
    }

    private static String toJson(Object value) {
        StringBuilder out = new StringBuilder(16 * 1024);
        appendJson(out, value);
        return out.toString();
    }

    @SuppressWarnings("unchecked")
    private static void appendJson(StringBuilder out, Object value) {
        if (value == null) {
            out.append("null");
        } else if (value instanceof String) {
            appendJsonString(out, (String) value);
        } else if (value instanceof Boolean || value instanceof Byte
                || value instanceof Short || value instanceof Integer || value instanceof Long) {
            out.append(value);
        } else if (value instanceof Float || value instanceof Double) {
            double number = ((Number) value).doubleValue();
            if (!Double.isFinite(number)) {
                throw new IllegalStateException("non-finite JSON number");
            }
            out.append(Double.toString(number));
        } else if (value instanceof Map) {
            out.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
                if (!(entry.getKey() instanceof String)) {
                    throw new IllegalStateException("JSON object key is not a string");
                }
                if (!first) {
                    out.append(',');
                }
                first = false;
                appendJsonString(out, (String) entry.getKey());
                out.append(':');
                appendJson(out, entry.getValue());
            }
            out.append('}');
        } else if (value instanceof Iterable) {
            out.append('[');
            boolean first = true;
            for (Object item : (Iterable<Object>) value) {
                if (!first) {
                    out.append(',');
                }
                first = false;
                appendJson(out, item);
            }
            out.append(']');
        } else {
            throw new IllegalStateException("unsupported JSON type " + value.getClass().getName());
        }
    }

    private static void appendJsonString(StringBuilder out, String value) {
        out.append('"');
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            switch (ch) {
                case '"': out.append("\\\""); break;
                case '\\': out.append("\\\\"); break;
                case '\b': out.append("\\b"); break;
                case '\f': out.append("\\f"); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                default:
                    if (ch < 0x20) {
                        out.append(String.format(Locale.ROOT, "\\u%04x", (int) ch));
                    } else {
                        out.append(ch);
                    }
            }
        }
        out.append('"');
    }

    private static final class Args {
        final Path repoRoot;
        final String expectedCommit;
        final int actors;
        final long baseSeed;
        final long warmupNanos;
        final long measureNanos;
        final String trialId;
        final String bindingMode;
        final String affinityContractId;
        final int expectedAvailableProcessors;
        final String cpuContractId;
        final String topologyContractId;
        final String hostContractId;
        final String powerContractId;

        private Args(Path repoRoot, String expectedCommit, int actors, long baseSeed,
                     long warmupNanos, long measureNanos, String trialId, String bindingMode,
                     String affinityContractId, int expectedAvailableProcessors,
                     String cpuContractId, String topologyContractId,
                     String hostContractId, String powerContractId) {
            this.repoRoot = repoRoot;
            this.expectedCommit = expectedCommit;
            this.actors = actors;
            this.baseSeed = baseSeed;
            this.warmupNanos = warmupNanos;
            this.measureNanos = measureNanos;
            this.trialId = trialId;
            this.bindingMode = bindingMode;
            this.affinityContractId = affinityContractId;
            this.expectedAvailableProcessors = expectedAvailableProcessors;
            this.cpuContractId = cpuContractId;
            this.topologyContractId = topologyContractId;
            this.hostContractId = hostContractId;
            this.powerContractId = powerContractId;
        }

        static Args parse(String[] raw) {
            if (raw.length != 28) {
                throw new IllegalArgumentException("benchmark requires exactly fourteen explicit --key value pairs");
            }
            Map<String, String> values = new LinkedHashMap<>();
            for (int i = 0; i < raw.length; i += 2) {
                String key = raw[i];
                if (!key.startsWith("--") || key.length() < 3) {
                    throw new IllegalArgumentException("expected --key at argument " + i);
                }
                if (values.put(key, raw[i + 1]) != null) {
                    throw new IllegalArgumentException("duplicate argument " + key);
                }
            }
            Set<String> expected = new HashSet<>(Arrays.asList(
                    "--repo-root", "--expected-commit", "--actors", "--base-seed",
                    "--warmup-seconds", "--measure-seconds", "--trial-id", "--binding-mode",
                    "--affinity-contract-id", "--expected-available-processors",
                    "--cpu-contract-id", "--topology-contract-id",
                    "--host-contract-id", "--power-contract-id"));
            if (!values.keySet().equals(expected)) {
                Set<String> missing = new HashSet<>(expected);
                missing.removeAll(values.keySet());
                Set<String> unknown = new HashSet<>(values.keySet());
                unknown.removeAll(expected);
                throw new IllegalArgumentException("argument set mismatch; missing=" + missing + " unknown=" + unknown);
            }
            Path root = Paths.get(values.get("--repo-root"));
            if (!root.isAbsolute()) {
                throw new IllegalArgumentException("--repo-root must be absolute");
            }
            String expectedCommit = values.get("--expected-commit").toLowerCase(Locale.ROOT);
            if (!COMMIT_PATTERN.matcher(expectedCommit).matches()) {
                throw new IllegalArgumentException("--expected-commit must be exactly 40 lowercase hex digits");
            }
            int actors = parseInt(values.get("--actors"), "--actors");
            if (!ACTOR_COUNTS.contains(actors)) {
                throw new IllegalArgumentException("--actors must be exactly one of 1,4,8,16");
            }
            long baseSeed = parseLong(values.get("--base-seed"), "--base-seed");
            if (baseSeed < 0L) {
                throw new IllegalArgumentException("--base-seed must be in uint63 domain");
            }
            long warmup = parseSeconds(values.get("--warmup-seconds"), "--warmup-seconds");
            long measure = parseSeconds(values.get("--measure-seconds"), "--measure-seconds");
            String trialId = values.get("--trial-id");
            if (!TRIAL_PATTERN.matcher(trialId).matches()) {
                throw new IllegalArgumentException("--trial-id must match " + TRIAL_PATTERN.pattern());
            }
            String bindingMode = values.get("--binding-mode");
            if (!"strict".equals(bindingMode) && !"dirty-smoke".equals(bindingMode)) {
                throw new IllegalArgumentException("--binding-mode must be strict or dirty-smoke");
            }
            String affinityContractId = parseContractId(
                    values.get("--affinity-contract-id"), "--affinity-contract-id");
            int expectedAvailableProcessors = parseInt(
                    values.get("--expected-available-processors"), "--expected-available-processors");
            if (expectedAvailableProcessors <= 0) {
                throw new IllegalArgumentException("--expected-available-processors must be positive");
            }
            String cpuContractId = parseContractId(values.get("--cpu-contract-id"), "--cpu-contract-id");
            String topologyContractId = parseContractId(
                    values.get("--topology-contract-id"), "--topology-contract-id");
            String hostContractId = parseContractId(values.get("--host-contract-id"), "--host-contract-id");
            String powerContractId = parseContractId(values.get("--power-contract-id"), "--power-contract-id");
            return new Args(root, expectedCommit, actors, baseSeed, warmup, measure,
                    trialId, bindingMode, affinityContractId, expectedAvailableProcessors,
                    cpuContractId, topologyContractId, hostContractId, powerContractId);
        }

        private static String parseContractId(String value, String name) {
            if (value == null || !TRIAL_PATTERN.matcher(value).matches()) {
                throw new IllegalArgumentException(name + " must be a privacy-safe opaque identifier");
            }
            return value;
        }

        private static int parseInt(String value, String name) {
            if (!value.matches("(?:0|[1-9][0-9]*)")) {
                throw new IllegalArgumentException(name + " must be canonical nonnegative decimal");
            }
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(name + " is outside int range", e);
            }
        }

        private static long parseLong(String value, String name) {
            if (!value.matches("(?:0|[1-9][0-9]*)")) {
                throw new IllegalArgumentException(name + " must be canonical nonnegative decimal");
            }
            try {
                return Long.parseLong(value);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(name + " is outside uint63 range", e);
            }
        }

        private static long parseSeconds(String value, String name) {
            if (!SECONDS_PATTERN.matcher(value).matches()) {
                throw new IllegalArgumentException(name + " must be positive decimal seconds with at most 9 fractional digits");
            }
            try {
                long nanos = new BigDecimal(value).movePointRight(9).longValueExact();
                if (nanos <= 0L) {
                    throw new IllegalArgumentException(name + " must be positive");
                }
                return nanos;
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException(name + " is outside exact nanosecond range", e);
            }
        }
    }

    private static final class SourceBinding {
        final String expectedCommit;
        final String actualCommit;
        final String bindingMode;
        final List<String> statusLines;
        final List<String> invalidReasons;
        final Map<String, String> selectedSourceHashes;

        private SourceBinding(String expectedCommit, String actualCommit, String bindingMode,
                              List<String> statusLines, List<String> invalidReasons,
                              Map<String, String> selectedSourceHashes) {
            this.expectedCommit = expectedCommit;
            this.actualCommit = actualCommit;
            this.bindingMode = bindingMode;
            this.statusLines = statusLines;
            this.invalidReasons = invalidReasons;
            this.selectedSourceHashes = selectedSourceHashes;
        }

        static SourceBinding inspect(Path root, String expectedCommit, String bindingMode) throws Exception {
            String actual = runGit(root, "rev-parse", "HEAD").trim().toLowerCase(Locale.ROOT);
            if (!COMMIT_PATTERN.matcher(actual).matches()) {
                throw new IllegalStateException("git rev-parse did not return a commit hash");
            }
            if (!actual.equals(expectedCommit)) {
                throw new IllegalStateException("expected commit " + expectedCommit + " but found " + actual);
            }
            String status = runGit(root, "status", "--porcelain=v1", "--untracked-files=all");
            List<String> lines = statusLines(status);
            List<String> invalid = new ArrayList<>();
            if ("strict".equals(bindingMode) && !lines.isEmpty()) {
                invalid.add("source_tree_not_clean");
            } else if ("dirty-smoke".equals(bindingMode)) {
                invalid.add("dirty_smoke_binding_mode_is_nonclaiming");
            }
            return new SourceBinding(expectedCommit, actual, bindingMode, lines, invalid,
                    selectedSourceHashes(root));
        }

        void verifyUnchanged(Path root) throws Exception {
            String finalCommit = runGit(root, "rev-parse", "HEAD").trim().toLowerCase(Locale.ROOT);
            if (!actualCommit.equals(finalCommit)) {
                addUnique(invalidReasons, "source_commit_changed_during_trial");
            }
            List<String> finalStatus = statusLines(runGit(
                    root, "status", "--porcelain=v1", "--untracked-files=all"));
            if (!statusLines.equals(finalStatus)) {
                addUnique(invalidReasons, "source_status_changed_during_trial");
            }
            if (!selectedSourceHashes.equals(selectedSourceHashes(root))) {
                addUnique(invalidReasons, "selected_source_bytes_changed_during_trial");
            }
        }

        Map<String, Object> toMap() throws Exception {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("expected_commit", expectedCommit);
            map.put("actual_commit", actualCommit);
            map.put("binding_mode", bindingMode);
            map.put("working_tree_clean", statusLines.isEmpty());
            map.put("git_status_entry_count", statusLines.size());
            map.put("git_status_sha256", digestStrings(statusLines));
            map.put("selected_source_sha256", selectedSourceHashes);
            map.put("verified_unchanged_through_trial_end", !invalidReasons.contains("source_commit_changed_during_trial")
                    && !invalidReasons.contains("source_status_changed_during_trial")
                    && !invalidReasons.contains("selected_source_bytes_changed_during_trial"));
            return map;
        }

        private static Map<String, String> selectedSourceHashes(Path root) throws Exception {
            Map<String, String> files = new TreeMap<>();
            for (String relative : Arrays.asList(
                    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/XMageUniformMirrorBenchmark.java",
                    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/SeededUniformMirrorPolicy.java",
                    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/ComputerPlayerUniformMirror.java",
                    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/ComputerPlayerRL.java",
                    DECK_RELATIVE_PATH,
                    "Mage/src/main/java/mage/game/GameImpl.java",
                    "Mage/src/main/java/mage/players/PlayerImpl.java",
                    "Mage/src/main/java/mage/util/RandomUtil.java")) {
                files.put(relative, sha256(root.resolve(relative)));
            }
            return files;
        }

        private static List<String> statusLines(String status) {
            List<String> lines = new ArrayList<>();
            for (String line : status.split("\\R")) {
                if (!line.trim().isEmpty()) {
                    lines.add(line);
                }
            }
            return lines;
        }

        private static String runGit(Path root, String... args) throws Exception {
            List<String> command = new ArrayList<>();
            command.add("git");
            command.add("-C");
            command.add(root.toString());
            command.addAll(Arrays.asList(args));
            Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            Thread reader = new Thread(() -> {
                try (InputStream in = process.getInputStream()) {
                    byte[] buffer = new byte[8192];
                    int count;
                    while ((count = in.read(buffer)) >= 0) {
                        if (count > 0) {
                            bytes.write(buffer, 0, count);
                        }
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }, "XMAGE-BENCH-GIT-READER");
            reader.setDaemon(true);
            reader.start();
            if (!process.waitFor(30L, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new IllegalStateException("git command exceeded 30 seconds");
            }
            reader.join();
            String output = new String(bytes.toByteArray(), StandardCharsets.UTF_8);
            if (process.exitValue() != 0) {
                throw new IllegalStateException("git command failed: " + output.trim());
            }
            return output;
        }
    }

    private static final class RuntimeConfiguration {
        private static final Set<String> ALLOWED_XX_OPTIONS =
                Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
                        "UseG1GC", "UseParallelGC", "UseSerialGC", "UseZGC",
                        "UseShenandoahGC", "AlwaysPreTouch", "ActiveProcessorCount",
                        "MaxRAMPercentage", "InitialRAMPercentage", "MinRAMPercentage",
                        "ParallelGCThreads", "ConcGCThreads", "CICompilerCount",
                        "TieredCompilation", "UseNUMA", "UseLargePages",
                        "UseCompressedOops")));
        private static final Set<String> SOURCE_AUDITED_FORBIDDEN_ENV =
                Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
                        // RandomUtil and replay-boundary tracing/isolation.
                        "EVAL_RANDOM_UTIL_DIRECT_TRACE_JSON",
                        "EVAL_RANDOM_UTIL_DIRECT_TRACE_FILE",
                        "EVAL_RANDOM_UTIL_DIRECT_TRACE",
                        "EVAL_RANDOM_UTIL_WRAPPER_TRACE_JSON",
                        "EVAL_RANDOM_UTIL_WRAPPER_TRACE_FILE",
                        "EVAL_RANDOM_UTIL_WRAPPER_TRACE",
                        "EVAL_AGENT_SEARCH_TRACE_FILE",
                        "EVAL_AGENT_SEARCH_TRACE_JSON",
                        "EVAL_AGENT_ACTUAL_SEARCH_TRACE_JSON",
                        "EVAL_SOURCE_ACTUAL_SEARCH_TRACE_JSON",
                        "EVAL_SUPPRESS_ENGINE_BOUNDARY_SEARCH_TRACE_FOR_ACTUAL",
                        "EVAL_REPLAY_METADATA",
                        "EVAL_REPLAY_DECISION_LOG",
                        "EVAL_REPLAY_PREGAME_DECISION_LOG",
                        "EVAL_REPLAY_LIBRARY_COPY_DETERMINISM",
                        "EVAL_REPLAY_SIMULATED_PLAYER_RNG_ISOLATION",
                        "EVAL_REPLAY_VALIDATION_RNG_ISOLATION",
                        "EVAL_SOURCE_SIMULATED_PLAYER_RNG_ISOLATION",
                        "EVAL_REPLAY_TOKEN_METADATA_RNG_ISOLATION",
                        "EVAL_REPLAY_TOKEN_METADATA_RNG_TRACE",

                        // Inherited AI behavior and search controls.
                        "AI_MAX_THREADS_FOR_SIMULATIONS",
                        "AI_DETERMINISTIC_TIEBREAKS", "AI_DETERMINISTIC_SEARCH",
                        "AI_DETERMINISTIC_ROOT_TRACE", "AI_DETERMINISTIC_MAX_NODES",
                        "AI_VERBOSE_SIM", "TORCH_DETERMINISTIC_EVAL",
                        "ISMCTS_ENABLE", "ISMCTS_ARCHETYPE_DECK_LIST",
                        "ISMCTS_RANDOM_ROLLOUT_ROOT", "MCTS_TRAINING_ENABLE",
                        "MCTS_TRAINING_FORCE_ACTION", "MCTS_ROOT_INCLUDE_PASS",
                        "MCTS_SELECTIVE_ALLOW_ANY", "MCTS_SELECTIVE_ENABLE",
                        "MCTS_SELECTIVE_KEYWORDS", "MULTI_PLY_MCTS",
                        "ROUTER_ENABLE", "SEARCH_OP_APPLY_OVERRIDE",
                        "SEARCH_OP_ARBITER_CAST_FILTER", "SEARCH_OP_CAPTURE_FILE",
                        "SEARCH_OP_ENABLE", "SEARCH_OP_EVAL_ENABLE",
                        "SEARCH_OP_GENERIC_BRANCH_ORDER", "SEARCH_OP_LOG",
                        "SEARCH_OP_MODEL_GUIDED", "SIL_GRAVEYARD_GATED",
                        "SPY_FINISH_GATE", "PAUSE_ON_ACTIVATION_FAILURE",

                        // Mulligan, draft, and diagnostic output controls reached
                        // while ComputerPlayerRL and RLLogPaths initialize.
                        "CANDIDATE_EXPLOSIONS_LOG_FILE",
                        "MULLIGAN_DECISION_LOG", "MULLIGAN_FREEZE_TRAINING",
                        "MULLIGAN_HARD_APPLY_DURING_TRAINING",
                        "MULLIGAN_HARD_FORCE_KEEP_AFTER_MAX",
                        "MULLIGAN_HARD_FORCE_KEEP_GOOD", "MULLIGAN_HARD_LOG",
                        "MULLIGAN_HARD_OVERRIDES_ENABLE",
                        "MULLIGAN_HARD_PSEUDO_LANDS",
                        "MULLIGAN_HARD_USE_EFFECTIVE_LANDS",
                        "MULLIGAN_TRAINING_LOG_FILE", "MULLIGAN_TRACE_JSONL_FILE",
                        "DRAFT_MIN_LANDS_ENABLE", "DRAFT_MIN_LANDS_START",
                        "DRAFT_MIN_LANDS_END", "DRAFT_MIN_LANDS_DECAY_EPISODES",

                        // Trainer/game-logger behavior reachable through inherited
                        // diagnostic hooks and static path initialization.
                        "ACTOR_LEARNER_ASYNC", "ACTOR_LEARNER_BACKPRESSURE",
                        "ADAPTIVE_CURRICULUM", "BENCHMARK_LIVE_REPORT_PATH",
                        "BOT_MIX", "DECK_LIST_FILE", "DECK_SAMPLE_FLOOR", "DECKS_DIR",
                        "EVAL_AT_START", "EVAL_GAME_LOGGING", "EVAL_OPPONENT_DECK",
                        "EVAL_OPPONENT_ON_PLAY", "EVAL_RESULTS_FILE",
                        "GAME_LOGGING", "GAME_STATS_WRITER", "GAME_LOG_DIR",
                        "GAME_LOG_FORMAT", "GAME_LOG_STYLE", "GOEXPLORE_PHASE",
                        "LADDER_SKILLS", "LEAGUE_ANCHOR_ENABLE",
                        "LEAGUE_BASELINE_DECKLIST_FILE", "LEAGUE_DEBUG",
                        "LEAGUE_EVAL_FORCE", "LEAGUE_EVAL_GAME_LOGGING",
                        "LEAGUE_MODE", "LEAGUE_REGISTRY_PATH", "LEAGUE_REPORTS_DIR",
                        "MATCHUP_BALANCED_SAMPLING", "MODE", "MTG_AI_LOG_LEVEL",
                        "OPPONENT_SAMPLER", "POP_ANCHOR_SKILLS",
                        "POP_COMPLETION_AWARE", "POP_GATE_HOLD_FALLBACK",
                        "POP_GATE_PAIRED", "POP_MIX_GUARD_ABORT", "POP_SCHED_DEBUG",
                        "SELFPLAY_OPPONENT_TRAINING", "SKILL_MIX",
                        "TERMINAL_REWARD_SCALE", "TRAIN_DIAG", "TRAIN_PROFILES_LIST",
                        "VALUE_ACCURACY_MCTS_THRESHOLD", "WIN_REPLAY_DEDUP",
                        "WIN_REPLAY_ENABLE", "WORLD_MODEL_DIAG",

                        // All non-RL path/log overrides read by RLLogPaths.
                        "MTG_MODEL_PATH", "MODEL_PATH", "EPISODE_COUNTER_PATH",
                        "MULLIGAN_EPISODE_COUNTER_PATH", "MULLIGAN_MODEL_PATH",
                        "SNAPSHOT_DIR", "HEALTH_LOG_PATH", "GAME_KILLS_LOG_PATH",
                        "ACTIVATION_FAILURES_LOG_PATH", "HEAD_USAGE_LOG_PATH",
                        "TRAINING_LOSSES_PATH", "STATS_PATH", "EVAL_STATS_PATH",
                        "MULLIGAN_STATS_PATH", "TRAINING_GAME_LOGS_DIR",
                        "EVAL_GAME_LOGS_DIR", "LEAGUE_EVENTS_LOG_PATH",
                        "LEAGUE_STATUS_PATH", "LEAGUE_STATE_PATH", "PYTHON_LOGS_DIR",
                        "MTG_AI_LOG_FILE", "VRAM_DIAGNOSTICS_LOG_FILE",
                        "DRAFT_MODEL_PROFILE", "DRAFT_MODELS_DIR", "DRAFT_LOGS_DIR",
                        "DRAFT_MODEL_PATH", "DRAFT_EPISODE_COUNTER_PATH",
                        "DRAFT_GAME_LOGS_DIR", "DRAFT_EVAL_GAME_LOGS_DIR",
                        "DRAFT_BENCHMARK_GAME_LOGS_DIR",
                        "DRAFT_BENCHMARK_STATS_PATH",
                        "DRAFT_BENCHMARK_STATE_PATH", "DRAFT_STATS_PATH",
                        "DRAFT_LOSSES_PATH", "PY_BACKEND_MODE", "PY_BATCH_LOG_LEVEL")));
        private static final List<String> FAIL_SAFE_BEHAVIOR_ENV_PREFIXES =
                Collections.unmodifiableList(Arrays.asList(
                        "EVAL_", "AI_", "TORCH_", "MTG_", "GAME_",
                        "MULLIGAN_", "DRAFT_", "MCTS_", "ISMCTS_",
                        "ROUTER_", "SEARCH_OP_", "SPY_", "SIL_", "PY_",
                        "LEAGUE_", "POP_", "ACTOR_", "ADAPTIVE_",
                        "BENCHMARK_", "BOT_", "DECK_", "GOEXPLORE_",
                        "LADDER_", "MATCHUP_", "MODEL_", "OPPONENT_",
                        "SELFPLAY_", "SKILL_", "SNAPSHOT_", "TERMINAL_",
                        "TRAIN_", "VALUE_", "WIN_", "WORLD_", "CANDIDATE_",
                        "PAUSE_", "MULTI_", "ACTIVATION_", "EPISODE_",
                        "HEALTH_", "HEAD_", "STATS_", "TRAINING_",
                        "VRAM_", "PYTHON_"));
        private static final Set<String> ALWAYS_TRACKED_ENV = trackedEnvironmentNames();

        final List<Map<String, Object>> jvmArguments;
        final String jvmArgumentManifestSha256;
        final int systemPropertyCount;
        final String systemPropertyManifestSha256;
        final List<Map<String, Object>> relevantEnvironment;
        final String relevantEnvironmentManifestSha256;
        final List<String> invalidReasons;

        private RuntimeConfiguration(List<Map<String, Object>> jvmArguments,
                                     String jvmArgumentManifestSha256,
                                     int systemPropertyCount,
                                     String systemPropertyManifestSha256,
                                     List<Map<String, Object>> relevantEnvironment,
                                     String relevantEnvironmentManifestSha256,
                                     List<String> invalidReasons) {
            this.jvmArguments = jvmArguments;
            this.jvmArgumentManifestSha256 = jvmArgumentManifestSha256;
            this.systemPropertyCount = systemPropertyCount;
            this.systemPropertyManifestSha256 = systemPropertyManifestSha256;
            this.relevantEnvironment = relevantEnvironment;
            this.relevantEnvironmentManifestSha256 = relevantEnvironmentManifestSha256;
            this.invalidReasons = invalidReasons;
        }

        private static Set<String> trackedEnvironmentNames() {
            Set<String> names = new HashSet<>(Arrays.asList(
                    "RL_ACTIVATION_DIAG", "MAGE_DB_DIR", "MAGE_DB_AUTO_SERVER",
                    "RL_ARTIFACTS_ROOT", "RL_MODELS_DIR", "RL_LOGS_DIR",
                    "MODEL_PROFILE", "JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS",
                    "JDK_JAVA_OPTIONS"));
            names.addAll(SOURCE_AUDITED_FORBIDDEN_ENV);
            return Collections.unmodifiableSet(names);
        }

        static RuntimeConfiguration capture(List<String> inputArguments) throws Exception {
            List<Map<String, Object>> argumentRecords = new ArrayList<>();
            MessageDigest argumentManifest = MessageDigest.getInstance("SHA-256");
            List<String> invalid = new ArrayList<>();
            for (int i = 0; i < inputArguments.size(); i++) {
                String argument = inputArguments.get(i);
                JvmArgumentClassification classification = classifyJvmArgument(argument);
                updateFramed(argumentManifest, Integer.toString(i));
                updateFramed(argumentManifest, argument);
                Map<String, Object> record = new LinkedHashMap<>();
                record.put("argument_id", String.format(Locale.ROOT, "jvm_arg_%03d", i));
                record.put("category", classification.category);
                record.put("allowed", classification.allowed);
                record.put("value_sha256", digestUtf8(argument));
                argumentRecords.add(record);
                if (!classification.allowed) {
                    addUnique(invalid, "unallowlisted_jvm_argument_present");
                }
            }

            Properties properties = System.getProperties();
            TreeMap<String, String> normalizedProperties = new TreeMap<>();
            for (String name : properties.stringPropertyNames()) {
                normalizedProperties.put(name, properties.getProperty(name, ""));
                String lower = name.toLowerCase(Locale.ROOT);
                if (lower.startsWith("rl.") || lower.startsWith("mage.")
                        || lower.startsWith("xmage.") || lower.startsWith("eval.")
                        || lower.startsWith("kernel.")) {
                    addUnique(invalid, "behavior_system_property_present");
                }
            }
            MessageDigest propertyManifest = MessageDigest.getInstance("SHA-256");
            for (Map.Entry<String, String> entry : normalizedProperties.entrySet()) {
                updateFramed(propertyManifest, entry.getKey());
                updateFramed(propertyManifest, entry.getValue());
            }

            EnvironmentInventory environmentInventory = captureEnvironment(System.getenv());
            for (String reason : environmentInventory.invalidReasons) {
                addUnique(invalid, reason);
            }
            return new RuntimeConfiguration(
                    argumentRecords, hex(argumentManifest.digest()),
                    normalizedProperties.size(), hex(propertyManifest.digest()),
                    environmentInventory.records, environmentInventory.manifestSha256, invalid);
        }

        private static EnvironmentInventory captureEnvironment(
                Map<String, String> environment) throws Exception {
            if (environment == null) {
                throw new IllegalStateException("environment inventory is null");
            }
            Set<String> relevantNames = new HashSet<>(ALWAYS_TRACKED_ENV);
            for (String name : environment.keySet()) {
                if (name.startsWith("RL_") || name.startsWith("MAGE_")
                        || SOURCE_AUDITED_FORBIDDEN_ENV.contains(name)
                        || isFailSafeBehaviorEnvironmentName(name)) {
                    relevantNames.add(name);
                }
            }
            List<String> orderedNames = new ArrayList<>(relevantNames);
            Collections.sort(orderedNames);
            List<Map<String, Object>> environmentRecords = new ArrayList<>();
            MessageDigest environmentManifest = MessageDigest.getInstance("SHA-256");
            List<String> invalid = new ArrayList<>();
            for (String name : orderedNames) {
                String value = environment.get(name);
                boolean present = value != null;
                EnvironmentClassification classification = classifyEnvironment(name, value);
                updateFramed(environmentManifest, name);
                updateFramed(environmentManifest, present ? value : "<absent>");
                Map<String, Object> record = new LinkedHashMap<>();
                record.put("name", name);
                record.put("present", present);
                record.put("category", classification.category);
                record.put("allowed", classification.allowed);
                record.put("value_sha256", present ? digestUtf8(value) : null);
                environmentRecords.add(record);
                if (present && !classification.allowed) {
                    addUnique(invalid, classification.invalidReason);
                }
            }
            return new EnvironmentInventory(
                    environmentRecords, hex(environmentManifest.digest()), invalid);
        }

        private static JvmArgumentClassification classifyJvmArgument(String argument) {
            if (argument.matches("-X(?:ms|mx|ss)[0-9]+[kKmMgG]?")) {
                return new JvmArgumentClassification(true, "heap_or_stack_size");
            }
            java.util.regex.Matcher xx = Pattern
                    .compile("-XX:(?:[+-])?([A-Za-z0-9]+)(?:=([A-Za-z0-9._+-]+))?")
                    .matcher(argument);
            if (xx.matches() && ALLOWED_XX_OPTIONS.contains(xx.group(1))) {
                return new JvmArgumentClassification(true, "allowlisted_xx_runtime_option");
            }
            if ("-Dfile.encoding=UTF-8".equals(argument)
                    || "-Duser.timezone=UTC".equals(argument)
                    || "-Djava.awt.headless=true".equals(argument)) {
                return new JvmArgumentClassification(true, "allowlisted_system_property");
            }
            if (argument.equals("-Xint") || argument.equals("-Xcomp")
                    || argument.equals("-Xmixed") || argument.startsWith("-XX:TieredStopAtLevel")) {
                return new JvmArgumentClassification(false, "compilation_mode_or_control");
            }
            if (argument.startsWith("-javaagent:") || argument.startsWith("-agentlib:")
                    || argument.startsWith("-agentpath:")) {
                return new JvmArgumentClassification(false, "agent");
            }
            if (argument.startsWith("-D")) {
                return new JvmArgumentClassification(false, "unallowlisted_system_property");
            }
            return new JvmArgumentClassification(false, "unallowlisted_jvm_option");
        }

        private static EnvironmentClassification classifyEnvironment(String name, String value) {
            if (value == null) {
                return new EnvironmentClassification(true, "tracked_absence", "");
            }
            if ("JAVA_TOOL_OPTIONS".equals(name) || "_JAVA_OPTIONS".equals(name)
                    || "JDK_JAVA_OPTIONS".equals(name)) {
                return new EnvironmentClassification(
                        false, "java_option_environment",
                        "java_option_environment_variable_present");
            }
            if ("MAGE_DB_DIR".equals(name)) {
                boolean allowed = !value.trim().isEmpty() && Paths.get(value.trim()).isAbsolute();
                return new EnvironmentClassification(
                        allowed, "guarded_absolute_database_path",
                        "MAGE_DB_DIR_must_be_absolute_when_present");
            }
            if ("MAGE_DB_AUTO_SERVER".equals(name)) {
                return new EnvironmentClassification(
                        "0".equals(value), "database_auto_server_exact_zero",
                        "MAGE_DB_AUTO_SERVER_must_equal_0_when_present");
            }
            if ("RL_ACTIVATION_DIAG".equals(name)) {
                return new EnvironmentClassification(
                        "0".equals(value), "activation_diagnostics_exact_zero",
                        "RL_ACTIVATION_DIAG_must_equal_0_when_present");
            }
            if ("RL_ARTIFACTS_ROOT".equals(name) || "RL_MODELS_DIR".equals(name)
                    || "RL_LOGS_DIR".equals(name)) {
                boolean allowed = !value.trim().isEmpty() && Paths.get(value.trim()).isAbsolute();
                return new EnvironmentClassification(
                        allowed, "guarded_absolute_artifact_path",
                        "RL_artifact_paths_must_be_absolute_when_present");
            }
            if ("MODEL_PROFILE".equals(name)) {
                boolean allowed = value.matches("[A-Za-z0-9._-]{1,64}");
                return new EnvironmentClassification(
                        allowed, "model_profile_label",
                        "MODEL_PROFILE_is_not_privacy_safe");
            }
            if (SOURCE_AUDITED_FORBIDDEN_ENV.contains(name)) {
                return new EnvironmentClassification(
                        false, "source_audited_benchmark_environment",
                        "source_audited_benchmark_environment_present");
            }
            if (name.startsWith("RL_") || name.startsWith("MAGE_")) {
                return new EnvironmentClassification(
                        false, name.startsWith("RL_") ? "unallowlisted_rl_environment"
                                : "unallowlisted_mage_environment",
                        "unallowlisted_RL_or_MAGE_environment_present");
            }
            if (isFailSafeBehaviorEnvironmentName(name)) {
                return new EnvironmentClassification(
                        false, "fail_safe_behavior_environment",
                        "fail_safe_behavior_environment_present");
            }
            return new EnvironmentClassification(
                    false, "unexpected_tracked_environment",
                    "unexpected_tracked_environment_present");
        }

        private static boolean isFailSafeBehaviorEnvironmentName(String name) {
            if (name == null || name.isEmpty()) {
                return false;
            }
            if ("MODE".equals(name)) {
                return true;
            }
            for (String prefix : FAIL_SAFE_BEHAVIOR_ENV_PREFIXES) {
                if (name.startsWith(prefix)) {
                    return true;
                }
            }
            return false;
        }
    }

    private static final class EnvironmentInventory {
        final List<Map<String, Object>> records;
        final String manifestSha256;
        final List<String> invalidReasons;

        private EnvironmentInventory(List<Map<String, Object>> records,
                                     String manifestSha256,
                                     List<String> invalidReasons) {
            this.records = records;
            this.manifestSha256 = manifestSha256;
            this.invalidReasons = invalidReasons;
        }
    }

    private static final class JvmArgumentClassification {
        final boolean allowed;
        final String category;

        JvmArgumentClassification(boolean allowed, String category) {
            this.allowed = allowed;
            this.category = category;
        }
    }

    private static final class EnvironmentClassification {
        final boolean allowed;
        final String category;
        final String invalidReason;

        EnvironmentClassification(boolean allowed, String category, String invalidReason) {
            this.allowed = allowed;
            this.category = category;
            this.invalidReason = invalidReason;
        }
    }

    private static final class RuntimeBinding {
        final int availableProcessors;
        final Map<String, Object> stableRuntimeContract;
        final List<String> configurationInvalidReasons;
        final List<Path> entryPaths;
        final List<ContentBinding> entries;
        final String aggregateSha256;

        private RuntimeBinding(int availableProcessors,
                               Map<String, Object> stableRuntimeContract,
                               List<String> configurationInvalidReasons,
                               List<Path> entryPaths,
                               List<ContentBinding> entries,
                               String aggregateSha256) {
            this.availableProcessors = availableProcessors;
            this.stableRuntimeContract = stableRuntimeContract;
            this.configurationInvalidReasons = configurationInvalidReasons;
            this.entryPaths = entryPaths;
            this.entries = entries;
            this.aggregateSha256 = aggregateSha256;
        }

        static RuntimeBinding capture(Args args) throws Exception {
            int processors = Runtime.getRuntime().availableProcessors();
            Map<String, Object> stable = new LinkedHashMap<>(runtimeContractMap(args, processors));
            @SuppressWarnings("unchecked")
            List<String> invalid = new ArrayList<>((List<String>)
                    stable.get("configuration_invalid_reasons"));
            List<Path> paths = actualClasspathEntries();
            if (paths.isEmpty()) {
                throw new IllegalStateException("actual runtime classpath entry set is empty");
            }
            List<ContentBinding> entries = new ArrayList<>(paths.size());
            MessageDigest aggregate = MessageDigest.getInstance("SHA-256");
            for (int i = 0; i < paths.size(); i++) {
                ContentBinding binding = ContentBinding.capture(paths.get(i), path -> true);
                if ("absent".equals(binding.kind)) {
                    throw new IllegalStateException("actual runtime classpath entry is absent");
                }
                entries.add(binding);
                updateFramed(aggregate, Integer.toString(i));
                updateFramed(aggregate, binding.kind);
                updateFramed(aggregate, Long.toString(binding.fileCount));
                updateFramed(aggregate, Long.toString(binding.byteCount));
                updateFramed(aggregate, binding.manifestSha256);
            }
            return new RuntimeBinding(processors, stable, invalid, paths, entries,
                    hex(aggregate.digest()));
        }

        boolean sameContractAndBytes(RuntimeBinding other) {
            if (other == null || availableProcessors != other.availableProcessors
                    || !stableRuntimeContract.equals(other.stableRuntimeContract)
                    || !entryPaths.equals(other.entryPaths)
                    || entries.size() != other.entries.size()
                    || !aggregateSha256.equals(other.aggregateSha256)) {
                return false;
            }
            for (int i = 0; i < entries.size(); i++) {
                if (!entries.get(i).sameContent(other.entries.get(i))) {
                    return false;
                }
            }
            return true;
        }

        Map<String, Object> toMap(RuntimeBinding after) throws Exception {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("binding_type", "every_actual_runtime_classpath_entry_recursive_sha256");
            map.put("absolute_paths_emitted", false);
            map.put("before", classpathMap());
            map.put("after", after.classpathMap());
            map.put("runtime_contract_before_sha256", digestUtf8(toJson(stableRuntimeContract)));
            map.put("runtime_contract_after_sha256", digestUtf8(toJson(after.stableRuntimeContract)));
            map.put("unchanged", sameContractAndBytes(after));
            return map;
        }

        private Map<String, Object> classpathMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("entry_count", entries.size());
            map.put("aggregate_manifest_sha256", aggregateSha256);
            List<Map<String, Object>> outputEntries = new ArrayList<>(entries.size());
            for (int i = 0; i < entries.size(); i++) {
                Map<String, Object> entry = new LinkedHashMap<>();
                entry.put("entry_id", String.format(Locale.ROOT, "entry_%04d", i));
                entry.putAll(entries.get(i).toMap());
                outputEntries.add(entry);
            }
            map.put("entries", outputEntries);
            return map;
        }

        private static List<Path> actualClasspathEntries() throws Exception {
            Set<Path> paths = new LinkedHashSet<>();
            String rawClasspath = System.getProperty("java.class.path", "");
            if (!rawClasspath.isEmpty()) {
                for (String raw : rawClasspath.split(Pattern.quote(File.pathSeparator))) {
                    if (!raw.trim().isEmpty()) {
                        paths.add(canonicalClasspathPath(Paths.get(raw)));
                    }
                }
            }
            Set<ClassLoader> seen = Collections.newSetFromMap(new java.util.IdentityHashMap<>());
            List<ClassLoader> starts = Arrays.asList(
                    Thread.currentThread().getContextClassLoader(),
                    XMageUniformMirrorBenchmark.class.getClassLoader(),
                    ClassLoader.getSystemClassLoader());
            for (ClassLoader start : starts) {
                for (ClassLoader loader = start; loader != null && seen.add(loader);
                     loader = loader.getParent()) {
                    if (loader instanceof URLClassLoader) {
                        for (URL url : ((URLClassLoader) loader).getURLs()) {
                            if (!"file".equalsIgnoreCase(url.getProtocol())) {
                                throw new IllegalStateException("runtime classpath contains a non-file URL");
                            }
                            URI uri = url.toURI();
                            paths.add(canonicalClasspathPath(Paths.get(uri)));
                        }
                    }
                }
            }
            List<Path> ordered = new ArrayList<>(paths);
            ordered.sort(Comparator.comparing(Path::toString));
            return ordered;
        }

        private static Path canonicalClasspathPath(Path path) throws Exception {
            Path absolute = path.toAbsolutePath().normalize();
            return Files.exists(absolute) ? absolute.toRealPath() : absolute;
        }
    }

    private static final class MutableArtifactGuard {
        final Map<String, Path> watchedRoots;
        final Map<String, ContentBinding> before;
        final boolean prebuiltCatalogPresent;
        final boolean catalogFilesReadOnly;
        final String databaseLocationRole;
        final String resolvedWorkingDirectorySha256;
        final String resolvedDatabaseLocationSha256;
        final List<String> configurationInvalidReasons;
        Map<String, ContentBinding> after;

        private MutableArtifactGuard(Map<String, Path> watchedRoots,
                                     Map<String, ContentBinding> before,
                                     boolean prebuiltCatalogPresent,
                                     boolean catalogFilesReadOnly,
                                     String databaseLocationRole,
                                     String resolvedWorkingDirectorySha256,
                                     String resolvedDatabaseLocationSha256,
                                     List<String> configurationInvalidReasons) {
            this.watchedRoots = watchedRoots;
            this.before = before;
            this.prebuiltCatalogPresent = prebuiltCatalogPresent;
            this.catalogFilesReadOnly = catalogFilesReadOnly;
            this.databaseLocationRole = databaseLocationRole;
            this.resolvedWorkingDirectorySha256 = resolvedWorkingDirectorySha256;
            this.resolvedDatabaseLocationSha256 = resolvedDatabaseLocationSha256;
            this.configurationInvalidReasons = configurationInvalidReasons;
        }

        static MutableArtifactGuard captureBeforeScanner(
                Path repoRoot, String bindingMode) throws Exception {
            String userDir = System.getProperty("user.dir");
            if (userDir == null || userDir.trim().isEmpty()) {
                throw new IllegalStateException("user.dir is absent");
            }
            Path workingDirectory = Paths.get(userDir).toRealPath();
            if (!workingDirectory.equals(repoRoot)) {
                throw new IllegalStateException(
                        "canonical user.dir must equal canonical repository root");
            }
            Path defaultArtifacts = repoRoot.resolve(
                    "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl");
            Path artifacts = configuredPath(repoRoot, "RL_ARTIFACTS_ROOT", defaultArtifacts);
            String profile = envValue("MODEL_PROFILE");
            Path defaultModels = profile.isEmpty()
                    ? artifacts.resolve("models") : artifacts.resolve("models").resolve(profile);
            Path defaultLogs = profile.isEmpty()
                    ? artifacts.resolve("logs") : artifacts.resolve("logs").resolve(profile);
            String configuredDb = envValue("MAGE_DB_DIR");
            Path configuredDbPath = configuredDb.isEmpty() ? null : Paths.get(configuredDb);
            String databaseRole;
            Path db;
            if (configuredDbPath == null) {
                databaseRole = "repository_relative_default_db";
                db = workingDirectory.resolve("db");
            } else if (configuredDbPath.isAbsolute()) {
                databaseRole = "explicit_absolute_mage_db_dir";
                db = configuredDbPath;
            } else {
                databaseRole = "explicit_relative_mage_db_dir";
                db = workingDirectory.resolve(configuredDbPath);
            }
            db = db.normalize().toAbsolutePath();
            if (Files.exists(db)) {
                db = db.toRealPath();
            }
            Map<String, Path> roots = new LinkedHashMap<>();
            roots.put("artifacts_root", artifacts);
            roots.put("models_dir", configuredPath(repoRoot, "RL_MODELS_DIR", defaultModels));
            roots.put("logs_dir", configuredPath(repoRoot, "RL_LOGS_DIR", defaultLogs));
            roots.put("profiles_dir", artifacts.resolve("profiles").normalize().toAbsolutePath());
            roots.put("card_database_catalog", db);
            Map<String, ContentBinding> before = snapshot(roots);
            ContentBinding catalog = before.get("card_database_catalog");
            boolean prebuilt = catalog != null
                    && catalog.fileCount > 0L && catalog.byteCount > 0L;
            boolean readOnly = persistentCatalogFilesReadOnly(db);
            List<String> invalid = new ArrayList<>();
            if ("strict".equals(bindingMode)) {
                if (configuredDbPath == null || !configuredDbPath.isAbsolute()) {
                    invalid.add("strict_requires_explicit_absolute_MAGE_DB_DIR");
                }
                if (!readOnly) {
                    invalid.add("strict_requires_read_only_prebuilt_card_catalog");
                }
                if (!"0".equals(envValue("MAGE_DB_AUTO_SERVER"))) {
                    invalid.add("strict_requires_MAGE_DB_AUTO_SERVER_0");
                }
            }
            return new MutableArtifactGuard(
                    roots, before, prebuilt, readOnly, databaseRole,
                    digestUtf8(workingDirectory.toString()), digestUtf8(db.toString()), invalid);
        }

        List<String> verifyUnchanged() throws Exception {
            after = snapshot(watchedRoots);
            List<String> invalid = new ArrayList<>();
            for (String label : before.keySet()) {
                if (!before.get(label).sameContent(after.get(label))) {
                    invalid.add("card_database_catalog".equals(label)
                            ? "card_catalog_bytes_changed_after_pre_scan_snapshot"
                            : "mutable_" + label + "_bytes_changed");
                }
            }
            return invalid;
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("watched_root_labels", new ArrayList<>(watchedRoots.keySet()));
            map.put("prebuilt_card_catalog_present", prebuiltCatalogPresent);
            map.put("catalog_files_read_only", catalogFilesReadOnly);
            map.put("working_directory_role", "canonical_repository_root");
            map.put("database_location_role", databaseLocationRole);
            map.put("resolved_working_directory_sha256", resolvedWorkingDirectorySha256);
            map.put("resolved_database_location_sha256", resolvedDatabaseLocationSha256);
            map.put("baseline_timing", "before CardScanner or any card repository access");
            map.put("before", contentBindingsMap(before));
            map.put("after", contentBindingsMap(after));
            map.put("unchanged", sameBindings(before, after));
            return map;
        }

        private static Map<String, ContentBinding> snapshot(Map<String, Path> roots) throws Exception {
            Map<String, ContentBinding> snapshot = new LinkedHashMap<>();
            for (Map.Entry<String, Path> entry : roots.entrySet()) {
                if ("card_database_catalog".equals(entry.getKey())) {
                    snapshot.put(entry.getKey(), ContentBinding.capture(
                            entry.getValue(), MutableArtifactGuard::isPersistentCardDatabase));
                } else {
                    snapshot.put(entry.getKey(), ContentBinding.capture(entry.getValue(), path -> true));
                }
            }
            return snapshot;
        }

        private static boolean isPersistentCardDatabase(Path path) {
            String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
            return name.startsWith("cards.h2")
                    && (name.endsWith(".mv.db") || name.endsWith(".h2.db"));
        }

        private static boolean persistentCatalogFilesReadOnly(Path db) throws IOException {
            if (!Files.isDirectory(db)) {
                return false;
            }
            List<Path> files;
            try (java.util.stream.Stream<Path> stream = Files.list(db)) {
                files = stream.filter(Files::isRegularFile)
                        .filter(MutableArtifactGuard::isPersistentCardDatabase)
                        .collect(java.util.stream.Collectors.toList());
            }
            if (files.isEmpty()) {
                return false;
            }
            for (Path file : files) {
                boolean readOnly;
                try {
                    readOnly = Files.readAttributes(
                            file, java.nio.file.attribute.DosFileAttributes.class).isReadOnly();
                } catch (UnsupportedOperationException e) {
                    readOnly = !Files.isWritable(file);
                }
                if (!readOnly && Files.isWritable(file)) {
                    return false;
                }
            }
            return true;
        }

        private static Path configuredPath(Path repoRoot, String envName, Path defaultPath) {
            String raw = envValue(envName);
            Path path = raw.isEmpty() ? defaultPath : Paths.get(raw);
            if (!path.isAbsolute()) {
                path = repoRoot.resolve(path);
            }
            return path.normalize().toAbsolutePath();
        }

        private static String envValue(String name) {
            String value = System.getenv(name);
            return value == null ? "" : value.trim();
        }
    }

    private static final class ContentBinding {
        final String kind;
        final long fileCount;
        final long byteCount;
        final String manifestSha256;

        private ContentBinding(String kind, long fileCount, long byteCount,
                               String manifestSha256) {
            this.kind = kind;
            this.fileCount = fileCount;
            this.byteCount = byteCount;
            this.manifestSha256 = manifestSha256;
        }

        static ContentBinding capture(Path root,
                                      java.util.function.Predicate<Path> include) throws Exception {
            if (!Files.exists(root)) {
                return new ContentBinding("absent", 0L, 0L, digestUtf8("absent"));
            }
            if (Files.isRegularFile(root)) {
                if (!include.test(root)) {
                    return new ContentBinding("filtered_empty", 0L, 0L, digestUtf8("filtered_empty"));
                }
                long size = Files.size(root);
                return new ContentBinding("file", 1L, size, sha256(root));
            }
            if (!Files.isDirectory(root)) {
                throw new IllegalStateException("watched content root has unsupported filesystem type");
            }
            List<Path> files = new ArrayList<>();
            try (java.util.stream.Stream<Path> stream = Files.walk(root)) {
                stream.filter(Files::isRegularFile).filter(include).forEach(files::add);
            }
            files.sort(Comparator.comparing(path ->
                    root.relativize(path).toString().replace('\\', '/')));
            MessageDigest manifest = MessageDigest.getInstance("SHA-256");
            long bytes = 0L;
            for (Path file : files) {
                String relative = root.relativize(file).toString().replace('\\', '/');
                long size = Files.size(file);
                bytes = Math.addExact(bytes, size);
                updateFramed(manifest, relative);
                updateFramed(manifest, Long.toString(size));
                updateFramed(manifest, sha256(file));
            }
            return new ContentBinding("directory", files.size(), bytes,
                    hex(manifest.digest()));
        }

        boolean sameContent(ContentBinding other) {
            return other != null
                    && kind.equals(other.kind)
                    && fileCount == other.fileCount
                    && byteCount == other.byteCount
                    && manifestSha256.equals(other.manifestSha256);
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("kind", kind);
            map.put("file_count", fileCount);
            map.put("byte_count", byteCount);
            map.put("manifest_sha256", manifestSha256);
            return map;
        }
    }

    private static boolean sameBindings(Map<String, ContentBinding> left,
                                        Map<String, ContentBinding> right) {
        if (left == null || right == null || !left.keySet().equals(right.keySet())) {
            return false;
        }
        for (String key : left.keySet()) {
            if (!left.get(key).sameContent(right.get(key))) {
                return false;
            }
        }
        return true;
    }

    private static Map<String, Object> contentBindingsMap(Map<String, ContentBinding> bindings) {
        if (bindings == null) {
            return Collections.emptyMap();
        }
        Map<String, Object> map = new LinkedHashMap<>();
        for (Map.Entry<String, ContentBinding> entry : bindings.entrySet()) {
            map.put(entry.getKey(), entry.getValue().toMap());
        }
        return map;
    }

    private static final class DeckResources {
        final Deck p0Template;
        final Deck p1Template;
        final int mainCount;
        final int sideboardCount;
        final int cardClassCount;
        final String cardClassManifestSha256;

        private DeckResources(Deck p0Template, Deck p1Template, int mainCount,
                              int sideboardCount, int cardClassCount,
                              String cardClassManifestSha256) {
            this.p0Template = p0Template;
            this.p1Template = p1Template;
            this.mainCount = mainCount;
            this.sideboardCount = sideboardCount;
            this.cardClassCount = cardClassCount;
            this.cardClassManifestSha256 = cardClassManifestSha256;
        }

        static DeckResources load(Path deckPath) throws Exception {
            StringBuilder warnings = new StringBuilder();
            DeckCardLists lists = DeckImporter.importDeckFromFile(deckPath.toString(), warnings, false);
            if (lists == null) {
                throw new IllegalStateException("deck importer returned null");
            }
            if (!warnings.toString().trim().isEmpty()) {
                throw new IllegalStateException("deck importer warnings: "
                        + warnings.toString().replace('\r', ' ').replace('\n', ' '));
            }
            Deck p0 = Deck.load(lists, false, false, null);
            Deck p1 = Deck.load(lists, false, false, null);
            if (p0.getCards().size() != 60 || p1.getCards().size() != 60) {
                throw new IllegalStateException("canonical Rally maindeck is not exactly 60 cards per seat");
            }
            if (p0.getSideboard().size() != 15 || p1.getSideboard().size() != 15) {
                throw new IllegalStateException("canonical Rally source sideboard is not exactly 15 cards per seat");
            }
            verifyDisjointCardIds(p0, p1, "fresh seat templates");
            Set<Class<?>> classes = new HashSet<>();
            for (Card card : p0.getCards()) {
                classes.add(card.getClass());
            }
            for (Card card : p0.getSideboard()) {
                classes.add(card.getClass());
            }
            List<Class<?>> ordered = new ArrayList<>(classes);
            ordered.sort(Comparator.comparing(Class::getName));
            MessageDigest manifest = MessageDigest.getInstance("SHA-256");
            for (Class<?> cls : ordered) {
                updateFramed(manifest, cls.getName());
                updateFramed(manifest, sha256ClassResource(cls));
            }
            return new DeckResources(
                    p0, p1, 60, 15, ordered.size(), hex(manifest.digest()));
        }
    }

    private static final class MutableSplitMix64 {
        private long state;

        MutableSplitMix64(long seed) {
            this.state = seed;
        }

        long next() {
            state += SeededUniformMirrorPolicy.GOLDEN_RATIO_64;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58_476D_1CE4_E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D0_49BB_1331_11EBL;
            return z ^ (z >>> 31);
        }
    }

    private static final class ActorResult {
        final int actorIndex;
        long phaseStartNanos;
        long deadlineNanos;
        long finishNanos;
        long attempted;
        long firstEpisodeId = -1L;
        long lastEpisodeId = -1L;
        long natural;
        long gamesWithAnyInvalidity;
        long p0Wins;
        long p1Wins;
        long draws;
        long finishedAfterDeadline;
        long turns;
        long maxTurns;
        long physicalDecisions;
        long policyActionSelections;
        long policyLeafEvaluations;
        long canonicalTies;
        long forcedNoPolicySelections;
        long explicitConcessionAttempts;
        long policyFailures;
        long canonicalizationFailures;
        long modelFailures;
        long terminalLeftFlags;
        long gameNanos;
        final Map<String, Long> invalidClasses = new TreeMap<>();
        final Map<String, Long> decisionCategories = new TreeMap<>();
        final Map<String, Long> outcomeHistogram = new TreeMap<>();
        final Map<String, Long> throwableClasses = new TreeMap<>();
        Throwable actorFailure;

        ActorResult(int actorIndex) {
            this.actorIndex = actorIndex;
        }

        void recordAttempt(long episodeId) {
            attempted++;
            if (firstEpisodeId < 0L) {
                firstEpisodeId = episodeId;
            }
            lastEpisodeId = episodeId;
        }

        void record(GameResult game) {
            if (game.natural) {
                natural++;
                if (game.p0Win) p0Wins++;
                if (game.p1Win) p1Wins++;
                if (game.draw) draws++;
            } else {
                gamesWithAnyInvalidity++;
            }
            if (game.finishedAfterDeadline) finishedAfterDeadline++;
            turns += game.turns;
            maxTurns = Math.max(maxTurns, game.turns);
            physicalDecisions += game.physicalDecisions;
            policyActionSelections += game.policyActionSelections;
            policyLeafEvaluations += game.policyLeafEvaluations;
            canonicalTies += game.canonicalTies;
            forcedNoPolicySelections += game.forcedNoPolicySelections;
            explicitConcessionAttempts += game.explicitConcessionAttempts;
            policyFailures += game.policyFailures;
            canonicalizationFailures += game.canonicalizationFailures;
            modelFailures += game.modelFailures;
            terminalLeftFlags += game.terminalLeftFlags;
            gameNanos += Math.max(0L, game.finishNanos - game.startNanos);
            merge(invalidClasses, game.invalidClasses);
            merge(decisionCategories, game.decisionCategories);
            merge(outcomeHistogram, game.outcomeHistogram);
            merge(throwableClasses, game.throwableClasses);
        }
    }

    private static final class PhaseResult {
        final String name;
        final long requestedNanos;
        final long elapsedNanos;
        final long attempted;
        final long naturalCompletions;
        final long gamesWithAnyInvalidity;
        final long p0Wins;
        final long p1Wins;
        final long draws;
        final long finishedAfterDeadline;
        final long turns;
        final long maxTurns;
        final long physicalDecisions;
        final long policyActionSelections;
        final long policyLeafEvaluations;
        final long canonicalTies;
        final long forcedNoPolicySelections;
        final long explicitConcessionAttempts;
        final long policyFailures;
        final long canonicalizationFailures;
        final long modelFailures;
        final long terminalLeftFlags;
        final long actorFailures;
        final Map<String, Long> invalidClasses;
        final Map<String, Long> decisionCategories;
        final Map<String, Long> outcomeHistogram;
        final Map<String, Long> throwableClasses;
        final List<Map<String, Object>> actors;

        private PhaseResult(String name, long requestedNanos, long elapsedNanos,
                            long attempted, long naturalCompletions, long gamesWithAnyInvalidity,
                            long p0Wins, long p1Wins, long draws, long finishedAfterDeadline,
                            long turns, long maxTurns, long physicalDecisions,
                            long policyActionSelections, long policyLeafEvaluations,
                            long canonicalTies, long forcedNoPolicySelections,
                            long explicitConcessionAttempts, long policyFailures,
                            long canonicalizationFailures, long modelFailures,
                            long terminalLeftFlags, long actorFailures,
                            Map<String, Long> invalidClasses,
                            Map<String, Long> decisionCategories,
                            Map<String, Long> outcomeHistogram,
                            Map<String, Long> throwableClasses,
                            List<Map<String, Object>> actors) {
            this.name = name;
            this.requestedNanos = requestedNanos;
            this.elapsedNanos = elapsedNanos;
            this.attempted = attempted;
            this.naturalCompletions = naturalCompletions;
            this.gamesWithAnyInvalidity = gamesWithAnyInvalidity;
            this.p0Wins = p0Wins;
            this.p1Wins = p1Wins;
            this.draws = draws;
            this.finishedAfterDeadline = finishedAfterDeadline;
            this.turns = turns;
            this.maxTurns = maxTurns;
            this.physicalDecisions = physicalDecisions;
            this.policyActionSelections = policyActionSelections;
            this.policyLeafEvaluations = policyLeafEvaluations;
            this.canonicalTies = canonicalTies;
            this.forcedNoPolicySelections = forcedNoPolicySelections;
            this.explicitConcessionAttempts = explicitConcessionAttempts;
            this.policyFailures = policyFailures;
            this.canonicalizationFailures = canonicalizationFailures;
            this.modelFailures = modelFailures;
            this.terminalLeftFlags = terminalLeftFlags;
            this.actorFailures = actorFailures;
            this.invalidClasses = invalidClasses;
            this.decisionCategories = decisionCategories;
            this.outcomeHistogram = outcomeHistogram;
            this.throwableClasses = throwableClasses;
            this.actors = actors;
        }

        static PhaseResult aggregate(String name, long requestedNanos, long phaseStart,
                                     long deadline, List<ActorResult> actorResults) {
            long slowestFinish = phaseStart;
            long attempted = 0L;
            long natural = 0L;
            long invalid = 0L;
            long p0Wins = 0L;
            long p1Wins = 0L;
            long draws = 0L;
            long tails = 0L;
            long turns = 0L;
            long maxTurns = 0L;
            long physical = 0L;
            long actions = 0L;
            long leaves = 0L;
            long ties = 0L;
            long forcedNoPolicy = 0L;
            long explicitConcessions = 0L;
            long policyFailures = 0L;
            long canonicalizationFailures = 0L;
            long modelFailures = 0L;
            long terminalLeftFlags = 0L;
            long actorFailures = 0L;
            Map<String, Long> invalidClasses = new TreeMap<>();
            Map<String, Long> categories = new TreeMap<>();
            Map<String, Long> histogram = new TreeMap<>();
            Map<String, Long> throwableClasses = new TreeMap<>();
            List<Map<String, Object>> actors = new ArrayList<>();
            for (ActorResult actor : actorResults) {
                slowestFinish = Math.max(slowestFinish, actor.finishNanos);
                attempted += actor.attempted;
                natural += actor.natural;
                invalid += actor.gamesWithAnyInvalidity;
                p0Wins += actor.p0Wins;
                p1Wins += actor.p1Wins;
                draws += actor.draws;
                tails += actor.finishedAfterDeadline;
                turns += actor.turns;
                maxTurns = Math.max(maxTurns, actor.maxTurns);
                physical += actor.physicalDecisions;
                actions += actor.policyActionSelections;
                leaves += actor.policyLeafEvaluations;
                ties += actor.canonicalTies;
                forcedNoPolicy += actor.forcedNoPolicySelections;
                explicitConcessions += actor.explicitConcessionAttempts;
                policyFailures += actor.policyFailures;
                canonicalizationFailures += actor.canonicalizationFailures;
                modelFailures += actor.modelFailures;
                terminalLeftFlags += actor.terminalLeftFlags;
                merge(invalidClasses, actor.invalidClasses);
                merge(categories, actor.decisionCategories);
                merge(histogram, actor.outcomeHistogram);
                merge(throwableClasses, actor.throwableClasses);
                if (actor.actorFailure != null) {
                    actorFailures++;
                    increment(throwableClasses, throwableClassLabel(actor.actorFailure), 1L);
                }
                Map<String, Object> actorMap = new LinkedHashMap<>();
                actorMap.put("actor_index", actor.actorIndex);
                actorMap.put("attempted", actor.attempted);
                actorMap.put("natural_completions", actor.natural);
                actorMap.put("first_episode_id", actor.firstEpisodeId < 0L ? null : actor.firstEpisodeId);
                actorMap.put("last_episode_id", actor.lastEpisodeId < 0L ? null : actor.lastEpisodeId);
                actorMap.put("finish_offset_seconds", nanosToSeconds(Math.max(1L, actor.finishNanos - phaseStart)));
                actorMap.put("actor_failure", actor.actorFailure == null
                        ? null : throwableFingerprint(actor.actorFailure));
                actors.add(actorMap);
            }
            long elapsed = Math.max(1L, slowestFinish - phaseStart);
            return new PhaseResult(name, requestedNanos, elapsed, attempted, natural, invalid,
                    p0Wins, p1Wins, draws, tails, turns, maxTurns, physical, actions, leaves,
                    ties, forcedNoPolicy, explicitConcessions, policyFailures,
                    canonicalizationFailures, modelFailures, terminalLeftFlags, actorFailures,
                    invalidClasses, categories, histogram, throwableClasses, actors);
        }

        Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("requested_wall_seconds", nanosToSeconds(requestedNanos));
            map.put("elapsed_slowest_actor_seconds", nanosToSeconds(elapsedNanos));
            map.put("attempted_games", attempted);
            map.put("natural_completions", naturalCompletions);
            map.put("games_with_any_invalidity", gamesWithAnyInvalidity);
            map.put("outcomes", outcomeMap(p0Wins, p1Wins, draws));
            map.put("in_flight_at_deadline_finished_naturally", finishedAfterDeadline);
            map.put("unfinished_after_join", 0);
            map.put("turns", turns);
            map.put("max_turns", maxTurns);
            map.put("physical_decisions", physicalDecisions);
            map.put("policy_action_selections", policyActionSelections);
            map.put("policy_leaf_evaluations", policyLeafEvaluations);
            map.put("canonical_encounter_ties", canonicalTies);
            map.put("forced_zero_policy_selections", forcedNoPolicySelections);
            map.put("explicit_concession_attempts", explicitConcessionAttempts);
            map.put("policy_failures", policyFailures);
            map.put("canonicalization_failures", canonicalizationFailures);
            map.put("model_failures", modelFailures);
            map.put("terminal_left_flags", terminalLeftFlags);
            map.put("terminal_left_interpretation", "allowed only after a flag-consistent terminal outcome; harness has no external leave/concede source");
            map.put("invalid_classes", invalidClasses);
            map.put("throwable_classes", throwableClasses);
            map.put("physical_decision_categories", decisionCategories);
            map.put("policy_outcome_histogram", outcomeHistogram);
            map.put("actor_failures", actorFailures);
            map.put("actors", actors);
            return map;
        }
    }

    private static Map<String, Object> outcomeMap(long p0Wins, long p1Wins, long draws) {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("p0_wins", p0Wins);
        map.put("p1_wins", p1Wins);
        map.put("draws", draws);
        return map;
    }

    private static final class GameResult {
        final long startNanos;
        final long finishNanos;
        boolean finishedAfterDeadline;
        boolean natural;
        boolean p0Win;
        boolean p1Win;
        boolean draw;
        long turns;
        long physicalDecisions;
        long policyActionSelections;
        long policyLeafEvaluations;
        long canonicalTies;
        long forcedNoPolicySelections;
        long explicitConcessionAttempts;
        long policyFailures;
        long canonicalizationFailures;
        long modelFailures;
        long terminalLeftFlags;
        final Map<String, Long> invalidClasses = new TreeMap<>();
        final Map<String, Long> decisionCategories = new TreeMap<>();
        final Map<String, Long> outcomeHistogram = new TreeMap<>();
        final Map<String, Long> throwableClasses = new TreeMap<>();

        private GameResult(long startNanos, long finishNanos) {
            this.startNanos = startNanos;
            this.finishNanos = finishNanos;
        }

        static GameResult classify(long startNanos, long finishNanos, Game game,
                                   ComputerPlayerUniformMirror p0,
                                   ComputerPlayerUniformMirror p1,
                                   Throwable thrown) {
            GameResult result = new GameResult(startNanos, finishNanos);
            if (thrown != null) {
                result.invalid("throwable");
                increment(result.throwableClasses, throwableClassLabel(thrown), 1L);
                if (thrown instanceof ComputerPlayerUniformMirror.UniformMirrorPolicyViolation) {
                    result.policyFailures++;
                    String internal = thrown.getMessage() == null
                            ? "" : thrown.getMessage().toLowerCase(Locale.ROOT);
                    if (internal.contains("canonical") || internal.contains("candidate")
                            || internal.contains("uuid") || internal.contains("card")
                            || internal.contains("trigger-order")) {
                        result.canonicalizationFailures++;
                    }
                }
                if (thrown instanceof AssertionError) {
                    result.modelFailures++;
                }
            }
            if (game == null) {
                result.invalid("game_null");
            } else {
                if (game.getState() == null) {
                    result.invalid("state_null");
                } else if (!game.getState().isGameOver()) {
                    result.invalid("not_game_over");
                }
                if (!game.hasEnded()) result.invalid("not_ended");
                if (game.getState() != null && game.isPaused()) result.invalid("paused");
                if (game.getTotalErrorsCount() != 0) result.invalid("engine_errors");
                result.turns = Math.max(0, game.getTurnNum());
            }
            if (p0 == null || p1 == null) {
                result.invalid("player_null");
            } else {
                if (game != null && !p0.getId().equals(game.getStartingPlayerId())) {
                    result.invalid("starting_player_mismatch");
                }
                if (p0.hasTimerTimeout() || p1.hasTimerTimeout()) result.invalid("timer_timeout");
                if (p0.hasIdleTimeout() || p1.hasIdleTimeout()) result.invalid("idle_timeout");
                if (p0.hasQuit() || p1.hasQuit()) result.invalid("quit");
                result.terminalLeftFlags = (p0.hasLeft() ? 1L : 0L) + (p1.hasLeft() ? 1L : 0L);
                SeededUniformMirrorPolicy p0Policy = p0.getMirrorPolicySnapshot();
                SeededUniformMirrorPolicy p1Policy = p1.getMirrorPolicySnapshot();
                result.physicalDecisions = p0Policy.getPhysicalDecisionCount()
                        + p1Policy.getPhysicalDecisionCount();
                result.policyActionSelections = p0Policy.getPolicyActionSelections()
                        + p1Policy.getPolicyActionSelections();
                result.policyLeafEvaluations = p0Policy.getPolicyLeafEvaluations()
                        + p1Policy.getPolicyLeafEvaluations();
                merge(result.decisionCategories, p0Policy.getPhysicalDecisionCategories());
                merge(result.decisionCategories, p1Policy.getPhysicalDecisionCategories());
                merge(result.outcomeHistogram, p0Policy.getOutcomeHistogram());
                merge(result.outcomeHistogram, p1Policy.getOutcomeHistogram());
                result.canonicalTies = p0.getCanonicalEncounterTieBreaks()
                        + p1.getCanonicalEncounterTieBreaks();
                result.forcedNoPolicySelections = p0.getForcedNoPolicySelections()
                        + p1.getForcedNoPolicySelections();
                result.explicitConcessionAttempts = p0.getExplicitConcedeAttempts()
                        + p1.getExplicitConcedeAttempts();
                if (result.explicitConcessionAttempts != 0L) {
                    result.invalid("explicit_concession_attempt");
                }
                if (result.canonicalTies != 0L) {
                    result.canonicalizationFailures += result.canonicalTies;
                    result.invalid("candidate_canonicalization_ties");
                }
                if (game != null) {
                    boolean gameDraw = game.isADraw();
                    boolean p0Win = p0.hasWon() && p1.hasLost()
                            && !p0.hasLost() && !p1.hasWon()
                            && !p0.hasDrew() && !p1.hasDrew() && !gameDraw;
                    boolean p1Win = p1.hasWon() && p0.hasLost()
                            && !p1.hasLost() && !p0.hasWon()
                            && !p0.hasDrew() && !p1.hasDrew() && !gameDraw;
                    boolean draw = gameDraw && !p0.hasWon() && !p1.hasWon()
                            && ((p0.hasLost() && p1.hasLost())
                            || (p0.hasDrew() && p1.hasDrew()));
                    int outcomeCount = (p0Win ? 1 : 0) + (p1Win ? 1 : 0) + (draw ? 1 : 0);
                    if (outcomeCount != 1) {
                        result.invalid("outcome_flags_inconsistent");
                    } else {
                        result.p0Win = p0Win;
                        result.p1Win = p1Win;
                        result.draw = draw;
                        // XMage's ordinary loss/draw cleanup calls
                        // lostForced/drew -> setConcedingPlayer -> leave, so a
                        // loser can have both hasLost and hasLeft at a natural
                        // terminal. A winner-left flag is never that cleanup.
                        if ((p0Win && p0.hasLeft()) || (p1Win && p1.hasLeft())) {
                            result.invalid("winning_player_left");
                        }
                    }
                }
            }
            result.natural = result.invalidClasses.isEmpty();
            return result;
        }

        private void invalid(String key) {
            increment(invalidClasses, key, 1L);
        }
    }

    private static void merge(Map<String, Long> target, Map<String, Long> source) {
        for (Map.Entry<String, Long> entry : source.entrySet()) {
            increment(target, entry.getKey(), entry.getValue());
        }
    }

    private static void increment(Map<String, Long> target, String key, long amount) {
        Long old = target.get(key);
        target.put(key, old == null ? amount : old + amount);
    }

    private static void addUnique(List<String> values, String value) {
        if (!values.contains(value)) {
            values.add(value);
        }
    }
}
