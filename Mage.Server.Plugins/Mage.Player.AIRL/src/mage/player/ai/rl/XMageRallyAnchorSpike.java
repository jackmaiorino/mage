package mage.player.ai.rl;

import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.cards.repository.CardScanner;
import mage.collectors.DataCollectorServices;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerMatch;
import mage.game.match.MatchOptions;
import mage.game.mulligan.MulliganType;
import mage.player.ai.ComputerPlayerUniformMirror;
import mage.players.Player;
import mage.util.RandomUtil;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Fast engineering spike for pinned checkpoints on the XMage Rally surface.
 *
 * <p>This is a paired external-anchor harness, not by itself a statistical
 * claim. It keeps one exact Rust checkpoint process alive, swaps the candidate
 * seat, and aborts immediately on any deck, decision, semantic, or terminal
 * mismatch. The noncandidate seat can use either the seeded-uniform mirror or
 * deterministic XMage CP7.</p>
 */
public final class XMageRallyAnchorSpike {

    public static final String DECK_RELATIVE_PATH =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono Red Rally.dek";
    public static final String DECK_SHA256 =
            "4b5019bd08f9387aeabebdca0d90aaa10dfd75fc75ed3a87c95a2fabf4dba834";
    public static final String DECK_SHA256_LF =
            "c6994cc1be913b15fec456d7baa8af6c049ef099463856344dea35b195a927a4";
    private static final long BRIDGE_TIMEOUT_MILLIS = 10L * 60_000L;

    private static final Map<String, Integer> RALLY_CARD_IDS = rallyCardIds();

    private XMageRallyAnchorSpike() {
    }

    public static void main(String[] rawArgs) throws Exception {
        quietLogging();
        Args args = Args.parse(rawArgs);
        Path deckPath = args.repoRoot.resolve(DECK_RELATIVE_PATH).normalize().toRealPath();
        if (!deckPath.startsWith(args.repoRoot)) {
            throw new IllegalStateException("Rally deck escaped repository root");
        }
        String deckSha = sha256(deckPath);
        // Git materializes this XML as CRLF in the certified Windows checkout
        // and LF in this WSL worktree. The parsed 75-card deck is identical.
        if (!DECK_SHA256.equals(deckSha) && !DECK_SHA256_LF.equals(deckSha)) {
            throw new IllegalStateException("Rally deck SHA-256 mismatch: " + deckSha);
        }

        DataCollectorServices.init(false, false);
        List<String> scannerErrors = new ArrayList<>();
        CardScanner.scan(scannerErrors);
        quietLogging();
        if (!scannerErrors.isEmpty()) {
            throw new IllegalStateException("card scanner errors: " + scannerErrors);
        }
        DeckTemplates decks = DeckTemplates.load(deckPath);

        List<String> command = new ArrayList<>();
        command.add(args.scorerExecutable.toString());
        if (args.outcomeRoot != null) {
            command.add("--xmage-cp7-outcome-root");
            command.add(args.outcomeRoot.toString());
        } else if (args.behaviorCloneRoot != null) {
            command.add("--cp7-behavior-clone-root");
            command.add(args.behaviorCloneRoot.toString());
        } else if (args.populationStoreRoot != null) {
            command.add("--population-store-root");
            command.add(args.populationStoreRoot.toString());
            command.add("--generation");
            command.add(Long.toString(args.checkpointGeneration));
        } else {
            command.add("--original-store-root");
            command.add(args.storeRoot.toString());
            if (args.checkpointGeneration != null) {
                command.add("--generation");
                command.add(Long.toString(args.checkpointGeneration));
            }
        }
        if (args.teacherExportPath != null) {
            command.add("--xmage-cp7-teacher-jsonl");
            command.add(args.teacherExportPath.toString());
        }
        if (args.outcomeExportPath != null) {
            command.add("--xmage-cp7-outcome-jsonl");
            command.add(args.outcomeExportPath.toString());
        }
        long sampleStart = System.nanoTime();
        try (RallyCp7CounterfactualTeacher counterfactualTeacher =
                     args.shadowCp7ExportPath == null ? null
                             : RallyCp7CounterfactualTeacher.create(
                             args.shadowCp7ExportPath,
                             args.cp7Skill,
                             args.shadowCp7MaxThinkSeconds);
             XMageRallyBridgeProcessClient bridge =
                     XMageRallyBridgeProcessClient.start(
                             command,
                             BRIDGE_TIMEOUT_MILLIS,
                             XMageRallyBridgeProcessClient.DEFAULT_MAX_LINE_BYTES,
                             System.err)) {
            int candidateWins = 0;
            int opponentWins = 0;
            int draws = 0;
            int onPlayWins = 0;
            int onPlayLosses = 0;
            int onPlayDraws = 0;
            int onDrawWins = 0;
            int onDrawLosses = 0;
            int onDrawDraws = 0;
            int candidateSweeps = 0;
            int opponentSweeps = 0;
            int splitPairs = 0;
            int drawAffectedPairs = 0;
            long totalTurns = 0L;
            long totalRustSteps = 0L;
            long totalPhysicalDecisions = 0L;
            long totalCandidatePriorityProjections = 0L;
            long totalCp7Steps = 0L;
            long totalCp7PhysicalDecisions = 0L;
            long totalCp7ForcedEvents = 0L;
            Map<String, Long> totalCp7Kinds = new HashMap<>();
            for (int pairOrdinal = 0; pairOrdinal < args.pairCount; pairOrdinal++) {
                long firstEpisode = Math.addExact(
                        args.firstEpisodeId, Math.multiplyExact(2L, pairOrdinal));
                long pairStart = System.nanoTime();
                LegResult first = runLeg(
                        bridge, decks, args, firstEpisode, counterfactualTeacher);
                LegResult second = runLeg(
                        bridge, decks, args, firstEpisode + 1L, counterfactualTeacher);
                if (first.pairIndex != second.pairIndex
                        || first.environmentSeed != second.environmentSeed) {
                    throw new IllegalStateException(
                            "paired episodes did not share environment seed");
                }
                if (first.candidateSeat == second.candidateSeat) {
                    throw new IllegalStateException(
                            "paired episodes did not swap candidate seat");
                }
                for (LegResult leg : Arrays.asList(first, second)) {
                    if ("draw".equals(leg.winner)) {
                        draws++;
                        if (leg.candidateSeat == XMageRallyBridgeProtocol.Seat.P0) {
                            onPlayDraws++;
                        } else {
                            onDrawDraws++;
                        }
                    } else if (leg.winner.equals(leg.candidateSeat.wire())) {
                        candidateWins++;
                        if (leg.candidateSeat == XMageRallyBridgeProtocol.Seat.P0) {
                            onPlayWins++;
                        } else {
                            onDrawWins++;
                        }
                    } else {
                        opponentWins++;
                        if (leg.candidateSeat == XMageRallyBridgeProtocol.Seat.P0) {
                            onPlayLosses++;
                        } else {
                            onDrawLosses++;
                        }
                    }
                    totalTurns = Math.addExact(totalTurns, leg.turns);
                    totalRustSteps = Math.addExact(totalRustSteps, leg.rustSteps);
                    totalPhysicalDecisions = Math.addExact(
                            totalPhysicalDecisions, leg.physicalDecisions);
                    totalCandidatePriorityProjections = Math.addExact(
                            totalCandidatePriorityProjections,
                            leg.candidatePriorityProjections);
                    totalCp7Steps = Math.addExact(totalCp7Steps, leg.cp7Steps);
                    totalCp7PhysicalDecisions = Math.addExact(
                            totalCp7PhysicalDecisions, leg.cp7PhysicalDecisions);
                    totalCp7ForcedEvents = Math.addExact(
                            totalCp7ForcedEvents, leg.cp7ForcedEvents);
                    mergeCounts(totalCp7Kinds, leg.cp7Kinds);
                }
                int pairCandidateWins = candidateWin(first) + candidateWin(second);
                int pairOpponentWins = opponentWin(first) + opponentWin(second);
                if (pairCandidateWins == 2) {
                    candidateSweeps++;
                } else if (pairOpponentWins == 2) {
                    opponentSweeps++;
                } else if (pairCandidateWins == 1 && pairOpponentWins == 1) {
                    splitPairs++;
                } else {
                    drawAffectedPairs++;
                }
                long pairElapsedMillis =
                        (System.nanoTime() - pairStart) / 1_000_000L;
                System.out.println("XMAGE_RALLY_ANCHOR_PAIR PASS"
                        + " base_seed=" + args.baseSeed
                        + " opponent=" + args.opponentMode.wire
                        + " cp7_skill=" + args.cp7Skill
                        + " episodes=" + first.episodeId + "," + second.episodeId
                        + " pair_index=" + first.pairIndex
                        + " environment_seed=" + unsignedHex(first.environmentSeed)
                        + " candidate_seats=" + first.candidateSeat.wire()
                        + "," + second.candidateSeat.wire()
                        + " winners=" + first.winner + "," + second.winner
                        + " turns=" + first.turns + "," + second.turns
                        + " rust_steps=" + first.rustSteps + "," + second.rustSteps
                        + " physical_decisions=" + first.physicalDecisions
                        + "," + second.physicalDecisions
                        + " candidate_priority_projections="
                        + first.candidatePriorityProjections + ","
                        + second.candidatePriorityProjections
                        + " alignment="
                        + alignment(Math.addExact(
                        first.candidatePriorityProjections,
                        second.candidatePriorityProjections))
                        + " elapsed_ms=" + pairElapsedMillis);
            }
            long elapsedMillis = (System.nanoTime() - sampleStart) / 1_000_000L;
            int games = Math.multiplyExact(args.pairCount, 2);
            System.out.println("XMAGE_RALLY_ANCHOR_SPIKE PASS"
                    + " base_seed=" + args.baseSeed
                    + " checkpoint_generation="
                    + (args.outcomeRoot != null
                    ? "xmage_cp7_outcome_reinforce"
                    : (args.behaviorCloneRoot != null
                    ? "cp7_behavior_clone"
                    : (args.populationStoreRoot != null
                    ? "population_store_" + args.checkpointGeneration
                    : (args.checkpointGeneration == null
                    ? "default" : args.checkpointGeneration))))
                    + " opponent=" + args.opponentMode.wire
                    + " cp7_skill=" + args.cp7Skill
                    + " first_episode=" + args.firstEpisodeId
                    + " pairs=" + args.pairCount
                    + " games=" + games
                    + " candidate_wins=" + candidateWins
                    + " opponent_wins=" + opponentWins
                    + " draws=" + draws
                    + " score=" + String.format(Locale.ROOT, "%.6f",
                    (candidateWins + 0.5d * draws) / games)
                    + " on_play=" + onPlayWins + "-" + onPlayLosses
                    + "-" + onPlayDraws
                    + " on_draw=" + onDrawWins + "-" + onDrawLosses
                    + "-" + onDrawDraws
                    + " candidate_sweeps=" + candidateSweeps
                    + " opponent_sweeps=" + opponentSweeps
                    + " split_pairs=" + splitPairs
                    + " draw_affected_pairs=" + drawAffectedPairs
                    + " total_turns=" + totalTurns
                    + " total_rust_steps=" + totalRustSteps
                    + " total_physical_decisions=" + totalPhysicalDecisions
                    + " total_candidate_priority_projections="
                    + totalCandidatePriorityProjections
                    + " alignment=" + alignment(totalCandidatePriorityProjections)
                    + " total_cp7_steps=" + totalCp7Steps
                    + " total_cp7_physical_decisions=" + totalCp7PhysicalDecisions
                    + " total_cp7_forced_events=" + totalCp7ForcedEvents
                    + " total_cp7_kinds=" + countsWire(totalCp7Kinds)
                    + " elapsed_ms=" + elapsedMillis);
        }
    }

    private static LegResult runLeg(XMageRallyBridgeProcessClient bridge,
                                    DeckTemplates decks,
                                    Args args,
                                    long episodeId,
                                    RallyCp7CounterfactualTeacher counterfactualTeacher)
            throws Exception {
        AtomicReference<LegResult> result = new AtomicReference<>();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread gameThread = new Thread(() -> {
            try {
                result.set(runLegInGameThread(
                        bridge, decks, args, episodeId, counterfactualTeacher));
            } catch (Throwable error) {
                failure.set(error);
            }
        }, "GAME-XMAGE-RALLY-ANCHOR-e" + episodeId);
        // A wedged XMage search must not keep the harness JVM alive after the
        // per-leg watchdog fails closed.
        gameThread.setDaemon(true);
        gameThread.start();
        try {
            gameThread.join(BRIDGE_TIMEOUT_MILLIS);
        } catch (InterruptedException error) {
            bridge.close();
            gameThread.interrupt();
            Thread.currentThread().interrupt();
            throw error;
        }
        if (gameThread.isAlive()) {
            bridge.close();
            gameThread.interrupt();
            throw new IllegalStateException(
                    "XMage Rally leg " + episodeId + " timed out after "
                            + BRIDGE_TIMEOUT_MILLIS + " ms");
        }
        Throwable error = failure.get();
        if (error instanceof Exception) {
            throw (Exception) error;
        }
        if (error instanceof Error) {
            throw (Error) error;
        }
        if (error != null) {
            throw new IllegalStateException("game thread failed", error);
        }
        if (result.get() == null) {
            throw new IllegalStateException("game thread returned no result");
        }
        return result.get();
    }

    private static LegResult runLegInGameThread(
            XMageRallyBridgeProcessClient bridge,
            DeckTemplates decks,
            Args args,
            long episodeId,
            RallyCp7CounterfactualTeacher counterfactualTeacher) throws Exception {
        long start = System.nanoTime();
        bridge.reset("anchor-reset-" + episodeId, episodeId, args.baseSeed);
        XMageRallyBridgeProtocol.DecisionBody resetDecision = bridge.getCurrentDecision();
        if (resetDecision == null) {
            throw new IllegalStateException("Rally reset unexpectedly returned terminal");
        }
        long environmentSeed = Long.parseUnsignedLong(
                bridge.getPairEnvironmentSeedU64Hex(), 16);
        List<List<Integer>> expectedLibraries =
                bridge.getInitialLibraryCardDefinitionIds();
        if (expectedLibraries == null || expectedLibraries.size() != 2) {
            throw new IllegalStateException("reset omitted the two initial Rally libraries");
        }

        XMageRallyBridgeProtocol.Seat candidateSeat = bridge.getCandidateSeat();
        KernelShadowRallyPolicy p0Policy = null;
        KernelShadowRallyPolicy p1Policy = null;
        RallyCp7KernelShadowMapper cp7Mapper = null;
        Player p0 = null;
        Player p1 = null;
        Game game = null;
        try (RandomUtil.RandomIsolation ignored =
                     RandomUtil.isolateThreadLocalRandom(environmentSeed)) {
            // Keep the same RNG boundary as the certified uniform benchmark:
            // deck copies and all game setup are part of the isolated episode.
            Deck p0Deck = decks.p0.copy();
            Deck p1Deck = decks.p1.copy();
            p0Deck.getSideboard().clear();
            p1Deck.getSideboard().clear();
            verifyDisjointCardIds(p0Deck, p1Deck);
            MutableSplitMix64 shuffle = new MutableSplitMix64(environmentSeed);
            shuffleDeck(p0Deck, shuffle);
            shuffleDeck(p1Deck, shuffle);
            verifyLibraryIds("p0", p0Deck, expectedLibraries.get(0));
            verifyLibraryIds("p1", p1Deck, expectedLibraries.get(1));
            Map<UUID, Integer> initialArenaIds = initialArenaIds(p0Deck, p1Deck);

            if (args.opponentMode == OpponentMode.UNIFORM) {
                p0Policy = policyForSeat(
                        bridge, args.baseSeed, episodeId,
                        XMageRallyBridgeProtocol.Seat.P0,
                        candidateSeat == XMageRallyBridgeProtocol.Seat.P0,
                        initialArenaIds, counterfactualTeacher);
                p1Policy = policyForSeat(
                        bridge, args.baseSeed, episodeId,
                        XMageRallyBridgeProtocol.Seat.P1,
                        candidateSeat == XMageRallyBridgeProtocol.Seat.P1,
                        initialArenaIds, counterfactualTeacher);
            } else {
                XMageRallyBridgeProtocol.Seat cp7Seat =
                        candidateSeat == XMageRallyBridgeProtocol.Seat.P0
                                ? XMageRallyBridgeProtocol.Seat.P1
                                : XMageRallyBridgeProtocol.Seat.P0;
                if (candidateSeat == XMageRallyBridgeProtocol.Seat.P0) {
                    p0Policy = policyForSeat(
                            bridge, args.baseSeed, episodeId,
                            XMageRallyBridgeProtocol.Seat.P0, true, initialArenaIds,
                            counterfactualTeacher);
                } else {
                    p1Policy = policyForSeat(
                            bridge, args.baseSeed, episodeId,
                            XMageRallyBridgeProtocol.Seat.P1, true, initialArenaIds,
                            counterfactualTeacher);
                }
                String cp7Name = "cp7-" + cp7Seat.wire() + "-e" + episodeId;
                cp7Mapper = new RallyCp7KernelShadowMapper(
                        bridge, episodeId, cp7Seat, cp7Name, initialArenaIds);
            }
            MatchOptions matchOptions = fixedMatchOptions();
            TwoPlayerMatch match = new TwoPlayerMatch(matchOptions);
            match.startGame();
            game = match.getGames().get(0);
            String suffix = "-e" + episodeId;
            if (args.opponentMode == OpponentMode.UNIFORM) {
                p0 = new ComputerPlayerUniformMirror(
                        "shadow-p0" + suffix, RangeOfInfluence.ALL, p0Policy, "p0");
                p1 = new ComputerPlayerUniformMirror(
                        "shadow-p1" + suffix, RangeOfInfluence.ALL, p1Policy, "p1");
            } else if (candidateSeat == XMageRallyBridgeProtocol.Seat.P0) {
                p0 = new ComputerPlayerUniformMirror(
                        "candidate-p0" + suffix, RangeOfInfluence.ALL, p0Policy, "p0");
                RallyCp7ObservedPlayer cp7 = new RallyCp7ObservedPlayer(
                        "cp7-p1" + suffix, RangeOfInfluence.ALL,
                        args.cp7Skill, cp7Mapper);
                cp7Mapper.bindPlayer(cp7);
                p1 = cp7;
            } else {
                RallyCp7ObservedPlayer cp7 = new RallyCp7ObservedPlayer(
                        "cp7-p0" + suffix, RangeOfInfluence.ALL,
                        args.cp7Skill, cp7Mapper);
                cp7Mapper.bindPlayer(cp7);
                p0 = cp7;
                p1 = new ComputerPlayerUniformMirror(
                        "candidate-p1" + suffix, RangeOfInfluence.ALL, p1Policy, "p1");
            }
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
                throw new IllegalStateException("XMage rejected the fixed p0 starter");
            }
            game.start(p0.getId());
        }

        requireNaturalTerminal(game, p0, p1, p0Policy, p1Policy, cp7Mapper);
        XMageRallyBridgeProtocol.TerminalBody nativeTerminal = bridge.getTerminal();
        if (nativeTerminal == null) {
            throw new IllegalStateException("XMage ended before the native Rally episode");
        }
        requireNaturalNativeTerminal(
                nativeTerminal, p0Policy, p1Policy, cp7Mapper);
        String winner = xmageWinner(game, p0, p1);
        String nativeWinner = nativeTerminal.getTerminal().getWinner() == null
                ? "draw" : nativeTerminal.getTerminal().getWinner().wire();
        if (!winner.equals(nativeWinner)) {
            throw new IllegalStateException("terminal winner mismatch: XMage="
                    + winner + " Rust=" + nativeWinner);
        }
        long rustSteps = nativeTerminal.getTerminal().getPolicyStepCount();
        long physicalDecisions =
                nativeTerminal.getTerminal().getPhysicalDecisionCount();
        KernelShadowRallyPolicy candidatePolicy =
                candidateSeat == XMageRallyBridgeProtocol.Seat.P0
                        ? p0Policy : p1Policy;
        if (candidatePolicy == null) {
            throw new IllegalStateException("candidate policy is missing at terminal");
        }
        long candidatePriorityProjections =
                candidatePolicy.getSelectedPriorityProjectionCount();
        long cp7Steps = cp7Mapper == null ? 0L : cp7Mapper.getAppliedPolicySteps();
        long cp7PhysicalDecisions = cp7Mapper == null
                ? 0L : cp7Mapper.getAppliedPhysicalDecisionCount();
        long cp7ForcedEvents = cp7Mapper == null
                ? 0L : cp7Mapper.getForcedNoPolicyEvents();
        Map<String, Long> cp7Kinds = cp7Mapper == null
                ? Collections.emptyMap() : cp7Mapper.getAppliedKinds();
        long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;
        System.out.println("XMAGE_RALLY_ANCHOR_LEG PASS"
                + " episode=" + episodeId
                + " opponent=" + args.opponentMode.wire
                + " cp7_skill=" + args.cp7Skill
                + " candidate=" + candidateSeat.wire()
                + " winner=" + winner
                + " turns=" + game.getTurnNum()
                + " rust_steps=" + rustSteps
                + " physical_decisions=" + physicalDecisions
                + " candidate_priority_projections="
                + candidatePriorityProjections
                + " alignment=" + alignment(candidatePriorityProjections)
                + " cp7_steps=" + cp7Steps
                + " cp7_physical_decisions=" + cp7PhysicalDecisions
                + " cp7_forced_events=" + cp7ForcedEvents
                + " cp7_kinds=" + countsWire(cp7Kinds)
                + " environment_seed=" + unsignedHex(environmentSeed)
                + " elapsed_ms=" + elapsedMillis);
        return new LegResult(
                episodeId,
                resetDecision.getPairIndex(),
                environmentSeed,
                candidateSeat,
                winner,
                game.getTurnNum(),
                rustSteps,
                physicalDecisions,
                candidatePriorityProjections,
                cp7Steps,
                cp7PhysicalDecisions,
                cp7ForcedEvents,
                cp7Kinds);
    }

    private static int candidateWin(LegResult leg) {
        return leg.winner.equals(leg.candidateSeat.wire()) ? 1 : 0;
    }

    private static int opponentWin(LegResult leg) {
        return !"draw".equals(leg.winner)
                && !leg.winner.equals(leg.candidateSeat.wire()) ? 1 : 0;
    }

    private static String alignment(long candidatePriorityProjections) {
        return candidatePriorityProjections == 0L
                ? "no_selected_action_projection"
                : "selected_action_projection";
    }

    private static void mergeCounts(
            Map<String, Long> destination,
            Map<String, Long> source) {
        for (Map.Entry<String, Long> entry : source.entrySet()) {
            destination.put(entry.getKey(), Math.addExact(
                    destination.getOrDefault(entry.getKey(), 0L), entry.getValue()));
        }
    }

    private static String countsWire(Map<String, Long> counts) {
        if (counts == null || counts.isEmpty()) {
            return "none";
        }
        List<String> keys = new ArrayList<>(counts.keySet());
        Collections.sort(keys);
        List<String> rows = new ArrayList<>(keys.size());
        for (String key : keys) {
            rows.add(key + "=" + counts.get(key));
        }
        return String.join(",", rows);
    }

    private static KernelShadowRallyPolicy policyForSeat(
            XMageRallyBridgeProcessClient bridge,
            long baseSeed,
            long episodeId,
            XMageRallyBridgeProtocol.Seat seat,
            boolean modelControlled,
            Map<UUID, Integer> initialArenaIds,
            RallyCp7CounterfactualTeacher counterfactualTeacher) {
        String wire = seat.wire();
        SeededUniformMirrorPolicy simulation =
                new SeededUniformMirrorPolicy(baseSeed, episodeId, wire);
        RallyCanonicalDecisionPolicy delegate = modelControlled
                ? null : new SeededUniformMirrorPolicy(baseSeed, episodeId, wire);
        return new KernelShadowRallyPolicy(
                bridge, episodeId, seat, modelControlled, delegate, simulation,
                initialArenaIds, modelControlled ? counterfactualTeacher : null);
    }

    private static void requireNaturalTerminal(
            Game game,
            Player p0,
            Player p1,
            KernelShadowRallyPolicy p0Policy,
            KernelShadowRallyPolicy p1Policy,
            RallyCp7KernelShadowMapper cp7Mapper) {
        if (game == null || game.getState() == null
                || !game.getState().isGameOver() || !game.hasEnded()) {
            throw new IllegalStateException("XMage did not reach a natural terminal");
        }
        if (game.isPaused() || game.getTotalErrorsCount() != 0) {
            throw new IllegalStateException("XMage terminal contains engine errors or pause");
        }
        if (p0 == null || p1 == null || p0.hasTimerTimeout() || p1.hasTimerTimeout()
                || p0.hasIdleTimeout() || p1.hasIdleTimeout()
                || p0.hasQuit() || p1.hasQuit()) {
            throw new IllegalStateException("XMage player terminal flags are invalid");
        }
        if (p0Policy != null && p0Policy.isFailed()
                || p1Policy != null && p1Policy.isFailed()) {
            throw new IllegalStateException("shadow policy failed: p0="
                    + (p0Policy == null ? "none" : p0Policy.getFirstFailure())
                    + " p1="
                    + (p1Policy == null ? "none" : p1Policy.getFirstFailure()));
        }
        if (cp7Mapper != null && cp7Mapper.isFailed()) {
            throw new IllegalStateException(
                    "CP7 shadow mapper failed: " + cp7Mapper.getFirstFailure());
        }
        xmageWinner(game, p0, p1);
    }

    private static void requireNaturalNativeTerminal(
            XMageRallyBridgeProtocol.TerminalBody nativeTerminal,
            KernelShadowRallyPolicy p0Policy,
            KernelShadowRallyPolicy p1Policy,
            RallyCp7KernelShadowMapper cp7Mapper) {
        XMageRallyBridgeProtocol.TerminalRecord terminal = nativeTerminal.getTerminal();
        if (!"natural".equals(terminal.getTerminalClassification())
                || !"natural_game_over".equals(terminal.getTerminalCode())) {
            throw new IllegalStateException("native Rally episode did not end naturally: "
                    + terminal.getTerminalClassification() + "/" + terminal.getTerminalCode());
        }
        long xmageSteps = 0L;
        long xmagePhysicalDecisions = 0L;
        for (KernelShadowRallyPolicy policy : Arrays.asList(p0Policy, p1Policy)) {
            if (policy != null) {
                xmageSteps = Math.addExact(
                        xmageSteps, policy.getPolicyActionSelections());
                xmagePhysicalDecisions = Math.addExact(
                        xmagePhysicalDecisions, policy.getPhysicalDecisionCount());
            }
        }
        if (cp7Mapper != null) {
            xmageSteps = Math.addExact(
                    xmageSteps, cp7Mapper.getAppliedPolicySteps());
            xmagePhysicalDecisions = Math.addExact(
                    xmagePhysicalDecisions,
                    cp7Mapper.getAppliedPhysicalDecisionCount());
        }
        if (terminal.getPolicyStepCount() != xmageSteps
                || terminal.getPhysicalDecisionCount() != xmagePhysicalDecisions) {
            throw new IllegalStateException("native/XMage policy counts differ: steps="
                    + terminal.getPolicyStepCount() + "/" + xmageSteps
                    + " physical=" + terminal.getPhysicalDecisionCount()
                    + "/" + xmagePhysicalDecisions);
        }
    }

    private static String xmageWinner(
            Game game, Player p0, Player p1) {
        boolean draw = game.isADraw() && !p0.hasWon() && !p1.hasWon();
        boolean p0Win = p0.hasWon() && p1.hasLost() && !p0.hasLost() && !p1.hasWon();
        boolean p1Win = p1.hasWon() && p0.hasLost() && !p1.hasLost() && !p0.hasWon();
        int count = (draw ? 1 : 0) + (p0Win ? 1 : 0) + (p1Win ? 1 : 0);
        if (count != 1) {
            throw new IllegalStateException("XMage terminal outcome flags are inconsistent");
        }
        return draw ? "draw" : p0Win ? "p0" : "p1";
    }

    private static MatchOptions fixedMatchOptions() {
        MatchOptions options = new MatchOptions(
                "XMageRallyAnchorSpike", "TwoPlayerDuel", false);
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
            throw new IllegalStateException("forced library must contain 60 cards");
        }
        player.getLibrary().clear();
        player.getLibrary().addAll(ordered, game);
        if (player.getLibrary().size() != 60) {
            throw new IllegalStateException("XMage library size mismatch");
        }
    }

    private static void shuffleDeck(Deck deck, MutableSplitMix64 random) {
        List<Card> cards = new ArrayList<>(deck.getCards());
        for (int i = cards.size() - 1; i > 0; i--) {
            int j = SeededUniformMirrorPolicy.unsignedModulo(random.next(), i + 1);
            Collections.swap(cards, i, j);
        }
        deck.getCards().clear();
        deck.getCards().addAll(cards);
    }

    private static void verifyLibraryIds(
            String seat, Deck deck, List<Integer> expected) {
        if (expected == null || expected.size() != 60 || deck.getCards().size() != 60) {
            throw new IllegalStateException(seat + " initial library width mismatch");
        }
        int index = 0;
        for (Card card : deck.getCards()) {
            Integer observed = RALLY_CARD_IDS.get(card.getName());
            if (observed == null) {
                throw new IllegalStateException("unmapped Rally card: " + card.getName());
            }
            if (!observed.equals(expected.get(index))) {
                throw new IllegalStateException(seat + " initial library mismatch at " + index
                        + ": XMage=" + observed + " Rust=" + expected.get(index));
            }
            index++;
        }
    }

    private static void verifyDisjointCardIds(Deck p0, Deck p1) {
        Set<java.util.UUID> ids = new HashSet<>();
        for (Card card : p0.getCards()) {
            ids.add(card.getId());
        }
        for (Card card : p1.getCards()) {
            if (!ids.add(card.getId())) {
                throw new IllegalStateException("seat decks share a card UUID");
            }
        }
    }

    private static Map<UUID, Integer> initialArenaIds(Deck p0, Deck p1) {
        Map<UUID, Integer> bindings = new HashMap<>();
        int arenaId = 0;
        for (Card card : p0.getCards()) {
            if (card == null || bindings.put(card.getId(), arenaId) != null) {
                throw new IllegalStateException("invalid p0 initial card identity");
            }
            arenaId++;
        }
        for (Card card : p1.getCards()) {
            if (card == null || bindings.put(card.getId(), arenaId) != null) {
                throw new IllegalStateException("invalid p1 initial card identity");
            }
            arenaId++;
        }
        if (arenaId != 120 || bindings.size() != 120) {
            throw new IllegalStateException("initial arena binding must cover 120 cards");
        }
        return Collections.unmodifiableMap(bindings);
    }

    private static Map<String, Integer> rallyCardIds() {
        Map<String, Integer> ids = new HashMap<>();
        ids.put("Clockwork Percussionist", 16);
        ids.put("Voldaren Epicure", 127);
        ids.put("Goblin Bushwhacker", 44);
        ids.put("Goblin Tomb Raider", 45);
        ids.put("Burning-Tree Emissary", 10);
        ids.put("Galvanic Blast", 41);
        ids.put("Experimental Synthesizer", 30);
        ids.put("Lightning Bolt", 66);
        ids.put("Reckless Impulse", 93);
        ids.put("Rally at the Hornburg", 92);
        ids.put("Great Furnace", 48);
        ids.put("Mountain", 76);
        ids.put("Chain Lightning", 13);
        ids.put("End the Festivities", 27);
        return Collections.unmodifiableMap(ids);
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        digest.update(Files.readAllBytes(path));
        StringBuilder out = new StringBuilder(64);
        for (byte value : digest.digest()) {
            out.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return out.toString();
    }

    private static String unsignedHex(long value) {
        return String.format(Locale.ROOT, "%016x", value);
    }

    private static void quietLogging() {
        LogManager.getRootLogger().setLevel(Level.OFF);
        @SuppressWarnings("unchecked")
        java.util.Enumeration<Logger> loggers = LogManager.getCurrentLoggers();
        while (loggers.hasMoreElements()) {
            loggers.nextElement().setLevel(Level.OFF);
        }
    }

    private static final class DeckTemplates {
        final Deck p0;
        final Deck p1;

        DeckTemplates(Deck p0, Deck p1) {
            this.p0 = p0;
            this.p1 = p1;
        }

        static DeckTemplates load(Path deckPath) throws Exception {
            StringBuilder warnings = new StringBuilder();
            DeckCardLists lists = DeckImporter.importDeckFromFile(
                    deckPath.toString(), warnings, false);
            if (lists == null || warnings.length() != 0) {
                throw new IllegalStateException("Rally deck import failed: " + warnings);
            }
            Deck p0 = Deck.load(lists, false, false, null);
            Deck p1 = Deck.load(lists, false, false, null);
            if (p0.getCards().size() != 60 || p1.getCards().size() != 60) {
                throw new IllegalStateException("Rally main deck is not 60 cards");
            }
            if (p0.getSideboard().size() != 15 || p1.getSideboard().size() != 15) {
                throw new IllegalStateException("Rally source sideboard is not 15 cards");
            }
            verifyDisjointCardIds(p0, p1);
            return new DeckTemplates(p0, p1);
        }
    }

    private static final class MutableSplitMix64 {
        private long state;

        MutableSplitMix64(long seed) {
            state = seed;
        }

        long next() {
            state += SeededUniformMirrorPolicy.GOLDEN_RATIO_64;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58_476D_1CE4_E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D0_49BB_1331_11EBL;
            return z ^ (z >>> 31);
        }
    }

    private static final class LegResult {
        final long episodeId;
        final long pairIndex;
        final long environmentSeed;
        final XMageRallyBridgeProtocol.Seat candidateSeat;
        final String winner;
        final int turns;
        final long rustSteps;
        final long physicalDecisions;
        final long candidatePriorityProjections;
        final long cp7Steps;
        final long cp7PhysicalDecisions;
        final long cp7ForcedEvents;
        final Map<String, Long> cp7Kinds;

        LegResult(long episodeId,
                  long pairIndex,
                  long environmentSeed,
                  XMageRallyBridgeProtocol.Seat candidateSeat,
                  String winner,
                  int turns,
                  long rustSteps,
                  long physicalDecisions,
                  long candidatePriorityProjections,
                  long cp7Steps,
                  long cp7PhysicalDecisions,
                  long cp7ForcedEvents,
                  Map<String, Long> cp7Kinds) {
            this.episodeId = episodeId;
            this.pairIndex = pairIndex;
            this.environmentSeed = environmentSeed;
            this.candidateSeat = candidateSeat;
            this.winner = winner;
            this.turns = turns;
            this.rustSteps = rustSteps;
            this.physicalDecisions = physicalDecisions;
            this.candidatePriorityProjections = candidatePriorityProjections;
            this.cp7Steps = cp7Steps;
            this.cp7PhysicalDecisions = cp7PhysicalDecisions;
            this.cp7ForcedEvents = cp7ForcedEvents;
            this.cp7Kinds = Collections.unmodifiableMap(
                    new HashMap<>(cp7Kinds));
        }
    }

    private static final class Args {
        final Path repoRoot;
        final Path scorerExecutable;
        final Path storeRoot;
        final Path populationStoreRoot;
        final long baseSeed;
        final long firstEpisodeId;
        final int pairCount;
        final OpponentMode opponentMode;
        final int cp7Skill;
        final Path teacherExportPath;
        final Path outcomeExportPath;
        final Path shadowCp7ExportPath;
        final int shadowCp7MaxThinkSeconds;
        final Long checkpointGeneration;
        final Path behaviorCloneRoot;
        final Path outcomeRoot;

        Args(Path repoRoot,
             Path scorerExecutable,
             Path storeRoot,
             Path populationStoreRoot,
             long baseSeed,
             long firstEpisodeId,
             int pairCount,
             OpponentMode opponentMode,
             int cp7Skill,
             Path teacherExportPath,
             Path outcomeExportPath,
             Path shadowCp7ExportPath,
             int shadowCp7MaxThinkSeconds,
             Long checkpointGeneration,
             Path behaviorCloneRoot,
             Path outcomeRoot) throws Exception {
            this.repoRoot = repoRoot.toRealPath();
            this.scorerExecutable = scorerExecutable.toRealPath();
            this.storeRoot = storeRoot == null ? null : storeRoot.toRealPath();
            this.populationStoreRoot = populationStoreRoot == null
                    ? null : populationStoreRoot.toRealPath();
            this.baseSeed = baseSeed;
            this.firstEpisodeId = firstEpisodeId;
            this.pairCount = pairCount;
            this.opponentMode = opponentMode;
            this.cp7Skill = cp7Skill;
            this.teacherExportPath = teacherExportPath;
            this.outcomeExportPath = outcomeExportPath;
            this.shadowCp7ExportPath = shadowCp7ExportPath;
            this.shadowCp7MaxThinkSeconds = shadowCp7MaxThinkSeconds;
            this.checkpointGeneration = checkpointGeneration;
            this.behaviorCloneRoot = behaviorCloneRoot == null
                    ? null : behaviorCloneRoot.toRealPath();
            this.outcomeRoot = outcomeRoot == null ? null : outcomeRoot.toRealPath();
        }

        static Args parse(String[] raw) throws Exception {
            Map<String, String> values = new HashMap<>();
            for (int i = 0; i < raw.length; i += 2) {
                if (i + 1 >= raw.length || !raw[i].startsWith("--")) {
                    throw new IllegalArgumentException("arguments must be --name value pairs");
                }
                if (values.put(raw[i], raw[i + 1]) != null) {
                    throw new IllegalArgumentException("duplicate argument: " + raw[i]);
                }
            }
            Set<String> required = new HashSet<>(Arrays.asList(
                    "--repo-root", "--scorer-exe",
                    "--base-seed", "--first-episode"));
            Set<String> allowed = new HashSet<>(required);
            allowed.add("--store-root");
            allowed.add("--population-store-root");
            allowed.add("--behavior-clone-root");
            allowed.add("--outcome-root");
            allowed.add("--pairs");
            allowed.add("--opponent");
            allowed.add("--cp7-skill");
            allowed.add("--teacher-export");
            allowed.add("--outcome-export");
            allowed.add("--shadow-cp7-export");
            allowed.add("--shadow-cp7-max-think-seconds");
            allowed.add("--generation");
            if (!values.keySet().containsAll(required)
                    || !allowed.containsAll(values.keySet())) {
                throw new IllegalArgumentException(
                        "required arguments: " + required
                                + "; optional: --pairs, --opponent, --cp7-skill,"
                                + " --store-root, --population-store-root, --behavior-clone-root, --outcome-root,"
                                + " --teacher-export, --outcome-export,"
                                + " --shadow-cp7-export, --shadow-cp7-max-think-seconds,"
                                + " --generation");
            }
            long baseSeed = Long.parseLong(values.get("--base-seed"));
            long firstEpisode = Long.parseLong(values.get("--first-episode"));
            int pairCount = Integer.parseInt(values.getOrDefault("--pairs", "1"));
            OpponentMode opponentMode = OpponentMode.parse(
                    values.getOrDefault("--opponent", "uniform"));
            int cp7Skill = Integer.parseInt(
                    values.getOrDefault("--cp7-skill", "7"));
            Long checkpointGeneration = values.containsKey("--generation")
                    ? Long.parseLong(values.get("--generation")) : null;
            boolean hasStoreRoot = values.containsKey("--store-root");
            boolean hasPopulationStoreRoot = values.containsKey("--population-store-root");
            boolean hasBehaviorCloneRoot = values.containsKey("--behavior-clone-root");
            boolean hasOutcomeRoot = values.containsKey("--outcome-root");
            Path teacherExportPath = null;
            if (values.containsKey("--teacher-export")) {
                Path requested = Paths.get(values.get("--teacher-export"))
                        .toAbsolutePath().normalize();
                Path parent = requested.getParent();
                if (parent == null || requested.getFileName() == null) {
                    throw new IllegalArgumentException(
                            "teacher export must name a file inside an existing directory");
                }
                teacherExportPath = parent.toRealPath().resolve(requested.getFileName());
            }
            Path outcomeExportPath = null;
            if (values.containsKey("--outcome-export")) {
                Path requested = Paths.get(values.get("--outcome-export"))
                        .toAbsolutePath().normalize();
                Path parent = requested.getParent();
                if (parent == null || requested.getFileName() == null) {
                    throw new IllegalArgumentException(
                            "outcome export must name a file inside an existing directory");
                }
                outcomeExportPath = parent.toRealPath().resolve(requested.getFileName());
            }
            Path shadowCp7ExportPath = null;
            if (values.containsKey("--shadow-cp7-export")) {
                Path requested = Paths.get(values.get("--shadow-cp7-export"))
                        .toAbsolutePath().normalize();
                Path parent = requested.getParent();
                if (parent == null || requested.getFileName() == null) {
                    throw new IllegalArgumentException(
                            "shadow CP7 export must name a file inside an existing directory");
                }
                shadowCp7ExportPath = parent.toRealPath().resolve(requested.getFileName());
            }
            int shadowCp7MaxThinkSeconds = Integer.parseInt(
                    values.getOrDefault("--shadow-cp7-max-think-seconds", "5"));
            if (baseSeed < 0L || firstEpisode < 0L
                    || (firstEpisode & 1L) != 0L || pairCount < 1 || pairCount > 128
                    || cp7Skill < 1 || cp7Skill > 10
                    || (hasStoreRoot ? 1 : 0) + (hasPopulationStoreRoot ? 1 : 0)
                    + (hasBehaviorCloneRoot ? 1 : 0)
                    + (hasOutcomeRoot ? 1 : 0) != 1
                    || (checkpointGeneration != null
                    && (hasBehaviorCloneRoot || hasOutcomeRoot))
                    || (hasPopulationStoreRoot && checkpointGeneration == null)
                    || (checkpointGeneration != null && checkpointGeneration < 0L)
                    || (teacherExportPath != null && opponentMode != OpponentMode.CP7)
                    || (outcomeExportPath != null && opponentMode != OpponentMode.CP7)
                    || (shadowCp7ExportPath != null && opponentMode != OpponentMode.CP7)
                    || (values.containsKey("--shadow-cp7-max-think-seconds")
                    && shadowCp7ExportPath == null)
                    || shadowCp7MaxThinkSeconds < 1
                    || shadowCp7MaxThinkSeconds > 120) {
                throw new IllegalArgumentException(
                        "base seed must be nonnegative, first episode must be even,"
                                + " pairs must be in [1,128],"
                                + " CP7 skill must be in [1,10],"
                                + " exactly one original, population, or derivative root must be selected,"
                                + " population Store requires generation,"
                                + " generation applies only to a Store,"
                                + " generation must be nonnegative,"
                                + " exports require opponent cp7,"
                                + " and shadow think seconds must be in [1,120]");
            }
            try {
                Math.addExact(firstEpisode, Math.subtractExact(
                        Math.multiplyExact(2L, pairCount), 1L));
            } catch (ArithmeticException error) {
                throw new IllegalArgumentException("episode range overflows", error);
            }
            return new Args(
                    Paths.get(values.get("--repo-root")),
                    Paths.get(values.get("--scorer-exe")),
                    hasStoreRoot ? Paths.get(values.get("--store-root")) : null,
                    hasPopulationStoreRoot
                            ? Paths.get(values.get("--population-store-root")) : null,
                    baseSeed,
                    firstEpisode,
                    pairCount,
                    opponentMode,
                    cp7Skill,
                    teacherExportPath,
                    outcomeExportPath,
                    shadowCp7ExportPath,
                    shadowCp7MaxThinkSeconds,
                    checkpointGeneration,
                    hasBehaviorCloneRoot
                            ? Paths.get(values.get("--behavior-clone-root")) : null,
                    hasOutcomeRoot ? Paths.get(values.get("--outcome-root")) : null);
        }
    }

    private enum OpponentMode {
        UNIFORM("uniform"),
        CP7("cp7");

        private final String wire;

        OpponentMode(String wire) {
            this.wire = wire;
        }

        private static OpponentMode parse(String value) {
            for (OpponentMode mode : values()) {
                if (mode.wire.equals(value)) {
                    return mode;
                }
            }
            throw new IllegalArgumentException(
                    "opponent must be exactly uniform or cp7");
        }
    }
}
