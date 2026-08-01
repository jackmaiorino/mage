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
 * Fast engineering spike for the promoted checkpoint on the XMage Rally surface.
 *
 * <p>This is intentionally a two-game diagnostic, not a statistical claim. It
 * keeps one exact Rust checkpoint process alive, swaps the candidate seat, and
 * aborts immediately on any deck, decision, semantic, or terminal mismatch.</p>
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

        List<String> command = Arrays.asList(
                args.scorerExecutable.toString(),
                "--original-store-root",
                args.storeRoot.toString());
        long sampleStart = System.nanoTime();
        try (XMageRallyBridgeProcessClient bridge =
                     XMageRallyBridgeProcessClient.start(
                             command,
                             BRIDGE_TIMEOUT_MILLIS,
                             XMageRallyBridgeProcessClient.DEFAULT_MAX_LINE_BYTES,
                             System.err)) {
            int candidateWins = 0;
            int opponentWins = 0;
            int draws = 0;
            long totalTurns = 0L;
            long totalRustSteps = 0L;
            for (int pairOrdinal = 0; pairOrdinal < args.pairCount; pairOrdinal++) {
                long firstEpisode = Math.addExact(
                        args.firstEpisodeId, Math.multiplyExact(2L, pairOrdinal));
                long pairStart = System.nanoTime();
                LegResult first = runLeg(bridge, decks, args.baseSeed, firstEpisode);
                LegResult second = runLeg(bridge, decks, args.baseSeed, firstEpisode + 1L);
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
                    } else if (leg.winner.equals(leg.candidateSeat.wire())) {
                        candidateWins++;
                    } else {
                        opponentWins++;
                    }
                    totalTurns = Math.addExact(totalTurns, leg.turns);
                    totalRustSteps = Math.addExact(totalRustSteps, leg.rustSteps);
                }
                long pairElapsedMillis =
                        (System.nanoTime() - pairStart) / 1_000_000L;
                System.out.println("XMAGE_RALLY_ANCHOR_PAIR PASS"
                        + " base_seed=" + args.baseSeed
                        + " episodes=" + first.episodeId + "," + second.episodeId
                        + " pair_index=" + first.pairIndex
                        + " environment_seed=" + unsignedHex(first.environmentSeed)
                        + " candidate_seats=" + first.candidateSeat.wire()
                        + "," + second.candidateSeat.wire()
                        + " winners=" + first.winner + "," + second.winner
                        + " turns=" + first.turns + "," + second.turns
                        + " rust_steps=" + first.rustSteps + "," + second.rustSteps
                        + " elapsed_ms=" + pairElapsedMillis);
            }
            long elapsedMillis = (System.nanoTime() - sampleStart) / 1_000_000L;
            int games = Math.multiplyExact(args.pairCount, 2);
            System.out.println("XMAGE_RALLY_ANCHOR_SPIKE PASS"
                    + " base_seed=" + args.baseSeed
                    + " first_episode=" + args.firstEpisodeId
                    + " pairs=" + args.pairCount
                    + " games=" + games
                    + " candidate_wins=" + candidateWins
                    + " opponent_wins=" + opponentWins
                    + " draws=" + draws
                    + " total_turns=" + totalTurns
                    + " total_rust_steps=" + totalRustSteps
                    + " elapsed_ms=" + elapsedMillis);
        }
    }

    private static LegResult runLeg(XMageRallyBridgeProcessClient bridge,
                                    DeckTemplates decks,
                                    long baseSeed,
                                    long episodeId) throws Exception {
        AtomicReference<LegResult> result = new AtomicReference<>();
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Thread gameThread = new Thread(() -> {
            try {
                result.set(runLegInGameThread(bridge, decks, baseSeed, episodeId));
            } catch (Throwable error) {
                failure.set(error);
            }
        }, "GAME-XMAGE-RALLY-ANCHOR-e" + episodeId);
        gameThread.setDaemon(false);
        gameThread.start();
        gameThread.join();
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
            long baseSeed,
            long episodeId) throws Exception {
        long start = System.nanoTime();
        bridge.reset("anchor-reset-" + episodeId, episodeId, baseSeed);
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
        ComputerPlayerUniformMirror p0 = null;
        ComputerPlayerUniformMirror p1 = null;
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

            p0Policy = policyForSeat(
                    bridge, baseSeed, episodeId, XMageRallyBridgeProtocol.Seat.P0,
                    candidateSeat == XMageRallyBridgeProtocol.Seat.P0, initialArenaIds);
            p1Policy = policyForSeat(
                    bridge, baseSeed, episodeId, XMageRallyBridgeProtocol.Seat.P1,
                    candidateSeat == XMageRallyBridgeProtocol.Seat.P1, initialArenaIds);
            MatchOptions matchOptions = fixedMatchOptions();
            TwoPlayerMatch match = new TwoPlayerMatch(matchOptions);
            match.startGame();
            game = match.getGames().get(0);
            String suffix = "-e" + episodeId;
            p0 = new ComputerPlayerUniformMirror(
                    "shadow-p0" + suffix, RangeOfInfluence.ALL, p0Policy, "p0");
            p1 = new ComputerPlayerUniformMirror(
                    "shadow-p1" + suffix, RangeOfInfluence.ALL, p1Policy, "p1");
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

        requireNaturalTerminal(game, p0, p1, p0Policy, p1Policy);
        XMageRallyBridgeProtocol.TerminalBody nativeTerminal = bridge.getTerminal();
        if (nativeTerminal == null) {
            throw new IllegalStateException("XMage ended before the native Rally episode");
        }
        requireNaturalNativeTerminal(nativeTerminal, p0Policy, p1Policy);
        String winner = xmageWinner(game, p0, p1);
        String nativeWinner = nativeTerminal.getTerminal().getWinner() == null
                ? "draw" : nativeTerminal.getTerminal().getWinner().wire();
        if (!winner.equals(nativeWinner)) {
            throw new IllegalStateException("terminal winner mismatch: XMage="
                    + winner + " Rust=" + nativeWinner);
        }
        long rustSteps = nativeTerminal.getTerminal().getPolicyStepCount();
        long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;
        System.out.println("XMAGE_RALLY_ANCHOR_LEG PASS"
                + " episode=" + episodeId
                + " candidate=" + candidateSeat.wire()
                + " winner=" + winner
                + " turns=" + game.getTurnNum()
                + " rust_steps=" + rustSteps
                + " environment_seed=" + unsignedHex(environmentSeed)
                + " elapsed_ms=" + elapsedMillis);
        return new LegResult(
                episodeId,
                resetDecision.getPairIndex(),
                environmentSeed,
                candidateSeat,
                winner,
                game.getTurnNum(),
                rustSteps);
    }

    private static KernelShadowRallyPolicy policyForSeat(
            XMageRallyBridgeProcessClient bridge,
            long baseSeed,
            long episodeId,
            XMageRallyBridgeProtocol.Seat seat,
            boolean modelControlled,
            Map<UUID, Integer> initialArenaIds) {
        String wire = seat.wire();
        SeededUniformMirrorPolicy simulation =
                new SeededUniformMirrorPolicy(baseSeed, episodeId, wire);
        RallyCanonicalDecisionPolicy delegate = modelControlled
                ? null : new SeededUniformMirrorPolicy(baseSeed, episodeId, wire);
        return new KernelShadowRallyPolicy(
                bridge, episodeId, seat, modelControlled, delegate, simulation,
                initialArenaIds);
    }

    private static void requireNaturalTerminal(
            Game game,
            ComputerPlayerUniformMirror p0,
            ComputerPlayerUniformMirror p1,
            KernelShadowRallyPolicy p0Policy,
            KernelShadowRallyPolicy p1Policy) {
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
        if (p0Policy.isFailed() || p1Policy.isFailed()) {
            throw new IllegalStateException("shadow policy failed: p0="
                    + p0Policy.getFirstFailure() + " p1=" + p1Policy.getFirstFailure());
        }
        xmageWinner(game, p0, p1);
    }

    private static void requireNaturalNativeTerminal(
            XMageRallyBridgeProtocol.TerminalBody nativeTerminal,
            KernelShadowRallyPolicy p0Policy,
            KernelShadowRallyPolicy p1Policy) {
        XMageRallyBridgeProtocol.TerminalRecord terminal = nativeTerminal.getTerminal();
        if (!"natural".equals(terminal.getTerminalClassification())
                || !"natural_game_over".equals(terminal.getTerminalCode())) {
            throw new IllegalStateException("native Rally episode did not end naturally: "
                    + terminal.getTerminalClassification() + "/" + terminal.getTerminalCode());
        }
        long xmageSteps = Math.addExact(
                p0Policy.getPolicyActionSelections(),
                p1Policy.getPolicyActionSelections());
        long xmagePhysicalDecisions = Math.addExact(
                p0Policy.getPhysicalDecisionCount(),
                p1Policy.getPhysicalDecisionCount());
        if (terminal.getPolicyStepCount() != xmageSteps
                || terminal.getPhysicalDecisionCount() != xmagePhysicalDecisions) {
            throw new IllegalStateException("native/XMage policy counts differ: steps="
                    + terminal.getPolicyStepCount() + "/" + xmageSteps
                    + " physical=" + terminal.getPhysicalDecisionCount()
                    + "/" + xmagePhysicalDecisions);
        }
    }

    private static String xmageWinner(
            Game game, ComputerPlayerUniformMirror p0, ComputerPlayerUniformMirror p1) {
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

        LegResult(long episodeId,
                  long pairIndex,
                  long environmentSeed,
                  XMageRallyBridgeProtocol.Seat candidateSeat,
                  String winner,
                  int turns,
                  long rustSteps) {
            this.episodeId = episodeId;
            this.pairIndex = pairIndex;
            this.environmentSeed = environmentSeed;
            this.candidateSeat = candidateSeat;
            this.winner = winner;
            this.turns = turns;
            this.rustSteps = rustSteps;
        }
    }

    private static final class Args {
        final Path repoRoot;
        final Path scorerExecutable;
        final Path storeRoot;
        final long baseSeed;
        final long firstEpisodeId;
        final int pairCount;

        Args(Path repoRoot,
             Path scorerExecutable,
             Path storeRoot,
             long baseSeed,
             long firstEpisodeId,
             int pairCount) throws Exception {
            this.repoRoot = repoRoot.toRealPath();
            this.scorerExecutable = scorerExecutable.toRealPath();
            this.storeRoot = storeRoot.toRealPath();
            this.baseSeed = baseSeed;
            this.firstEpisodeId = firstEpisodeId;
            this.pairCount = pairCount;
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
                    "--repo-root", "--scorer-exe", "--store-root",
                    "--base-seed", "--first-episode"));
            Set<String> allowed = new HashSet<>(required);
            allowed.add("--pairs");
            if (!values.keySet().containsAll(required)
                    || !allowed.containsAll(values.keySet())) {
                throw new IllegalArgumentException(
                        "required arguments: " + required + "; optional: --pairs");
            }
            long baseSeed = Long.parseLong(values.get("--base-seed"));
            long firstEpisode = Long.parseLong(values.get("--first-episode"));
            int pairCount = Integer.parseInt(values.getOrDefault("--pairs", "1"));
            if (baseSeed < 0L || firstEpisode < 0L
                    || (firstEpisode & 1L) != 0L || pairCount < 1 || pairCount > 128) {
                throw new IllegalArgumentException(
                        "base seed must be nonnegative, first episode must be even,"
                                + " and pairs must be in [1,128]");
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
                    Paths.get(values.get("--store-root")),
                    baseSeed,
                    firstEpisode,
                    pairCount);
        }
    }
}
