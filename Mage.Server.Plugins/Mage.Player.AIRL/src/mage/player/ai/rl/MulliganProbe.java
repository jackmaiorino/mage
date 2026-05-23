package mage.player.ai.rl;

import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.constants.MultiplayerAttackOption;
import mage.constants.RangeOfInfluence;
import mage.game.Game;
import mage.game.GameOptions;
import mage.game.TwoPlayerDuel;
import mage.game.mulligan.MulliganType;
import mage.player.ai.ComputerPlayer7;
import mage.player.ai.ComputerPlayerRL;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * Cheap opening-hand diagnostic for the live RL mulligan policy.
 */
public final class MulliganProbe {

    private static final String DEFAULT_DECK_LIST =
            "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/decklist.active_profile_pool.txt";
    private static final String DEFAULT_OUT_ROOT =
            "local-training/local_pbt/mulligan_probes";

    private MulliganProbe() {
    }

    public static void main(String[] args) throws Exception {
        Args parsed = Args.parse(args);
        Path outDir = parsed.outDir;
        if (outDir == null) {
            String stamp = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss").format(LocalDateTime.now());
            outDir = Paths.get(DEFAULT_OUT_ROOT, stamp).toAbsolutePath().normalize();
        }
        Files.createDirectories(outDir);

        List<Path> deckPaths = loadDeckList(parsed.deckList);
        if (deckPaths.isEmpty()) {
            throw new IllegalArgumentException("No decks found in " + parsed.deckList);
        }

        RLTrainer trainer = new RLTrainer();
        List<PromptRecord> records = new ArrayList<>();

        long started = System.currentTimeMillis();
        for (Path deckPath : deckPaths) {
            String deckName = stripExtension(deckPath.getFileName().toString());
            System.out.println("Probing " + deckName + " samples=" + parsed.samplesPerDeck);
            for (int sample = 1; sample <= parsed.samplesPerDeck; sample++) {
                Deck rlDeck = trainer.loadDeckFresh(deckPath.toString());
                Deck oppDeck = trainer.loadDeckFresh(deckPath.toString());
                if (rlDeck == null || oppDeck == null) {
                    throw new IllegalStateException("Failed to load deck: " + deckPath);
                }

                ProbePlayer rlPlayer = new ProbePlayer("ProbeRL", deckName, sample, parsed.firstOnly);
                ComputerPlayer7 opp = new ComputerPlayer7("ProbeCP7", RangeOfInfluence.ALL, 7);
                Game game = new MulliganOnlyGame();
                GameOptions options = new GameOptions();
                options.rollbackTurnsAllowed = false;
                game.setGameOptions(options);
                game.addPlayer(rlPlayer, rlDeck);
                game.addPlayer(opp, oppDeck);
                game.loadCards(rlDeck.getCards(), rlPlayer.getId());
                game.loadCards(oppDeck.getCards(), opp.getId());
                game.setStartingPlayerId(rlPlayer.getId());

                game.start(rlPlayer.getId());
                records.addAll(rlPlayer.records);
                game.end();
                game.cleanUp();
            }
        }

        writePrompts(outDir.resolve("mulligan_probe_prompts.csv"), records);
        writeSummary(outDir.resolve("summary_by_deck_land.csv"), records, BucketMode.ACTUAL_LANDS);
        writeSummary(outDir.resolve("summary_by_deck_effective_land.csv"), records, BucketMode.EFFECTIVE_LANDS);
        writeSummary(outDir.resolve("summary_by_deck.csv"), records, BucketMode.ALL);
        writeReadme(outDir.resolve("README.md"), parsed, deckPaths, records, System.currentTimeMillis() - started);
        RLTrainer.sharedModel.shutdown();

        System.out.println("Wrote " + records.size() + " mulligan prompts to " + outDir);
        System.out.println("Summary by effective land count:");
        for (SummaryRow row : summarize(records, BucketMode.EFFECTIVE_LANDS)) {
            if (row.firstPrompts > 0) {
                System.out.println(row.toConsole("effectiveLands"));
            }
        }
    }

    private static final class MulliganOnlyGame extends TwoPlayerDuel {

        private MulliganOnlyGame() {
            super(MultiplayerAttackOption.LEFT, RangeOfInfluence.ONE,
                    MulliganType.GAME_DEFAULT.getMulligan(0), 60, 20, 7);
        }

        @Override
        protected void play(UUID nextPlayerId) {
            // Stop after game init/mulligan phase.
        }
    }

    private static final class ProbePlayer extends ComputerPlayerRL {

        private final String deckName;
        private final int sample;
        private final boolean firstOnly;
        private final List<PromptRecord> records = new ArrayList<>();

        private ProbePlayer(String name, String deckName, int sample, boolean firstOnly) {
            super(name, RangeOfInfluence.ALL, RLTrainer.getSharedModel(), true, false, "train");
            this.deckName = deckName;
            this.sample = sample;
            this.firstOnly = firstOnly;
            setCurrentEpisode(-1);
        }

        @Override
        public boolean chooseMulligan(Game game) {
            int prompt = records.size();
            int handSize = getHand() == null ? -1 : getHand().size();
            int lands = countLands(game);
            int pseudoLands = countPseudoLands(game);
            String hand = handCards(game);
            boolean shouldMulligan = super.chooseMulligan(game);
            records.add(new PromptRecord(
                    deckName,
                    sample,
                    prompt,
                    handSize,
                    lands,
                    pseudoLands,
                    shouldMulligan ? "MULLIGAN" : "KEEP",
                    getLastMulliganPKeep(),
                    getLastMulliganPMull(),
                    getLastMulliganMode(),
                    hand
            ));
            if (firstOnly && prompt == 0) {
                return false;
            }
            return shouldMulligan;
        }

        private int countLands(Game game) {
            int count = 0;
            try {
                for (Card card : getHand().getCards(game)) {
                    if (card != null && card.isLand(game)) {
                        count++;
                    }
                }
            } catch (Exception ignored) {
            }
            return count;
        }

        private int countPseudoLands(Game game) {
            if (deckName == null || !deckName.toLowerCase(Locale.ROOT).contains("spy")) {
                return 0;
            }
            int count = 0;
            try {
                for (Card card : getHand().getCards(game)) {
                    if (card != null && ("Land Grant".equals(card.getName()) || "Lotus Petal".equals(card.getName()))) {
                        count++;
                    }
                }
            } catch (Exception ignored) {
            }
            return count;
        }

        private String handCards(Game game) {
            try {
                return getHand().getCards(game).stream()
                        .map(Card::getName)
                        .collect(Collectors.joining("; "));
            } catch (Exception ignored) {
                return "";
            }
        }
    }

    private static final class PromptRecord {
        final String deck;
        final int sample;
        final int prompt;
        final int handSize;
        final int lands;
        final int pseudoLands;
        final int effectiveLands;
        final String decision;
        final float pKeep;
        final float pMull;
        final String mode;
        final String hand;

        private PromptRecord(String deck, int sample, int prompt, int handSize, int lands, int pseudoLands,
                             String decision, float pKeep, float pMull, String mode, String hand) {
            this.deck = deck;
            this.sample = sample;
            this.prompt = prompt;
            this.handSize = handSize;
            this.lands = lands;
            this.pseudoLands = Math.max(0, pseudoLands);
            this.effectiveLands = lands < 0 ? lands : lands + this.pseudoLands;
            this.decision = decision;
            this.pKeep = pKeep;
            this.pMull = pMull;
            this.mode = mode == null ? "" : mode;
            this.hand = hand == null ? "" : hand;
        }

        boolean isFirstPrompt() {
            return prompt == 0;
        }

        boolean kept() {
            return "KEEP".equals(decision);
        }
    }

    private enum BucketMode {
        ALL,
        ACTUAL_LANDS,
        EFFECTIVE_LANDS
    }

    private static final class SummaryRow {
        String deck;
        String landBucket;
        int prompts;
        int firstPrompts;
        int keeps;
        int mulls;
        double pKeepSum;
        double pMullSum;

        String toCsv() {
            return csv(deck) + ','
                    + csv(landBucket) + ','
                    + prompts + ','
                    + firstPrompts + ','
                    + keeps + ','
                    + mulls + ','
                    + fmt(rate(keeps, prompts)) + ','
                    + fmt(rate(mulls, prompts)) + ','
                    + fmt(prompts == 0 ? 0.0 : pKeepSum / prompts) + ','
                    + fmt(prompts == 0 ? 0.0 : pMullSum / prompts) + '\n';
        }

        String toConsole(String bucketLabel) {
            return String.format(Locale.ROOT,
                    "%s %s=%s first=%d keep=%.1f%% mull=%.1f%% meanPkeep=%.3f meanPmull=%.3f",
                    deck,
                    bucketLabel,
                    landBucket,
                    firstPrompts,
                    100.0 * rate(keeps, prompts),
                    100.0 * rate(mulls, prompts),
                    prompts == 0 ? 0.0 : pKeepSum / prompts,
                    prompts == 0 ? 0.0 : pMullSum / prompts);
        }
    }

    private static List<SummaryRow> summarize(List<PromptRecord> records, BucketMode mode) {
        Map<String, SummaryRow> rows = new LinkedHashMap<>();
        for (PromptRecord r : records) {
            String landBucket = bucketFor(r, mode);
            String key = r.deck + "\t" + landBucket;
            SummaryRow row = rows.get(key);
            if (row == null) {
                row = new SummaryRow();
                row.deck = r.deck;
                row.landBucket = landBucket;
                rows.put(key, row);
            }
            row.prompts++;
            if (r.isFirstPrompt()) {
                row.firstPrompts++;
            }
            if (r.kept()) {
                row.keeps++;
            } else {
                row.mulls++;
            }
            if (Float.isFinite(r.pKeep)) {
                row.pKeepSum += r.pKeep;
            }
            if (Float.isFinite(r.pMull)) {
                row.pMullSum += r.pMull;
            }
        }
        return new ArrayList<>(rows.values());
    }

    private static String bucketFor(PromptRecord r, BucketMode mode) {
        if (mode == BucketMode.ALL) {
            return "ALL";
        }
        int value = mode == BucketMode.EFFECTIVE_LANDS ? r.effectiveLands : r.lands;
        return Integer.toString(Math.max(0, Math.min(7, value)));
    }

    private static void writePrompts(Path path, List<PromptRecord> records) throws IOException {
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("deck,sample,prompt,hand_size,lands,pseudo_lands,effective_lands,decision,p_keep,p_mull,mode,hand\n");
            for (PromptRecord r : records) {
                out.write(csv(r.deck));
                out.write(',');
                out.write(Integer.toString(r.sample));
                out.write(',');
                out.write(Integer.toString(r.prompt));
                out.write(',');
                out.write(Integer.toString(r.handSize));
                out.write(',');
                out.write(Integer.toString(r.lands));
                out.write(',');
                out.write(Integer.toString(r.pseudoLands));
                out.write(',');
                out.write(Integer.toString(r.effectiveLands));
                out.write(',');
                out.write(r.decision);
                out.write(',');
                out.write(fmt(r.pKeep));
                out.write(',');
                out.write(fmt(r.pMull));
                out.write(',');
                out.write(csv(r.mode));
                out.write(',');
                out.write(csv(r.hand));
                out.write('\n');
            }
        }
    }

    private static void writeSummary(Path path, List<PromptRecord> records, BucketMode mode) throws IOException {
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            String bucket = mode == BucketMode.EFFECTIVE_LANDS ? "effective_land_bucket" : "land_bucket";
            out.write("deck," + bucket + ",prompts,first_prompts,keeps,mulligans,keep_rate,mulligan_rate,mean_p_keep,mean_p_mull\n");
            for (SummaryRow row : summarize(records, mode)) {
                out.write(row.toCsv());
            }
        }
    }

    private static void writeReadme(Path path, Args args, List<Path> deckPaths,
                                    List<PromptRecord> records, long elapsedMs) throws IOException {
        try (BufferedWriter out = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            out.write("# Mulligan Probe\n\n");
            out.write("- Profile: `" + args.profile + "`\n");
            out.write("- Samples per deck: `" + args.samplesPerDeck + "`\n");
            out.write("- First prompt only: `" + args.firstOnly + "`\n");
            out.write("- Prompt rows: `" + records.size() + "`\n");
            out.write("- Effective lands: actual lands plus Spy `Land Grant`/`Lotus Petal` pseudo-lands for diagnostic buckets only\n");
            out.write("- Elapsed seconds: `" + fmt(elapsedMs / 1000.0) + "`\n");
            out.write("- Deck list: `" + args.deckList + "`\n\n");
            out.write("Decks:\n\n");
            for (Path p : deckPaths) {
                out.write("- `" + p.getFileName().toString() + "`\n");
            }
        }
    }

    private static List<Path> loadDeckList(String deckList) throws IOException {
        Path listPath = Paths.get(deckList == null || deckList.trim().isEmpty() ? DEFAULT_DECK_LIST : deckList)
                .toAbsolutePath()
                .normalize();
        if (Files.isRegularFile(listPath)) {
            String fn = listPath.getFileName().toString().toLowerCase(Locale.ROOT);
            if (fn.endsWith(".dek") || fn.endsWith(".dck")) {
                List<Path> single = new ArrayList<>();
                single.add(listPath);
                return single;
            }
        }
        Path base = listPath.getParent();
        List<Path> decks = new ArrayList<>();
        for (String raw : Files.readAllLines(listPath, StandardCharsets.UTF_8)) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) {
                continue;
            }
            Path p = Paths.get(line);
            if (!p.isAbsolute()) {
                p = base.resolve(p).normalize();
            }
            if (Files.isRegularFile(p)) {
                decks.add(p);
            }
        }
        return decks;
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    private static double rate(int n, int d) {
        return d == 0 ? 0.0 : (double) n / (double) d;
    }

    private static String fmt(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return "";
        }
        return String.format(Locale.ROOT, "%.6f", value);
    }

    private static String csv(String s) {
        String v = s == null ? "" : s;
        if (v.indexOf(',') < 0 && v.indexOf('"') < 0 && v.indexOf('\n') < 0 && v.indexOf('\r') < 0) {
            return v;
        }
        return '"' + v.replace("\"", "\"\"") + '"';
    }

    private static final class Args {
        String profile = EnvConfig.str("MODEL_PROFILE", "Pauper-Generalist-Value-v2");
        String deckList = DEFAULT_DECK_LIST;
        int samplesPerDeck = 500;
        boolean firstOnly = true;
        Path outDir = null;

        static Args parse(String[] args) {
            Args parsed = new Args();
            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                if ("--profile".equals(arg) && i + 1 < args.length) {
                    parsed.profile = args[++i];
                } else if ("--deck-list".equals(arg) && i + 1 < args.length) {
                    parsed.deckList = args[++i];
                } else if ("--samples".equals(arg) && i + 1 < args.length) {
                    parsed.samplesPerDeck = Integer.parseInt(args[++i]);
                } else if ("--first-only".equals(arg) && i + 1 < args.length) {
                    parsed.firstOnly = Boolean.parseBoolean(args[++i]);
                } else if ("--out".equals(arg) && i + 1 < args.length) {
                    parsed.outDir = Paths.get(args[++i]).toAbsolutePath().normalize();
                }
            }
            return parsed;
        }
    }
}
