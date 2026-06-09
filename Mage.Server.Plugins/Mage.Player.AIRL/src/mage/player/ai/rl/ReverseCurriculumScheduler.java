package mage.player.ai.rl;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;

/**
 * Reverse-curriculum start-state scheduler.
 *
 * <p>Picks, per training episode, whether to start from a captured mid-game
 * checkpoint (and which one) instead of a fresh deal. Start states are selected
 * ONLY by (a) source outcome = won and (b) distance-from-terminal. The sampled
 * max-distance anneals from near-terminal toward the full game over training, so
 * terminal reward first credits the short finishing line (which the agent already
 * plays well) and progressively reaches the earlier assembly decisions.
 *
 * <p>Thesis constraint: selection encodes no combo predicate / card name /
 * zone-content rule. The corpus index is asserted at load to contain only
 * outcome+position columns; any card/zone/combo column is rejected. Changing the
 * initial-state distribution is standard reverse curriculum (Florensa 2017);
 * injecting "combo = good" beliefs is not allowed.
 *
 * <p>The index CSV (produced offline) has columns:
 * {@code snapshot_path,source_won,distance_from_terminal,bucket,ordinal,action_type}.
 */
public final class ReverseCurriculumScheduler {

    /** Columns allowed in the corpus index. Any other column is a thesis-purity violation. */
    private static final Set<String> ALLOWED_COLUMNS = new HashSet<>(Arrays.asList(
            "snapshot_path", "source_won", "distance_from_terminal", "bucket", "ordinal", "action_type"));

    private final List<String> paths = new ArrayList<>();
    private final List<Integer> distances = new ArrayList<>();
    private final double startProb;
    private final int startDist;
    private final int endDist;
    private final int annealEpisodes;
    private int corpusMaxDistance = 0;

    private ReverseCurriculumScheduler(double startProb, int startDist, int endDist, int annealEpisodes) {
        this.startProb = startProb;
        this.startDist = startDist;
        this.endDist = endDist;
        this.annealEpisodes = Math.max(1, annealEpisodes);
    }

    /** Build from EnvConfig: RC_CORPUS_INDEX, RC_START_FROM_CHECKPOINT_PROB,
     *  RC_START_DIST, RC_END_DIST, RC_ANNEAL_EPISODES. Returns null when RC is
     *  off or the corpus is unavailable/empty (caller falls back to fresh deals). */
    public static ReverseCurriculumScheduler fromEnv() {
        if (!EnvConfig.bool("RC_ENABLE", false)) {
            return null;
        }
        String idx = EnvConfig.str("RC_CORPUS_INDEX", "").trim();
        if (idx.isEmpty()) {
            System.out.println("[RC] RC_ENABLE=1 but RC_CORPUS_INDEX unset; reverse curriculum disabled.");
            return null;
        }
        double prob = EnvConfig.f64("RC_START_FROM_CHECKPOINT_PROB", 0.3);
        int startDist = EnvConfig.i32("RC_START_DIST", 8);
        int endDist = EnvConfig.i32("RC_END_DIST", 120);
        int anneal = EnvConfig.i32("RC_ANNEAL_EPISODES", 20000);
        try {
            ReverseCurriculumScheduler s = load(Paths.get(idx), prob, startDist, endDist, anneal);
            System.out.println("[RC] reverse curriculum enabled: " + s.size() + " won-game start states from " + idx
                    + " (prob=" + prob + " startDist=" + startDist + " endDist=" + endDist + " anneal=" + anneal + ")");
            return s.size() > 0 ? s : null;
        } catch (Exception e) {
            System.out.println("[RC] Failed to load corpus index " + idx + ": " + e + "; reverse curriculum disabled.");
            return null;
        }
    }

    public static ReverseCurriculumScheduler load(Path indexCsv, double startProb, int startDist,
            int endDist, int annealEpisodes) throws IOException {
        ReverseCurriculumScheduler s = new ReverseCurriculumScheduler(startProb, startDist, endDist, annealEpisodes);
        try (BufferedReader r = Files.newBufferedReader(indexCsv, StandardCharsets.UTF_8)) {
            String header = r.readLine();
            if (header == null) {
                throw new IOException("empty corpus index");
            }
            String[] cols = splitCsv(header);
            // Thesis-purity guard: only outcome+position columns permitted.
            for (String c : cols) {
                String norm = c.trim().toLowerCase(Locale.US);
                if (!ALLOWED_COLUMNS.contains(norm)) {
                    throw new IOException("THESIS-PURITY VIOLATION: corpus index column '" + c
                            + "' is not an allowed (outcome,position) column " + ALLOWED_COLUMNS
                            + ". Reverse-curriculum start selection must not encode combo/card/zone predicates.");
                }
            }
            int pathCol = indexOf(cols, "snapshot_path");
            int wonCol = indexOf(cols, "source_won");
            int distCol = indexOf(cols, "distance_from_terminal");
            if (pathCol < 0 || wonCol < 0 || distCol < 0) {
                throw new IOException("corpus index missing required columns snapshot_path/source_won/distance_from_terminal");
            }
            String line;
            while ((line = r.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }
                String[] f = splitCsv(line);
                int need = Math.max(pathCol, Math.max(wonCol, distCol));
                if (f.length <= need) {
                    continue;
                }
                // Outcome filter: WON source games only (outcome-defined).
                String won = f[wonCol].trim();
                if (!"1".equals(won) && !"true".equalsIgnoreCase(won)) {
                    continue;
                }
                int dist;
                try {
                    dist = Integer.parseInt(f[distCol].trim());
                } catch (NumberFormatException nfe) {
                    continue;
                }
                s.paths.add(unquote(f[pathCol].trim()));
                s.distances.add(dist);
                s.corpusMaxDistance = Math.max(s.corpusMaxDistance, dist);
            }
        }
        return s;
    }

    /** Current annealed max distance for the given training progress. */
    public int currentMaxDistance(long epNumber) {
        double frac = Math.min(1.0, Math.max(0.0, (double) epNumber / (double) annealEpisodes));
        int target = (int) Math.round(startDist + frac * (endDist - startDist));
        return Math.max(startDist, target);
    }

    /**
     * Choose a checkpoint to start this episode from, or null to start a fresh deal.
     * Deterministic per episode (seeded by epNumber) for reproducibility.
     */
    public String chooseStartSnapshot(long epNumber) {
        if (paths.isEmpty() || startProb <= 0.0) {
            return null;
        }
        Random rng = new Random(0x9E3779B97F4A7C15L ^ epNumber);
        if (rng.nextDouble() >= startProb) {
            return null; // fresh deal
        }
        int maxDist = currentMaxDistance(epNumber);
        List<Integer> eligible = new ArrayList<>();
        for (int i = 0; i < distances.size(); i++) {
            if (distances.get(i) <= maxDist) {
                eligible.add(i);
            }
        }
        if (eligible.isEmpty()) {
            return null;
        }
        return paths.get(eligible.get(rng.nextInt(eligible.size())));
    }

    public int size() {
        return paths.size();
    }

    public int corpusMaxDistance() {
        return corpusMaxDistance;
    }

    // --- small helpers ---
    private static String[] splitCsv(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQ = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') {
                inQ = !inQ;
            } else if (c == ',' && !inQ) {
                out.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        out.add(cur.toString());
        return out.toArray(new String[0]);
    }

    private static String unquote(String s) {
        String t = s.trim();
        if (t.length() >= 2 && t.startsWith("\"") && t.endsWith("\"")) {
            return t.substring(1, t.length() - 1).replace("\"\"", "\"");
        }
        return t;
    }

    private static int indexOf(String[] cols, String name) {
        for (int i = 0; i < cols.length; i++) {
            if (cols[i].trim().equalsIgnoreCase(name)) {
                return i;
            }
        }
        return -1;
    }

    /** Self-test: load an index and print the sampled distance distribution at several
     *  training-progress points. Runnable without the engine. */
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("usage: ReverseCurriculumScheduler <rc_corpus_index.csv> [prob] [startDist] [endDist] [annealEps]");
            return;
        }
        double prob = args.length > 1 ? Double.parseDouble(args[1]) : 1.0;
        int sd = args.length > 2 ? Integer.parseInt(args[2]) : 8;
        int ed = args.length > 3 ? Integer.parseInt(args[3]) : 120;
        int ae = args.length > 4 ? Integer.parseInt(args[4]) : 20000;
        ReverseCurriculumScheduler s = load(Paths.get(args[0]), prob, sd, ed, ae);
        System.out.println("loaded " + s.size() + " won-game start states; corpusMaxDistance=" + s.corpusMaxDistance());
        long[] eps = {0L, ae / 4L, ae / 2L, (long) ae, ae * 2L};
        for (long ep : eps) {
            int md = s.currentMaxDistance(ep);
            int picks = 0, fresh = 0, sumDist = 0;
            int n = 4000;
            for (int i = 0; i < n; i++) {
                String p = s.chooseStartSnapshot(ep + i);
                if (p == null) {
                    fresh++;
                } else {
                    picks++;
                }
            }
            System.out.printf(Locale.US,
                    "ep=%d annealMaxDist=%d -> over %d episodes: fromCheckpoint=%d freshDeal=%d%n",
                    ep, md, n, picks, fresh);
        }
    }
}
