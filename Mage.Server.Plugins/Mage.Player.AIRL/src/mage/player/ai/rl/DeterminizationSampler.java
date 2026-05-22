package mage.player.ai.rl;

import mage.cards.Card;
import mage.cards.decks.Deck;
import mage.cards.decks.DeckCardLists;
import mage.cards.decks.importer.DeckImporter;
import mage.game.Game;
import mage.game.permanent.Permanent;
import mage.game.stack.StackObject;
import mage.players.Player;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

/**
 * Deterministic ISMCTS determinization sampler.
 * <p>
 * No neural networks involved: given the known Pauper archetype decklists, archetype
 * classification is Bayesian elimination over visible cards, hand sampling
 * is uniform-without-replacement from the remaining pool, and library is
 * a shuffled remainder. See mcts_implementation_plan_apr2026.md for why
 * we don't need a card-level belief head here.
 */
public final class DeterminizationSampler {

    /**
     * Static-loaded archetype name -> card-name -> count map.
     * Loaded once via {@link #loadArchetypes}; thereafter immutable.
     */
    private final Map<String, Map<String, Integer>> archetypeDecklists;

    /**
     * The deck total size per archetype (normally 60). Cached for speed
     * and for sanity checks during sampling.
     */
    private final Map<String, Integer> archetypeDeckSizes;
    private static final ThreadLocal<CardBeliefContext> THREAD_LOCAL_CARD_BELIEF = new ThreadLocal<>();

    private DeterminizationSampler(Map<String, Map<String, Integer>> decklists,
                                   Map<String, Integer> sizes) {
        this.archetypeDecklists = decklists;
        this.archetypeDeckSizes = sizes;
    }

    private static final class CardBeliefContext {
        final List<String> vocab;
        final float[] maxCounts;
        final float[] predictions;

        CardBeliefContext(List<String> vocab, float[] maxCounts, float[] predictions) {
            this.vocab = vocab == null ? Collections.emptyList() : new ArrayList<>(vocab);
            this.maxCounts = maxCounts == null ? new float[0] : Arrays.copyOf(maxCounts, maxCounts.length);
            this.predictions = predictions == null ? new float[0] : Arrays.copyOf(predictions, predictions.length);
        }

        boolean usable() {
            return !vocab.isEmpty() && predictions.length == vocab.size();
        }
    }

    public static void setThreadLocalCardBelief(List<String> vocab, float[] maxCounts, float[] predictions) {
        CardBeliefContext ctx = new CardBeliefContext(vocab, maxCounts, predictions);
        if (ctx.usable()) {
            THREAD_LOCAL_CARD_BELIEF.set(ctx);
        } else {
            THREAD_LOCAL_CARD_BELIEF.remove();
        }
    }

    public static void clearThreadLocalCardBelief() {
        THREAD_LOCAL_CARD_BELIEF.remove();
    }

    /**
     * Load archetype decklists from a map of archetype name -> .dek path.
     */
    public static DeterminizationSampler loadArchetypes(Map<String, String> archetypePaths) {
        Map<String, Map<String, Integer>> decklists = new LinkedHashMap<>();
        Map<String, Integer> sizes = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : archetypePaths.entrySet()) {
            String arch = e.getKey();
            String path = e.getValue();
            if (!Files.exists(Paths.get(path))) {
                continue;
            }
            Deck deck = loadDeck(path);
            if (deck == null) continue;
            Map<String, Integer> counts = new HashMap<>();
            int total = 0;
            for (Card c : deck.getCards()) {
                counts.merge(c.getName(), 1, Integer::sum);
                total++;
            }
            decklists.put(arch, counts);
            sizes.put(arch, total);
        }
        return new DeterminizationSampler(decklists, sizes);
    }

    /**
     * Load archetypes from a deck-list file such as the training/eval pool
     * files used by RLTrainer. Relative entries are resolved against the
     * deck-list file's parent directory.
     */
    public static DeterminizationSampler loadFromDeckListFile(String deckListFile) {
        if (deckListFile == null || deckListFile.trim().isEmpty()) {
            return new DeterminizationSampler(Collections.emptyMap(), Collections.emptyMap());
        }
        Path listPath = Paths.get(deckListFile.trim());
        if (!Files.exists(listPath)) {
            return new DeterminizationSampler(Collections.emptyMap(), Collections.emptyMap());
        }
        Path base = listPath.getParent();
        Map<String, String> paths = new LinkedHashMap<>();
        try {
            for (String raw : Files.readAllLines(listPath)) {
                String line = raw == null ? "" : raw.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                Path deckPath = Paths.get(line);
                if (!deckPath.isAbsolute() && base != null) {
                    deckPath = base.resolve(deckPath).normalize();
                }
                String archetype = archetypeNameFromDeckPath(deckPath);
                String unique = archetype;
                int suffix = 2;
                while (paths.containsKey(unique)) {
                    unique = archetype + suffix++;
                }
                paths.put(unique, deckPath.toString());
            }
        } catch (Exception e) {
            return new DeterminizationSampler(Collections.emptyMap(), Collections.emptyMap());
        }
        return loadArchetypes(paths);
    }

    private static Deck loadDeck(String path) {
        try {
            StringBuilder warnings = new StringBuilder();
            DeckCardLists lists = DeckImporter.importDeckFromFile(path, warnings, false);
            return Deck.load(lists, false, false, null);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Default factory using the canonical Pauper decks shipped with the repo.
     * Returns null if any decklist fails to load.
     */
    public static DeterminizationSampler pauperDefaults() {
        String base = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper";
        Map<String, String> paths = new LinkedHashMap<>();
        paths.put("Wildfire", base + "/Deck - Jund Wildfire.dek");
        paths.put("Rally", base + "/Deck - Mono Red Rally.dek");
        paths.put("Affinity", base + "/Deck - Grixis Affinity.dek");
        paths.put("Elves", base + "/Deck - Elves.dek");
        paths.put("SpyCombo", base + "/Deck - Spy Combo.dek");
        paths.put("Burn", base + "/Deck - Mono-Red Burn.dek");
        paths.put("Terror", base + "/Deck - Mono-Blue Terror.dek");
        paths.put("CawGates", base + "/Deck - Caw-Gates.dek");
        paths.put("Faeries", base + "/Deck - Mono-Blue Faeries.dek");
        return loadArchetypes(paths);
    }

    private static String archetypeNameFromDeckPath(Path deckPath) {
        String name = deckPath == null || deckPath.getFileName() == null
                ? "Archetype"
                : deckPath.getFileName().toString();
        if (name.endsWith(".dek")) {
            name = name.substring(0, name.length() - 4);
        }
        if (name.startsWith("Deck - ")) {
            name = name.substring("Deck - ".length());
        }
        StringBuilder out = new StringBuilder();
        boolean capitalize = true;
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (Character.isLetterOrDigit(c)) {
                out.append(capitalize ? Character.toUpperCase(c) : c);
                capitalize = false;
            } else {
                capitalize = true;
            }
        }
        return out.length() == 0 ? "Archetype" : out.toString();
    }

    public List<String> getArchetypes() {
        return new ArrayList<>(archetypeDecklists.keySet());
    }

    // -----------------------------------------------------------------------
    // Archetype classification (Bayesian elimination)
    // -----------------------------------------------------------------------

    /**
     * Returns the posterior probability over archetypes given the cards
     * visible in the opponent's public zones (BF/GY/exile/stack). Uniform
     * prior over archetypes; each visible card eliminates any archetype
     * that doesn't contain that card in sufficient count.
     * <p>
     * Result keys preserve the order returned by {@link #getArchetypes()}.
     * Returns uniform over all archetypes when no cards are visible or when
     * all archetypes would be eliminated (off-meta fallback).
     */
    public Map<String, Float> classifyArchetype(Game game, UUID oppId) {
        Map<String, Integer> visible = collectVisibleCardCounts(game, oppId);
        return classifyArchetypeFromCounts(visible);
    }

    /**
     * Pure function variant for tests: classify given an observed multi-set
     * of visible card names.
     */
    public Map<String, Float> classifyArchetypeFromCounts(Map<String, Integer> visible) {
        Map<String, Float> logPrior = new LinkedHashMap<>();
        float uniform = (float) Math.log(1.0 / archetypeDecklists.size());
        for (String arch : archetypeDecklists.keySet()) {
            logPrior.put(arch, uniform);
        }
        // Hard-elimination step: for each visible card, kill archetypes that
        // don't have enough of it. We track log-probs to avoid underflow for
        // many-card observations, even though everything's 0 or -inf here.
        for (Map.Entry<String, Integer> e : visible.entrySet()) {
            String cardName = e.getKey();
            int needed = e.getValue();
            for (String arch : archetypeDecklists.keySet()) {
                int have = archetypeDecklists.get(arch).getOrDefault(cardName, 0);
                if (have < needed) {
                    logPrior.put(arch, Float.NEGATIVE_INFINITY);
                }
            }
        }
        // Normalize: pick max, exp-shift, sum, divide
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logPrior.values()) if (v > max) max = v;
        Map<String, Float> out = new LinkedHashMap<>();
        if (Float.isInfinite(max)) {
            // All archetypes eliminated -- fall back to uniform.
            float u = 1.0f / archetypeDecklists.size();
            for (String arch : archetypeDecklists.keySet()) out.put(arch, u);
            return out;
        }
        float sum = 0;
        for (Map.Entry<String, Float> e : logPrior.entrySet()) {
            float v = (float) Math.exp(e.getValue() - max);
            out.put(e.getKey(), v);
            sum += v;
        }
        if (sum <= 0) {
            float u = 1.0f / archetypeDecklists.size();
            for (String arch : archetypeDecklists.keySet()) out.put(arch, u);
            return out;
        }
        for (Map.Entry<String, Float> e : out.entrySet()) {
            out.put(e.getKey(), e.getValue() / sum);
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Determinization sampling
    // -----------------------------------------------------------------------

    public static final class Determinization {
        public final String archetype;
        public final List<String> oppHandCards;    // card names, size = opp's public hand size
        public final List<String> oppLibraryOrder; // card names, top-of-library first
        public final Map<String, Float> archetypePosterior;

        public Determinization(String archetype, List<String> hand, List<String> library,
                               Map<String, Float> posterior) {
            this.archetype = archetype;
            this.oppHandCards = Collections.unmodifiableList(hand);
            this.oppLibraryOrder = Collections.unmodifiableList(library);
            this.archetypePosterior = Collections.unmodifiableMap(posterior);
        }
    }

    /**
     * Sample a plausible full-info determinization of opp's hidden zones
     * (hand + library) given the current public state.
     */
    public Determinization sample(Game game, UUID oppId, Random rng) {
        CardBeliefContext cardCtx = THREAD_LOCAL_CARD_BELIEF.get();
        if (cardCtx != null && cardCtx.usable()) {
            return sampleWithCardBelief(game, oppId, cardCtx.vocab, cardCtx.maxCounts, cardCtx.predictions, rng);
        }
        Map<String, Float> posterior = classifyArchetype(game, oppId);
        String archetype = sampleFromPosterior(posterior, rng);
        return sampleForArchetype(game, oppId, archetype, posterior, rng);
    }

    /**
     * Sample a determinization from the usual public-card archetype posterior,
     * but order the remaining hidden card pool by generic neural card-belief
     * predictions. This keeps the decklist/archetype prior intact while using
     * learned hidden-zone expectations to bias which plausible cards land in
     * hand versus deeper library.
     */
    public Determinization sampleWithCardBelief(Game game, UUID oppId,
                                                List<String> vocab,
                                                float[] maxCounts,
                                                float[] predictions,
                                                Random rng) {
        Map<String, Float> posterior = classifyArchetype(game, oppId);
        String archetype = sampleFromPosterior(posterior, rng);
        return sampleForArchetypeWithCardBelief(game, oppId, archetype, posterior,
                vocab, maxCounts, predictions, rng);
    }

    /**
     * Sample a determinization conditional on a specific archetype choice.
     */
    public Determinization sampleForArchetype(Game game, UUID oppId, String archetype,
                                              Map<String, Float> posterior, Random rng) {
        Map<String, Integer> visible = collectVisibleCardCounts(game, oppId);
        int handSize = 0;
        if (game != null && oppId != null) {
            Player opp = game.getPlayer(oppId);
            if (opp != null) handSize = opp.getHand().size();
        }
        return sampleForArchetypePure(archetype, visible, handSize, posterior, rng);
    }

    /**
     * Pure-function variant of sampleForArchetype for tests and non-Game
     * callers: given an archetype, the visible-card multiset (opp's public
     * zones), and opp's public hand size, return a determinization. No
     * Game instance required.
     */
    public Determinization sampleForArchetypePure(String archetype,
                                                  Map<String, Integer> visible,
                                                  int handSize,
                                                  Map<String, Float> posterior,
                                                  Random rng) {
        Map<String, Integer> decklist = archetypeDecklists.get(archetype);
        if (decklist == null) {
            return new Determinization(archetype, Collections.emptyList(),
                    Collections.emptyList(), posterior);
        }
        // Start from the full decklist as a mutable multiset.
        Map<String, Integer> remaining = new HashMap<>(decklist);
        // Subtract everything visible: BF, GY, exile, stack. We DO NOT subtract
        // opp's hand or library -- we're about to sample those.
        if (visible != null) {
            for (Map.Entry<String, Integer> e : visible.entrySet()) {
                remaining.merge(e.getKey(), -e.getValue(), Integer::sum);
            }
        }
        // Drop non-positive entries (can happen if opp plays off-archetype).
        remaining.values().removeIf(v -> v <= 0);

        // Flatten remaining multiset into a pool, shuffle.
        List<String> pool = new ArrayList<>();
        for (Map.Entry<String, Integer> e : remaining.entrySet()) {
            for (int i = 0; i < e.getValue(); i++) pool.add(e.getKey());
        }
        Collections.shuffle(pool, rng);

        int effHand = Math.max(0, Math.min(handSize, pool.size()));
        List<String> hand = new ArrayList<>(pool.subList(0, effHand));
        List<String> library = new ArrayList<>(pool.subList(effHand, pool.size()));
        return new Determinization(archetype, hand, library, posterior);
    }

    private Determinization sampleForArchetypeWithCardBelief(Game game,
                                                             UUID oppId,
                                                             String archetype,
                                                             Map<String, Float> posterior,
                                                             List<String> vocab,
                                                             float[] maxCounts,
                                                             float[] predictions,
                                                             Random rng) {
        Map<String, Integer> decklist = archetypeDecklists.get(archetype);
        if (decklist == null || vocab == null || predictions == null || vocab.isEmpty()) {
            return sampleForArchetype(game, oppId, archetype, posterior, rng);
        }
        Map<String, Integer> visible = collectVisibleCardCounts(game, oppId);
        int handSize = 0;
        if (game != null && oppId != null) {
            Player opp = game.getPlayer(oppId);
            if (opp != null) {
                handSize = opp.getHand().size();
            }
        }

        Map<String, Integer> remaining = new HashMap<>(decklist);
        if (visible != null) {
            for (Map.Entry<String, Integer> e : visible.entrySet()) {
                remaining.merge(e.getKey(), -e.getValue(), Integer::sum);
            }
        }
        remaining.values().removeIf(v -> v <= 0);

        Map<String, Float> weightByName = new HashMap<>();
        int limit = Math.min(vocab.size(), predictions.length);
        for (int i = 0; i < limit; i++) {
            String name = vocab.get(i);
            if (name == null || name.isEmpty()) {
                continue;
            }
            float pred = predictions[i];
            if (Float.isNaN(pred) || Float.isInfinite(pred)) {
                pred = 0.0f;
            }
            float maxCount = maxCounts != null && i < maxCounts.length ? maxCounts[i] : 1.0f;
            float expectedCount = Math.max(0.0f, Math.min(1.0f, pred)) * Math.max(1.0f, maxCount);
            weightByName.put(name, expectedCount);
        }

        List<String> pool = new ArrayList<>();
        for (Map.Entry<String, Integer> e : remaining.entrySet()) {
            for (int i = 0; i < e.getValue(); i++) {
                pool.add(e.getKey());
            }
        }
        List<String> ordered = weightedShuffle(pool, weightByName, rng);
        int effHand = Math.max(0, Math.min(handSize, ordered.size()));
        List<String> hand = new ArrayList<>(ordered.subList(0, effHand));
        List<String> library = new ArrayList<>(ordered.subList(effHand, ordered.size()));
        return new Determinization(archetype, hand, library, posterior);
    }

    private static List<String> weightedShuffle(List<String> pool, Map<String, Float> weightByName, Random rng) {
        List<String> remaining = new ArrayList<>(pool == null ? Collections.emptyList() : pool);
        List<String> ordered = new ArrayList<>(remaining.size());
        Random safeRng = rng == null ? new Random() : rng;
        while (!remaining.isEmpty()) {
            double total = 0.0;
            for (String name : remaining) {
                total += cardWeight(name, weightByName);
            }
            if (total <= 0.0 || Double.isNaN(total) || Double.isInfinite(total)) {
                Collections.shuffle(remaining, safeRng);
                ordered.addAll(remaining);
                break;
            }
            double r = safeRng.nextDouble() * total;
            int pick = remaining.size() - 1;
            double seen = 0.0;
            for (int i = 0; i < remaining.size(); i++) {
                seen += cardWeight(remaining.get(i), weightByName);
                if (r <= seen) {
                    pick = i;
                    break;
                }
            }
            ordered.add(remaining.remove(pick));
        }
        return ordered;
    }

    private static double cardWeight(String name, Map<String, Float> weightByName) {
        float w = weightByName == null ? 0.0f : weightByName.getOrDefault(name, 0.0f);
        if (Float.isNaN(w) || Float.isInfinite(w) || w < 0.0f) {
            w = 0.0f;
        }
        return 0.05 + w;
    }

    /**
     * Exposed for tests: copy of the archetype's canonical decklist.
     */
    public Map<String, Integer> decklistCounts(String archetype) {
        Map<String, Integer> m = archetypeDecklists.get(archetype);
        return m == null ? Collections.emptyMap() : new HashMap<>(m);
    }

    private static String sampleFromPosterior(Map<String, Float> posterior, Random rng) {
        float r = rng.nextFloat();
        float cum = 0;
        String last = null;
        for (Map.Entry<String, Float> e : posterior.entrySet()) {
            last = e.getKey();
            cum += e.getValue();
            if (r <= cum) return e.getKey();
        }
        return last; // safety: return last entry if float rounding pushed past 1.0
    }

    // -----------------------------------------------------------------------
    // Visible-card collection helpers
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Applying a determinization to a game clone
    // -----------------------------------------------------------------------

    /**
     * Apply a determinization to a simulation-mode game clone: replace the
     * opponent's hand and library with the cards specified in {@code det}.
     * <p>
     * The opponent's visible zones (battlefield, graveyard, exile) are NOT
     * touched -- those are public info and stay as-is.
     * <p>
     * Our own library is shuffled (we know our hand but not our draw order).
     * <p>
     * MUST be called on a clone from {@code createSimulationForAI()}, never
     * on the live game.
     */
    public static void applyToClone(Game sim, UUID selfId, UUID oppId,
                                    Determinization det) {
        Player opp = sim.getPlayer(oppId);
        if (opp == null) return;

        // 1. Collect all of opp's hidden-zone cards (hand + library) into one pool.
        Map<String, List<Card>> poolByName = new HashMap<>();
        for (Card c : opp.getHand().getCards(sim)) {
            if (c != null) poolByName.computeIfAbsent(c.getName(), k -> new ArrayList<>()).add(c);
        }
        for (Card c : opp.getLibrary().getCards(sim)) {
            if (c != null) poolByName.computeIfAbsent(c.getName(), k -> new ArrayList<>()).add(c);
        }

        // 2. Clear opp's hand and library zones in the sim.
        opp.getHand().clear();
        opp.getLibrary().clear();

        // 3. Match determinization hand card names to actual Card objects.
        //    Each match consumes one Card from the pool (handles duplicates).
        List<Card> newHand = new ArrayList<>();
        for (String name : det.oppHandCards) {
            List<Card> available = poolByName.get(name);
            if (available != null && !available.isEmpty()) {
                Card card = available.remove(available.size() - 1);
                newHand.add(card);
            }
        }

        // 4. Remaining pool cards become the library (in determinization order
        //    if provided; otherwise just whatever's left, shuffled by caller).
        List<Card> newLibrary = new ArrayList<>();
        for (String name : det.oppLibraryOrder) {
            List<Card> available = poolByName.get(name);
            if (available != null && !available.isEmpty()) {
                Card card = available.remove(available.size() - 1);
                newLibrary.add(card);
            }
        }
        // Any leftover cards not matched by name (off-archetype) go to library end.
        for (List<Card> leftovers : poolByName.values()) {
            newLibrary.addAll(leftovers);
        }

        // 5. Assign hand cards with correct zone.
        for (Card card : newHand) {
            card.setZone(mage.constants.Zone.HAND, sim);
            opp.getHand().add(card);
        }
        // 6. Assign library cards. Library.addAll sets zone internally.
        opp.getLibrary().addAll(new java.util.LinkedHashSet<>(newLibrary), sim);

        // 7. Shuffle our own library (we know our hand but not draw order).
        Player self = sim.getPlayer(selfId);
        if (self != null) {
            self.getLibrary().shuffle();
        }
    }

    private static Map<String, Integer> collectVisibleCardCounts(Game game, UUID oppId) {
        Map<String, Integer> counts = new HashMap<>();
        if (game == null || oppId == null) return counts;
        Player opp = game.getPlayer(oppId);
        if (opp == null) return counts;

        // Battlefield (only opp's permanents).
        for (Permanent p : game.getBattlefield().getAllActivePermanents(oppId)) {
            if (p == null) continue;
            String name = p.getName();
            if (name == null || name.isEmpty()) continue;
            counts.merge(name, 1, Integer::sum);
        }
        // Graveyard.
        for (Card c : opp.getGraveyard().getCards(game)) {
            if (c == null) continue;
            counts.merge(c.getName(), 1, Integer::sum);
        }
        // Exile (only opp's own exile is "their cards"; the global exile
        // zone can contain cards from either player, so filter by owner).
        try {
            for (Card c : game.getExile().getAllCards(game)) {
                if (c != null && oppId.equals(c.getOwnerId())) {
                    counts.merge(c.getName(), 1, Integer::sum);
                }
            }
        } catch (Throwable ignored) {
            // Exile API quirks in some game states -- tolerate.
        }
        // Stack (only opp-controlled stack objects).
        try {
            for (StackObject so : game.getStack()) {
                if (so != null && oppId.equals(so.getControllerId())) {
                    String name = so.getName();
                    if (name != null && !name.isEmpty()) {
                        counts.merge(name, 1, Integer::sum);
                    }
                }
            }
        } catch (Throwable ignored) {
            // Stack iteration can throw during transient states -- tolerate.
        }
        return counts;
    }
}
