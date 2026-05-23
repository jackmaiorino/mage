package mage.player.ai.rl;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Smoke test for DeterminizationSampler. Run with:
 *   mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java \
 *     -Dexec.mainClass=mage.player.ai.rl.DeterminizationSamplerTest
 * <p>
 * Uses only the pure-function overloads -- no Game instance required.
 */
public final class DeterminizationSamplerTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        DeterminizationSampler s = DeterminizationSampler.pauperDefaults();
        if (s == null || s.getArchetypes().isEmpty()) {
            System.err.println("FATAL: could not load archetype decklists.");
            System.exit(1);
        }
        System.out.println("Loaded archetypes: " + s.getArchetypes());
        for (String arch : s.getArchetypes()) {
            int total = s.decklistCounts(arch).values().stream().mapToInt(Integer::intValue).sum();
            System.out.println("  " + arch + " : " + total + " cards, "
                    + s.decklistCounts(arch).size() + " unique names");
        }
        System.out.println();

        testEmptyObservationUniform(s);
        testSignatureCardUniqueArchetype(s);
        testMultipleConsistentArchetypes(s);
        testOffMetaFallback(s);
        testCountExceedsDecklistEliminates(s);
        testDeterminizationValidity(s);
        testDeterminizationWithVisibleSubtraction(s);
        testLoadFromDeckListFile();

        System.out.println();
        System.out.println("Total: " + (passed + failed) + "  Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) System.exit(1);
    }

    private static void testEmptyObservationUniform(DeterminizationSampler s) {
        Map<String, Float> p = s.classifyArchetypeFromCounts(new HashMap<>());
        float expected = 1.0f / s.getArchetypes().size();
        for (Map.Entry<String, Float> e : p.entrySet()) {
            if (!approxEq(e.getValue(), expected, 0.001f)) {
                fail("empty-observation-uniform", "archetype=" + e.getKey() + " got=" + e.getValue());
                return;
            }
        }
        pass("empty-observation-uniform");
    }

    private static void testSignatureCardUniqueArchetype(DeterminizationSampler s) {
        String[][] cases = {
                {"Cleansing Wildfire", "Wildfire"},
                {"Elvish Mystic", "Elves"},
                {"Goblin Bushwhacker", "Rally"},
                {"Myr Enforcer", "Affinity"},
                {"Balustrade Spy", "SpyCombo"},
                {"Tolarian Terror", "Terror"},
                {"Basilisk Gate", "CawGates"},
        };
        for (String[] c : cases) {
            Map<String, Integer> obs = new HashMap<>();
            obs.put(c[0], 1);
            Map<String, Float> p = s.classifyArchetypeFromCounts(obs);
            if (!approxEq(p.getOrDefault(c[1], 0f), 1.0f, 0.001f)) {
                fail("signature-" + c[1].toLowerCase(),
                        "card=" + c[0] + " expected " + c[1] + "=1.0 got " + p);
                return;
            }
        }
        pass("signature-card-unique-archetype");
    }

    private static void testMultipleConsistentArchetypes(DeterminizationSampler s) {
        Map<String, Integer> obs = new HashMap<>();
        obs.put("Mountain", 1);
        Map<String, Float> p = s.classifyArchetypeFromCounts(obs);
        float sum = 0;
        int survivors = 0;
        for (float v : p.values()) {
            sum += v;
            if (v > 0.01f) survivors++;
        }
        if (!approxEq(sum, 1.0f, 0.001f)) {
            fail("multiple-consistent-normalized", "sum=" + sum);
            return;
        }
        if (survivors < 2) {
            fail("multiple-consistent-survivors", "expected >= 2, got " + survivors + " post=" + p);
            return;
        }
        pass("multiple-consistent-archetypes");
    }

    private static void testOffMetaFallback(DeterminizationSampler s) {
        Map<String, Integer> obs = new HashMap<>();
        obs.put("Jace, the Mind Sculptor", 1);
        Map<String, Float> p = s.classifyArchetypeFromCounts(obs);
        float expected = 1.0f / s.getArchetypes().size();
        for (float v : p.values()) {
            if (!approxEq(v, expected, 0.001f)) {
                fail("off-meta-fallback", "got " + p);
                return;
            }
        }
        pass("off-meta-fallback-uniform");
    }

    private static void testCountExceedsDecklistEliminates(DeterminizationSampler s) {
        Map<String, Integer> obs = new HashMap<>();
        obs.put("Cleansing Wildfire", 5);  // max possible is 4
        Map<String, Float> p = s.classifyArchetypeFromCounts(obs);
        float expected = 1.0f / s.getArchetypes().size();
        for (float v : p.values()) {
            if (!approxEq(v, expected, 0.001f)) {
                fail("count-exceeds-fallback", "got " + p);
                return;
            }
        }
        pass("count-exceeds-decklist-fallback");
    }

    private static void testDeterminizationValidity(DeterminizationSampler s) {
        String archetype = "Wildfire";
        Map<String, Integer> decklist = s.decklistCounts(archetype);
        int deckSize = decklist.values().stream().mapToInt(Integer::intValue).sum();
        Random rng = new Random(42);
        Map<String, Float> uniform = uniform(s);
        for (int iter = 0; iter < 20; iter++) {
            DeterminizationSampler.Determinization d = s.sampleForArchetypePure(
                    archetype, new HashMap<>(), 7, uniform, rng);
            if (d.oppHandCards.size() != 7) {
                fail("determinization-hand-size",
                        "iter=" + iter + " size=" + d.oppHandCards.size());
                return;
            }
            int total = d.oppHandCards.size() + d.oppLibraryOrder.size();
            if (total != deckSize) {
                fail("determinization-total-size",
                        "iter=" + iter + " total=" + total + " expected=" + deckSize);
                return;
            }
            Map<String, Integer> counted = new HashMap<>();
            for (String c : d.oppHandCards) counted.merge(c, 1, Integer::sum);
            for (String c : d.oppLibraryOrder) counted.merge(c, 1, Integer::sum);
            for (Map.Entry<String, Integer> e : counted.entrySet()) {
                int allowed = decklist.getOrDefault(e.getKey(), 0);
                if (e.getValue() > allowed) {
                    fail("determinization-card-over-count",
                            "card=" + e.getKey() + " count=" + e.getValue() + " allowed=" + allowed);
                    return;
                }
                if (allowed == 0) {
                    fail("determinization-card-not-in-deck", "card=" + e.getKey());
                    return;
                }
            }
        }
        pass("determinization-validity-20-samples");
    }

    private static void testDeterminizationWithVisibleSubtraction(DeterminizationSampler s) {
        String archetype = "Wildfire";
        // Simulate: 3 Cleansing Wildfire already cast (in graveyard), 2 forests on board.
        Map<String, Integer> visible = new HashMap<>();
        visible.put("Cleansing Wildfire", 3);
        visible.put("Forest", 2);
        Random rng = new Random(7);
        int handSize = 5;
        for (int iter = 0; iter < 10; iter++) {
            DeterminizationSampler.Determinization d = s.sampleForArchetypePure(
                    archetype, visible, handSize, uniform(s), rng);
            long cwHand = d.oppHandCards.stream().filter(c -> c.equals("Cleansing Wildfire")).count();
            long cwLib = d.oppLibraryOrder.stream().filter(c -> c.equals("Cleansing Wildfire")).count();
            int deckCw = s.decklistCounts(archetype).getOrDefault("Cleansing Wildfire", 0);
            int remaining = deckCw - 3;
            if (cwHand + cwLib != remaining) {
                fail("visible-subtraction-cw-count",
                        "iter=" + iter + " cwHand=" + cwHand + " cwLib=" + cwLib +
                        " expected=" + remaining);
                return;
            }
            long fHand = d.oppHandCards.stream().filter(c -> c.equals("Forest")).count();
            long fLib = d.oppLibraryOrder.stream().filter(c -> c.equals("Forest")).count();
            int deckF = s.decklistCounts(archetype).getOrDefault("Forest", 0);
            int remF = Math.max(0, deckF - 2);
            if (fHand + fLib != remF) {
                fail("visible-subtraction-forest-count",
                        "iter=" + iter + " fHand=" + fHand + " fLib=" + fLib + " expected=" + remF);
                return;
            }
            if (d.oppHandCards.size() != handSize) {
                fail("visible-subtraction-hand-size",
                        "iter=" + iter + " got=" + d.oppHandCards.size());
                return;
            }
        }
        pass("determinization-with-visible-subtraction");
    }

    private static void testLoadFromDeckListFile() {
        String path = "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/"
                + "decklist.active_profile_pool_thesis_eval_unique_20260510.txt";
        DeterminizationSampler s = DeterminizationSampler.loadFromDeckListFile(path);
        if (s == null || s.getArchetypes().size() != 4) {
            fail("load-from-deck-list-file", "expected 4 archetypes, got "
                    + (s == null ? "null" : s.getArchetypes()));
            return;
        }
        if (!s.getArchetypes().contains("SpyCombo")
                || !s.getArchetypes().contains("JundWildfire")
                || !s.getArchetypes().contains("MonoRedRally")
                || !s.getArchetypes().contains("GrixisAffinity")) {
            fail("load-from-deck-list-file", "unexpected archetypes " + s.getArchetypes());
            return;
        }
        pass("load-from-deck-list-file");
    }

    private static boolean approxEq(float a, float b, float eps) {
        return Math.abs(a - b) <= eps;
    }

    private static void pass(String name) {
        System.out.println("  PASS  " + name);
        passed++;
    }

    private static void fail(String name, String why) {
        System.err.println("  FAIL  " + name + " : " + why);
        failed++;
    }

    private static Map<String, Float> uniform(DeterminizationSampler s) {
        Map<String, Float> p = new HashMap<>();
        float u = 1.0f / s.getArchetypes().size();
        for (String a : s.getArchetypes()) p.put(a, u);
        return p;
    }
}
