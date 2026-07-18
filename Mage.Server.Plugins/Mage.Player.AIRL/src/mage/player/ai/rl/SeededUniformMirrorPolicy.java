package mage.player.ai.rl;

import java.io.Serializable;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Stateless-per-decision seeded uniform policy used only by the XMage mirror benchmark.
 *
 * <p>The seed derivation is byte-for-byte compatible with
 * {@code kernel-python-rl-seed-v2}. A physical decision owns one group seed;
 * every aggregate substep is derived independently from that group. Candidate
 * count therefore cannot perturb any later physical decision.</p>
 */
public final class SeededUniformMirrorPolicy implements Serializable {

    private static final long serialVersionUID = 1L;

    public static final String POLICY_ID = "seeded_uniform_mirror/v2";
    public static final String SEED_DERIVATION_VERSION = "kernel-python-rl-seed-v2";
    public static final long GOLDEN_RATIO_64 = 0x9E37_79B9_7F4A_7C15L;
    public static final long PHYSICAL_DECISION_MULTIPLIER = 0xD1B5_4A32_D192_ED03L;
    public static final long POLICY_SUBSTEP_MULTIPLIER = 0x94D0_49BB_1331_11EBL;
    public static final long ENV_SEED_DOMAIN = 0x4556_5F52_4C5F_7632L;
    public static final long UNIFORM_POLICY_DOMAIN = 0x5059_5F55_4E49_7632L;
    public static final long POLICY_SUBSTEP_DOMAIN = 0x5355_4253_5445_5032L;
    public static final long P0_SEAT_TAG = 0x5030L;
    public static final long P1_SEAT_TAG = 0x5031L;

    private final long baseSeed;
    private final long episodeId;
    private final String seat;
    private long physicalDecisionIndex;
    private long policyActionSelections;
    private long policyLeafEvaluations;
    private final Map<String, Long> physicalDecisionCategories = new LinkedHashMap<>();
    private final Map<String, Long> outcomeHistogram = new LinkedHashMap<>();

    public SeededUniformMirrorPolicy(long baseSeed, long episodeId, String seat) {
        validateUint63(baseSeed, "baseSeed");
        validateUint63(episodeId, "episodeId");
        validateSeat(seat);
        this.baseSeed = baseSeed;
        this.episodeId = episodeId;
        this.seat = seat;
    }

    private SeededUniformMirrorPolicy(SeededUniformMirrorPolicy source) {
        this.baseSeed = source.baseSeed;
        this.episodeId = source.episodeId;
        this.seat = source.seat;
        this.physicalDecisionIndex = source.physicalDecisionIndex;
        this.policyActionSelections = source.policyActionSelections;
        this.policyLeafEvaluations = source.policyLeafEvaluations;
        this.physicalDecisionCategories.putAll(source.physicalDecisionCategories);
        this.outcomeHistogram.putAll(source.outcomeHistogram);
    }

    public SeededUniformMirrorPolicy copy() {
        return new SeededUniformMirrorPolicy(this);
    }

    public int chooseNoncombat(String category, int canonicalLegalCount) {
        return chooseNoncombatWithoutReplacement(category, canonicalLegalCount, 1)[0];
    }

    /**
     * Select canonical ranks sequentially without replacement inside one physical menu.
     */
    public int[] chooseNoncombatWithoutReplacement(String category, int canonicalLegalCount, int picks) {
        if (canonicalLegalCount <= 0) {
            throw new IllegalArgumentException("canonicalLegalCount must be positive");
        }
        if (picks <= 0 || picks > canonicalLegalCount) {
            throw new IllegalArgumentException("picks must be in [1, canonicalLegalCount]");
        }
        String normalizedCategory = requireCategory(category);
        long groupSeed = beginPhysicalDecision(normalizedCategory);
        int[] remaining = new int[canonicalLegalCount];
        for (int i = 0; i < canonicalLegalCount; i++) {
            remaining[i] = i;
        }
        int[] selected = new int[picks];
        int remainingCount = canonicalLegalCount;
        StringBuilder selectedText = new StringBuilder();
        for (int substep = 0; substep < picks; substep++) {
            int remainingRank = unsignedModulo(leafSeed(groupSeed, substep), remainingCount);
            selected[substep] = remaining[remainingRank];
            if (selectedText.length() > 0) {
                selectedText.append('-');
            }
            selectedText.append(selected[substep]);
            System.arraycopy(remaining, remainingRank + 1, remaining, remainingRank,
                    remainingCount - remainingRank - 1);
            remainingCount--;
        }
        policyLeafEvaluations += picks;
        policyActionSelections += picks;
        increment(outcomeHistogram, normalizedCategory
                + "|legal=" + canonicalLegalCount + "|selected=" + selectedText);
        return selected;
    }

    public boolean[] chooseAttackers(int canonicalEligibleCount) {
        if (canonicalEligibleCount <= 0) {
            throw new IllegalArgumentException("canonicalEligibleCount must be positive");
        }
        long groupSeed = beginPhysicalDecision("declare_attackers");
        boolean[] included = new boolean[canonicalEligibleCount];
        int includedCount = 0;
        for (int i = 0; i < canonicalEligibleCount; i++) {
            included[i] = unsignedModulo(leafSeed(groupSeed, i), 2) == 1;
            if (included[i]) {
                includedCount++;
            }
        }
        policyLeafEvaluations += canonicalEligibleCount;
        policyActionSelections += canonicalEligibleCount;
        increment(outcomeHistogram, "declare_attackers|eligible=" + canonicalEligibleCount
                + "|included=" + includedCount);
        return included;
    }

    /**
     * @return {@code -1} for no block, otherwise the canonical blocker rank.
     */
    public int chooseBlocker(int canonicalLegalBlockerCount) {
        if (canonicalLegalBlockerCount <= 0) {
            throw new IllegalArgumentException("canonicalLegalBlockerCount must be positive");
        }
        long groupSeed = beginPhysicalDecision("declare_blocker_for_attacker");
        long gate = leafSeed(groupSeed, 0);
        policyLeafEvaluations++;
        policyActionSelections++;
        int selected = -1;
        if (unsignedModulo(gate, 100) < 35) {
            selected = unsignedModulo(leafSeed(groupSeed, 1), canonicalLegalBlockerCount);
            policyLeafEvaluations++;
        }
        increment(outcomeHistogram, "declare_blocker_for_attacker|legal="
                + canonicalLegalBlockerCount + "|selected=" + (selected < 0 ? "none" : selected));
        return selected;
    }

    private long beginPhysicalDecision(String category) {
        if (physicalDecisionIndex == Long.MAX_VALUE) {
            throw new IllegalStateException("physical decision index exhausted uint63 domain");
        }
        long groupSeed = deriveGroupSeed(baseSeed, episodeId, physicalDecisionIndex, seat);
        physicalDecisionIndex++;
        increment(physicalDecisionCategories, category);
        return groupSeed;
    }

    public long getPhysicalDecisionCount() {
        return physicalDecisionIndex;
    }

    public long getPolicyActionSelections() {
        return policyActionSelections;
    }

    public long getPolicyLeafEvaluations() {
        return policyLeafEvaluations;
    }

    public Map<String, Long> getPhysicalDecisionCategories() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(physicalDecisionCategories));
    }

    public Map<String, Long> getOutcomeHistogram() {
        return Collections.unmodifiableMap(new LinkedHashMap<>(outcomeHistogram));
    }

    public static long deriveEnvSeed(long baseSeed, long episodeId) {
        validateUint63(baseSeed, "baseSeed");
        validateUint63(episodeId, "episodeId");
        return derive(baseSeed, episodeId, 0L, P0_SEAT_TAG, ENV_SEED_DOMAIN);
    }

    public static long deriveGroupSeed(long baseSeed, long episodeId, long physicalDecision, String seat) {
        validateUint63(baseSeed, "baseSeed");
        validateUint63(episodeId, "episodeId");
        validateUint63(physicalDecision, "physicalDecision");
        return derive(baseSeed, episodeId, physicalDecision, seatTag(seat), UNIFORM_POLICY_DOMAIN);
    }

    public static long leafSeed(long groupSeed, long substepIndex) {
        if (substepIndex < 0L || substepIndex > 0xFFFF_FFFFL) {
            throw new IllegalArgumentException("substepIndex must be in uint32 domain");
        }
        long mixed = groupSeed
                ^ POLICY_SUBSTEP_DOMAIN
                ^ (substepIndex * POLICY_SUBSTEP_MULTIPLIER);
        return splitMix64Once(mixed);
    }

    public static long splitMix64Once(long seed) {
        long z = seed + GOLDEN_RATIO_64;
        z = (z ^ (z >>> 30)) * 0xBF58_476D_1CE4_E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D0_49BB_1331_11EBL;
        return z ^ (z >>> 31);
    }

    public static int unsignedModulo(long value, int bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("bound must be positive");
        }
        return (int) Long.remainderUnsigned(value, (long) bound);
    }

    private static long derive(long baseSeed, long episodeId, long physicalDecision,
                               long seatTag, long domain) {
        long mixed = baseSeed
                ^ domain
                ^ (episodeId * GOLDEN_RATIO_64)
                ^ (physicalDecision * PHYSICAL_DECISION_MULTIPLIER)
                ^ seatTag;
        return splitMix64Once(mixed);
    }

    private static long seatTag(String seat) {
        validateSeat(seat);
        return "p0".equals(seat) ? P0_SEAT_TAG : P1_SEAT_TAG;
    }

    private static void validateSeat(String seat) {
        if (!"p0".equals(seat) && !"p1".equals(seat)) {
            throw new IllegalArgumentException("seat must be exactly p0 or p1");
        }
    }

    private static void validateUint63(long value, String name) {
        if (value < 0L) {
            throw new IllegalArgumentException(name + " must be in [0, 2**63 - 1]");
        }
    }

    private static String requireCategory(String category) {
        if (category == null || category.trim().isEmpty()) {
            throw new IllegalArgumentException("category must be nonempty");
        }
        return category.trim();
    }

    private static void increment(Map<String, Long> map, String key) {
        Long old = map.get(key);
        map.put(key, old == null ? 1L : old + 1L);
    }

    /** Fixed vectors generated from the authoritative Python v2 implementation. */
    public static void runFixedGoldenSelfTest() {
        assertUnsignedHex("6bbfb0c0fc58c50c", deriveEnvSeed(71_501L, 0L), "env(71501,0)");
        assertUnsignedHex("1f715016d3bd86dd", deriveEnvSeed(71_501L, 1L), "env(71501,1)");
        long p0Group = deriveGroupSeed(71_501L, 0L, 0L, "p0");
        long p1Group = deriveGroupSeed(71_501L, 0L, 0L, "p1");
        assertUnsignedHex("a70305ab1de383fc", p0Group, "group p0");
        assertUnsignedHex("a4668aedf2a77373", p1Group, "group p1");
        assertUnsignedHex("feaaa9f4fbe02bd4", leafSeed(p0Group, 0L), "leaf p0/0");
        assertUnsignedHex("65d383bac2c5e8f0", leafSeed(p0Group, 1L), "leaf p0/1");
        assertUnsignedHex("019ec8ee5277b4ee", leafSeed(p1Group, 0L), "leaf p1/0");
        assertUnsignedHex("f11b1a297a7289fa", leafSeed(p1Group, 1L), "leaf p1/1");
        if (unsignedModulo(leafSeed(p0Group, 0L), 7) != 6) {
            throw new IllegalStateException("unsigned modulo golden mismatch");
        }
        SeededUniformMirrorPolicy policy = new SeededUniformMirrorPolicy(71_501L, 0L, "p0");
        if (policy.chooseNoncombat("self_test", 7) != 6
                || policy.getPhysicalDecisionCount() != 1L
                || policy.getPolicyLeafEvaluations() != 1L) {
            throw new IllegalStateException("policy state golden mismatch");
        }
    }

    private static void assertUnsignedHex(String expected, long actual, String label) {
        String observed = String.format("%016x", actual);
        if (!expected.equals(observed)) {
            throw new IllegalStateException(label + " expected " + expected + " but got " + observed);
        }
    }
}
