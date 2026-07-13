package mage.player.ai.rl;

/**
 * Smoke test for the Sol #92 completion-aware population scheduler's pure
 * functions (RLTrainer.pickSourceByShare / pickCompletionAwareSourceFromState
 * / popSourceIndexForKey). These take an explicit state snapshot instead of
 * reading RLTrainer's mutable static POP_* fields, so they're testable here
 * without spinning up a trainer. Run with:
 *   mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests exec:java \
 *     -Dexec.mainClass=mage.player.ai.rl.PopulationSchedulerTest
 */
public final class PopulationSchedulerTest {

    private static int passed = 0;
    private static int failed = 0;

    public static void main(String[] args) {
        testPopSourceIndexForKey();
        testPickSourceByShareBoundaries();
        testCompletionAwareColdStartMatchesShare();
        testCompletionAwareNeverPicksZeroShareSource();
        testCompletionAwareCorrectsOverrepresentedSource();
        testCompletionAwareReservesMoreConcurrencyForSlowSource();
        testCompletionAwareThroughputRatioMatchesShareDespiteDurationSkew();
        testColdStartWarmupGuard();
        testSmallRunnerPoolOffByOneRegression();

        System.out.println();
        System.out.println("Total: " + (passed + failed) + "  Passed: " + passed + "  Failed: " + failed);
        if (failed > 0) System.exit(1);
    }

    private static void testPopSourceIndexForKey() {
        if (RLTrainer.popSourceIndexForKey("cp7:s7") != RLTrainer.POP_SRC_ANCHOR) {
            fail("pop-source-index-anchor", "cp7:s7 did not map to ANCHOR");
            return;
        }
        if (RLTrainer.popSourceIndexForKey("live") != RLTrainer.POP_SRC_LIVE) {
            fail("pop-source-index-live", "'live' did not map to LIVE");
            return;
        }
        if (RLTrainer.popSourceIndexForKey("snap:/tmp/snapshot_step_100.pt") != RLTrainer.POP_SRC_FROZEN) {
            fail("pop-source-index-frozen", "snap: key did not map to FROZEN");
            return;
        }
        if (RLTrainer.popSourceIndexForKey(null) != -1) {
            fail("pop-source-index-null", "null key did not map to -1");
            return;
        }
        pass("pop-source-index-for-key");
    }

    private static void testPickSourceByShareBoundaries() {
        double[] shares = {0.50, 0.35, 0.15};
        int[] expectSrc = {RLTrainer.POP_SRC_ANCHOR, RLTrainer.POP_SRC_ANCHOR,
                RLTrainer.POP_SRC_FROZEN, RLTrainer.POP_SRC_FROZEN, RLTrainer.POP_SRC_LIVE};
        double[] u = {0.0, 0.4999, 0.51, 0.8499, 0.86};
        for (int i = 0; i < u.length; i++) {
            int got = RLTrainer.pickSourceByShare(shares, u[i]);
            if (got != expectSrc[i]) {
                fail("pick-source-by-share-boundary", "u=" + u[i] + " expected=" + expectSrc[i] + " got=" + got);
                return;
            }
        }
        pass("pick-source-by-share-boundaries");
    }

    private static void testCompletionAwareColdStartMatchesShare() {
        // No history yet (uniform duration prior, zero in-flight): should
        // behave like plain share-proportional selection -- the argmax
        // deficit picks the largest share when nothing is in flight.
        double[] shares = {0.50, 0.35, 0.15};
        double[] durationEma = {1.0, 1.0, 1.0};
        int[] inFlight = {0, 0, 0};
        int got = RLTrainer.pickCompletionAwareSourceFromState(shares, durationEma, inFlight);
        if (got != RLTrainer.POP_SRC_ANCHOR) {
            fail("completion-aware-cold-start", "expected ANCHOR (largest share) at cold start, got " + got);
            return;
        }
        pass("completion-aware-cold-start-matches-share");
    }

    private static void testCompletionAwareNeverPicksZeroShareSource() {
        // Pool-empty spec: frozen share folded to 0 (85/0/15 per CLAUDE.md
        // note). Even if frozen is starved (0 in-flight) it must never be
        // picked once its target share is zero.
        double[] shares = {0.85, 0.0, 0.15};
        double[] durationEma = {1.0, 1.0, 1.0};
        int[] inFlight = {8, 0, 0}; // all runners currently on anchor
        int got = RLTrainer.pickCompletionAwareSourceFromState(shares, durationEma, inFlight);
        if (got == RLTrainer.POP_SRC_FROZEN) {
            fail("completion-aware-never-zero-share", "picked FROZEN despite zero target share");
            return;
        }
        pass("completion-aware-never-picks-zero-share-source");
    }

    private static void testCompletionAwareCorrectsOverrepresentedSource() {
        // Anchor is already over its target concurrency share (10/10 runners
        // busy with anchor vs a 50% target) -- the next freed runner must go
        // to whichever of frozen/live is furthest under its target, not anchor.
        double[] shares = {0.50, 0.35, 0.15};
        double[] durationEma = {1.0, 1.0, 1.0};
        int[] inFlight = {10, 0, 0};
        int got = RLTrainer.pickCompletionAwareSourceFromState(shares, durationEma, inFlight);
        if (got == RLTrainer.POP_SRC_ANCHOR) {
            fail("completion-aware-corrects-overrepresented", "kept picking ANCHOR while it was already over-reserved");
            return;
        }
        // Frozen (0.35) is further under-served than live (0.15) -- expect frozen.
        if (got != RLTrainer.POP_SRC_FROZEN) {
            fail("completion-aware-corrects-overrepresented", "expected FROZEN (largest deficit), got " + got);
            return;
        }
        pass("completion-aware-corrects-overrepresented-source");
    }

    private static void testCompletionAwareReservesMoreConcurrencyForSlowSource() {
        // Catastrophic-case shape: anchor games take 100x as long as frozen/
        // live. The target CONCURRENCY reservation for anchor should swell
        // to absorb almost all runner slots (~99%) -- this is the mechanism
        // that keeps anchor's COMPLETION rate at its 50% share despite being
        // 100x slower per game (see the throughput test below for the proof).
        double[] shares = {0.50, 0.35, 0.15};
        double[] durationEma = {100.0, 1.0, 1.0};
        double[] weight = new double[3];
        double weightTotal = 0;
        for (int i = 0; i < 3; i++) {
            weight[i] = shares[i] * durationEma[i];
            weightTotal += weight[i];
        }
        double anchorTargetFrac = weight[0] / weightTotal;
        if (anchorTargetFrac < 0.95) {
            fail("completion-aware-slow-source-reservation",
                    "expected anchor target concurrency share > 0.95 with 100x duration skew, got " + anchorTargetFrac);
            return;
        }
        // With inFlight already well ABOVE that reservation (150 anchor / 0
        // live in flight -- 150/151 capacity-adjusted -- comfortably clears
        // the 99.0099% target), anchor should no longer be preferentially
        // picked over a starved fast source.
        int[] inFlight = {150, 0, 0};
        int got = RLTrainer.pickCompletionAwareSourceFromState(shares, durationEma, inFlight);
        if (got == RLTrainer.POP_SRC_ANCHOR) {
            fail("completion-aware-slow-source-reservation", "picked ANCHOR even though it was already at its (huge) reservation");
            return;
        }
        pass("completion-aware-reserves-more-concurrency-for-slow-source (target=" + String.format("%.3f", anchorTargetFrac) + ")");
    }

    /**
     * End-to-end algebraic check of the core claim behind mechanism (ii):
     * reserving concurrency proportional to share_i * duration_i makes the
     * long-run COMPLETION RATE ratio between sources equal share_i : share_j,
     * regardless of duration_i, because completion_rate_i = (N * targetFrac_i)
     * / duration_i and the duration_i cancels against the targetFrac_i
     * numerator's duration_i factor. This is the formula the mid-run smoke
     * test is expected to reproduce empirically.
     */
    private static void testCompletionAwareThroughputRatioMatchesShareDespiteDurationSkew() {
        double[] shares = {0.50, 0.35, 0.15};
        double[] durationEma = {100.0, 1.0, 1.0}; // anchor 100x slower
        double[] weight = new double[3];
        double weightTotal = 0;
        for (int i = 0; i < 3; i++) {
            weight[i] = shares[i] * durationEma[i];
            weightTotal += weight[i];
        }
        double[] targetFrac = new double[3];
        double[] completionRate = new double[3]; // proportional to N * targetFrac_i / duration_i
        for (int i = 0; i < 3; i++) {
            targetFrac[i] = weight[i] / weightTotal;
            completionRate[i] = targetFrac[i] / durationEma[i];
        }
        double rateTotal = completionRate[0] + completionRate[1] + completionRate[2];
        for (int i = 0; i < 3; i++) {
            double impliedShare = completionRate[i] / rateTotal;
            if (Math.abs(impliedShare - shares[i]) > 1e-9) {
                fail("completion-aware-throughput-ratio", "source=" + i
                        + " implied completion share=" + impliedShare + " expected=" + shares[i]);
                return;
            }
        }
        pass("completion-aware-throughput-ratio-matches-share-despite-100x-duration-skew");
    }

    /**
     * Regression test for a real bug caught during the smoke run: durationEma
     * starts equal (1.0) for every source. If the fast source (live)
     * completes first, its EMA snaps to its true low duration while the slow
     * source (anchor) is still parked at the stale 1.0 placeholder -- a raw
     * pickCompletionAwareSourceFromState call at that point would compare
     * anchor's fake "1.0s" duration against live's real ~15s duration and
     * conclude LIVE deserves the bigger reservation, inverting the intended
     * 85/15 split to something like 16/84 (this is exactly what the smoke
     * run showed before allPopSourcesWarm was added). Verifies the warmup
     * gate correctly detects "anchor never completed" and would force a
     * fallback to the plain share draw in that state.
     */
    private static void testColdStartWarmupGuard() {
        double[] shares = {0.85, 0.0, 0.15}; // pool-empty spec (anchor absorbs frozen's mass)
        long[] noneCompletedYet = {0, 0, 0};
        if (RLTrainer.allPopSourcesWarm(shares, noneCompletedYet)) {
            fail("cold-start-warmup-guard", "expected NOT warm when nothing has completed yet");
            return;
        }
        long[] onlyLiveCompleted = {0, 0, 12}; // the failure mode: fast source got 12 samples, anchor 0
        if (RLTrainer.allPopSourcesWarm(shares, onlyLiveCompleted)) {
            fail("cold-start-warmup-guard", "expected NOT warm while anchor (nonzero share) has zero samples");
            return;
        }
        long[] frozenNeverNeeded = {5, 0, 3}; // frozen has zero share -- must not block warmup
        if (!RLTrainer.allPopSourcesWarm(shares, frozenNeverNeeded)) {
            fail("cold-start-warmup-guard", "frozen has zero share and should be ignored, but blocked warmup");
            return;
        }
        long[] allCompleted = {5, 0, 3};
        if (!RLTrainer.allPopSourcesWarm(shares, allCompleted)) {
            fail("cold-start-warmup-guard", "expected warm once every nonzero-share source has >=1 sample");
            return;
        }
        pass("cold-start-warmup-guard");
    }

    /**
     * Regression test for the exact bug caught live in the Sol #92 smoke run
     * (4 runners, real durations: anchor EMA~130s, live EMA~1.7s, pool-empty
     * spec 85/15). Once 3 of the 4 runners were parked on anchor (the 4th
     * mid-pick), the OLD formula divided by inFlightTotal=3 (excluding the
     * picking runner), read anchor as "already at 100%, over its ~99.8%
     * target," and diverted the 4th runner to live -- forever, because the
     * other 3 anchor games (each ~130s) never freed up fast enough relative
     * to how quickly the 4th runner cycled through ~1.7s live games and
     * re-asked. Empirically this produced ~85% LIVE completions instead of
     * ~85% anchor: a complete inversion of the spec. The capacity-plus-one
     * fix (dividing by inFlightTotal+1, since the picking runner isn't in
     * inFlight yet) must read this exact state as anchor still under its
     * target (3/4=75% &lt; 99.8%) and send the 4th runner back to anchor.
     */
    private static void testSmallRunnerPoolOffByOneRegression() {
        double[] shares = {0.85, 0.0, 0.15}; // pool-empty spec
        double[] durationEma = {130.0, 1.0, 1.7};
        int[] inFlight = {3, 0, 0}; // 3 of 4 runners on anchor; 4th is mid-pick
        int got = RLTrainer.pickCompletionAwareSourceFromState(shares, durationEma, inFlight);
        if (got != RLTrainer.POP_SRC_ANCHOR) {
            fail("small-runner-pool-off-by-one-regression",
                    "expected ANCHOR (3/4=75% is still under its ~99.8% target) but got " + got
                            + " -- this is the exact inversion bug caught in the live smoke run");
            return;
        }
        pass("small-runner-pool-off-by-one-regression");
    }

    private static void pass(String name) {
        System.out.println("  PASS  " + name);
        passed++;
    }

    private static void fail(String name, String why) {
        System.err.println("  FAIL  " + name + " : " + why);
        failed++;
    }
}
