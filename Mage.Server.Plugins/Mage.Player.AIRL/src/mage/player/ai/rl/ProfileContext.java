package mage.player.ai.rl;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Per-profile mutable state for multi-profile training.
 * When multiple profiles train inside a single JVM, each gets its own
 * ProfileContext.  Game-runner threads set the context via {@link #setCurrent}
 * so that instance methods on RLTrainer (and helpers) can read per-profile
 * counters, winrate tracking, league state, etc.
 */
public final class ProfileContext {

    private static final ThreadLocal<ProfileContext> CURRENT = new ThreadLocal<>();

    public static ProfileContext current() {
        return CURRENT.get();
    }

    public static void setCurrent(ProfileContext ctx) {
        CURRENT.set(ctx);
    }

    // Identity
    public final String profileName;
    public final ProfilePaths paths;

    // Episode counters
    public final AtomicInteger episodeCounter = new AtomicInteger(0);
    public final AtomicInteger mulliganEpisodeCounter = new AtomicInteger(0);
    public final AtomicInteger activeEpisodes = new AtomicInteger(0);

    // Winrate tracking (rolling window)
    public final ConcurrentLinkedQueue<Boolean> recentWins = new ConcurrentLinkedQueue<>();
    public final AtomicInteger winCount = new AtomicInteger(0);

    // Adaptive curriculum
    public volatile RLTrainer.OpponentLevel currentOpponentLevel = RLTrainer.OpponentLevel.WEAK;
    public volatile String lastOpponentType = "UNKNOWN";
    public final AtomicInteger gamesAtCurrentLevel = new AtomicInteger(0);

    // League state
    public final Object leagueLock = new Object();
    public volatile RLTrainer.LeagueState leagueState = null;
    public final AtomicInteger leagueLastTickEp = new AtomicInteger(0);

    // Ladder state
    public final Object ladderLock = new Object();
    public volatile RLTrainer.LadderState ladderState = null;
    public final AtomicInteger ladderLastTickEp = new AtomicInteger(0);

    // Game runners for this profile
    public int numGameRunners;

    // Deck pools (shared references, not copied)
    public List<Path> deckFiles;
    public List<Path> agentDeckFiles;

    public ProfileContext(String profileName, ProfilePaths paths) {
        this.profileName = profileName;
        this.paths = paths;
    }
}
