# Opponent Configuration Analysis

## Current Setup

### Training Opponent (RLTrainer.java:324)
```java
opponentPlayer = new ComputerPlayer7("HeuristicBot", RangeOfInfluence.ALL, 3);
```

**ComputerPlayer7 with skill=3 means:**
- `maxDepth` = 4 (minimum depth, since skill < 4)
- `maxThinkTimeSecs` = 9 seconds (skill * 3)
- Uses minimax search with alpha-beta pruning
- Simulates up to 5,000 nodes per decision

### Curriculum Schedule (RLTrainer.java:981-988)
```
Episodes < 5000:    100% Heuristic opponent
Episodes 5000-15K:   50% Heuristic / 50% Self-play
Episodes >= 15K:    100% Self-play
```

## Performance Issues

### Think Time
- Heuristic bot takes **4+ seconds** per complex decision
- This is **within** the 9-second timeout but slows training
- Current speed: **~0.43 episodes/second** (~2.3 seconds per episode)
- Significant time spent waiting for opponent

### When It Gets Slow
The bot struggles with:
1. **Many targets** (Makeshift Munitions can target anything)
2. **Many artifacts/creatures** to sacrifice
3. **Deep decision trees** (multiple activations in sequence)
4. **Complex board states** (12+ permanents)

## Options for Improvement

### Option 1: Lower Skill Level (Fastest Fix)
```java
// Change from skill=3 to skill=1
new ComputerPlayer7("HeuristicBot", RangeOfInfluence.ALL, 1);
```
**Effect:**
- `maxDepth` = 4 (same, minimum)
- `maxThinkTimeSecs` = 3 seconds (instead of 9)
- Forces faster decisions but same search depth
- **Impact:** ~30-40% faster episodes

### Option 2: Use ComputerPlayer (Legacy, Simpler)
```java
// Much simpler heuristic, no simulations
new ComputerPlayer("SimpleBot", RangeOfInfluence.ALL);
```
**Effect:**
- No simulation/search tree
- Instant decisions
- Much weaker player
- **Impact:** ~70-80% faster episodes

### Option 3: Use Random Player (Fastest Baseline)
```java
new ComputerPlayerRandom("RandomBot", RangeOfInfluence.ALL);
```
**Effect:**
- Nearly instant decisions
- Very weak baseline
- Good for initial learning
- **Impact:** ~90% faster episodes

### Option 4: Curriculum Learning (RECOMMENDED)
Dynamically adjust opponent difficulty based on RL agent winrate:

```java
private Player createOpponent(double currentWinRate, Game game) {
    // Weak baseline for initial learning
    if (currentWinRate < 0.3) {
        return new ComputerPlayer("WeakBot", RangeOfInfluence.ALL);
    }
    // Medium opponent with fast timeout
    else if (currentWinRate < 0.5) {
        return new ComputerPlayer7("MediumBot", RangeOfInfluence.ALL, 1);
    }
    // Strong opponent with normal timeout
    else if (currentWinRate < 0.7) {
        return new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, 3);
    }
    // Self-play for advanced learning
    else {
        return new ComputerPlayerRL("SelfPlay", RangeOfInfluence.ALL, sharedModel);
    }
}
```

### Option 5: Reduce Search Depth (Custom)
Extend ComputerPlayer7 with:
```java
public class ComputerPlayer7Fast extends ComputerPlayer7 {
    public ComputerPlayer7Fast(String name, RangeOfInfluence range) {
        super(name, range, 1);
        this.maxDepth = 2;  // Override to be even shallower
        this.maxThinkTimeSecs = 2;  // 2-second timeout
    }
}
```

## Recommended Implementation

### Phase 1: Immediate Speed Improvement
**Change skill from 3 → 1** in all locations:
- Training: Line 324
- Eval: Line 523  
- Benchmark: Line 753

**Expected gain:** ~40% faster training

### Phase 2: Curriculum Learning
Track rolling winrate and adjust opponent:

```java
// Track last 100 games
private static final CircularFifoQueue<Boolean> recentWins = new CircularFifoQueue<>(100);

private Player createAdaptiveOpponent(int episodeNum, Random rand) {
    double winRate = recentWins.isEmpty() ? 0.0 : 
        recentWins.stream().filter(w -> w).count() / (double) recentWins.size();
    
    // First 500 episodes: weak opponent for bootstrapping
    if (episodeNum < 500) {
        return new ComputerPlayer("WeakBot", RangeOfInfluence.ALL);
    }
    // 500-5000: adaptive based on winrate
    else if (episodeNum < 5000) {
        if (winRate < 0.4) {
            return new ComputerPlayer("WeakBot", RangeOfInfluence.ALL);
        } else {
            return new ComputerPlayer7("MediumBot", RangeOfInfluence.ALL, 1);
        }
    }
    // 5000-15000: mix medium and strong
    else if (episodeNum < 15000) {
        if (winRate < 0.5) {
            return new ComputerPlayer7("MediumBot", RangeOfInfluence.ALL, 1);
        } else if (rand.nextDouble() < 0.5) {
            return new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, 3);
        } else {
            return new ComputerPlayerRL("SelfPlay", RangeOfInfluence.ALL, sharedModel);
        }
    }
    // 15000+: self-play with occasional strong opponent
    else {
        return rand.nextDouble() < 0.9 
            ? new ComputerPlayerRL("SelfPlay", RangeOfInfluence.ALL, sharedModel)
            : new ComputerPlayer7("StrongBot", RangeOfInfluence.ALL, 3);
    }
}
```

### Phase 3: Metrics & Monitoring
Add environment variables:
```bash
export OPPONENT_SKILL=1          # Override skill level
export ADAPTIVE_OPPONENT=1       # Enable curriculum
export WINRATE_WINDOW=100        # Rolling window size
```

## Expected Performance Gains

| Configuration | Episodes/sec | Speedup | Agent Strength |
|--------------|--------------|---------|----------------|
| Current (skill=3) | 0.43 | 1.0x | Strong |
| Skill=1 | 0.60 | 1.4x | Medium |
| ComputerPlayer | 1.20 | 2.8x | Weak |
| Random | 1.50 | 3.5x | Very Weak |
| **Adaptive (Recommended)** | **0.80** | **1.9x** | **Progressive** |

## Benefits of Curriculum Learning

1. **Fast early training** against weak opponents
2. **Smooth difficulty progression** as agent improves
3. **No wasted compute** on overly difficult or too-easy opponents
4. **Better exploration** in early episodes
5. **Maintains challenge** as agent gets stronger

## Implementation Status

✅ **IMPLEMENTED** - Adaptive curriculum with hysteresis and rolling winrate tracking

### What Was Implemented

1. **Thread-safe winrate tracking** using concurrent queue (100-game rolling window)

2. **Adaptive opponent selection with hysteresis** to prevent oscillation:
   - **Upgrade thresholds:**
     - 40% → MediumBot (CP7 skill=1)
     - 55% → StrongBot (CP7 skill=3)
     - 65% → SelfPlay
   - **Downgrade thresholds (5% gap):**
     - <35% → WeakBot
     - <50% → MediumBot  
     - <60% → StrongBot

3. **Hysteresis prevents bouncing** - Agent must drop 5% below upgrade threshold before downgrading

4. **Evaluation uses stronger opponents** (skill=6 by default)

5. **Comprehensive logging** of opponent transitions (upgrades/downgrades) and winrates

6. **Configurable via environment variables**

## Usage

### Default Behavior (Adaptive Curriculum)

Simply run training as normal - adaptive curriculum is **enabled by default**:

```bash
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java \
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" \
  "-Dexec.args=train"
```

The agent will automatically:
- Start against weak opponents
- Upgrade opponents as it improves
- Transition to self-play at ~65% winrate
- Log opponent changes: `"Opponent upgraded to: MEDIUM-CP7(skill=1) (winrate=0.425 over 100 games)"`

### Environment Variables

Control curriculum behavior:

```bash
# Adaptive curriculum (default=1)
export ADAPTIVE_CURRICULUM=1

# Rolling winrate window size (default=100)
export WINRATE_WINDOW=100

# Winrate thresholds for upgrading opponents
export THRESHOLD_WEAK_MEDIUM=0.40      # Upgrade to medium at 40%
export THRESHOLD_MEDIUM_STRONG=0.55    # Upgrade to strong at 55%
export THRESHOLD_STRONG_SELFPLAY=0.65  # Switch to self-play at 65%

# Evaluation/benchmark opponent strength (default=6)
export EVAL_OPPONENT_SKILL=6
export BENCHMARK_OPPONENT_SKILL=6
```

### Fixed Schedule (Legacy Mode)

Disable adaptive curriculum for fixed episode-based transitions:

```bash
export ADAPTIVE_CURRICULUM=0
export FIXED_WEAK_UNTIL=500       # Weak opponent until episode 500
export FIXED_MEDIUM_UNTIL=3000    # Medium opponent 500-3000
export FIXED_STRONG_UNTIL=8000    # Strong opponent 3000-8000
# Self-play after episode 8000
```

### Monitoring Progress

The training logs now include:
```
Episode 1523 summary: turns=18, reward=1.050, opponent=MEDIUM-CP7, winrate=0.447 (100 games)
Opponent upgraded to: STRONG-CP7(skill=3) (winrate=0.562 over 100 games, episode=1524)
```

CSV stats include opponent type and winrate:
```csv
episode,turns,final_reward,opponent_type,winrate,episode_seconds
1,12,1.050,CP_WEAK,0.000,2.34
100,15,-1.050,CP_WEAK,0.450,1.89
500,18,1.050,CP7,0.556,2.71
```

## Expected Performance

### Training Speed by Opponent

| Opponent Stage | Episodes/sec | Training Phase | Winrate Range |
|---------------|--------------|----------------|---------------|
| **WeakBot (CP)** | ~1.2 | Bootstrap | 0-40% |
| **MediumBot (CP7-1)** | ~0.8 | Fundamentals | 40-55% |
| **StrongBot (CP7-3)** | ~0.5 | Advanced | 55-65% |
| **Self-Play** | ~0.7 | Mastery | 65%+ |
| **Overall Average** | **~0.8** | Mixed | All |

### Compared to Old Fixed Schedule

| Metric | Old (Fixed skill=3) | New (Adaptive) | Improvement |
|--------|---------------------|----------------|-------------|
| Episodes/sec | 0.43 | 0.80 | **+86%** |
| Time to 10K eps | 6.5 hours | 3.5 hours | **46% faster** |
| Early learning | Slow (hard opponent) | Fast (easy start) | **3x faster** |
| Late learning | Limited (fixed opp) | Best (self-play) | Better exploration |

## Design Rationale

### Why Switch to Self-Play at 65%?

1. **Agent has fundamentals** - Can beat mid-level heuristic consistently
2. **Ready to explore** - Heuristic limitations become bottleneck
3. **Co-evolution begins** - Both agents improve together
4. **Not too early** - Avoids two weak agents learning bad habits

### Why Keep 10% Heuristic in Self-Play?

1. **Prevents mode collapse** - Diversity in training signal
2. **Maintains stability** - Heuristic provides consistent challenge
3. **Better generalization** - Agent doesn't overfit to self-play

### Why Reserve skill=6+ for Eval?

1. **Training efficiency** - Skill=6 thinks 18+ seconds (too slow)
2. **Stable benchmark** - Consistent evaluation target
3. **Curriculum design** - Agent graduates to self-play before hitting skill=6

## Implementation Priority

1. ✅ **DONE:** Adaptive curriculum with winrate tracking
2. ✅ **DONE:** Thread-safe concurrent implementation
3. ✅ **DONE:** Comprehensive logging and monitoring
4. ✅ **DONE:** Environment variable configuration
5. ✅ **DONE:** Stronger evaluation opponents (skill=6)
