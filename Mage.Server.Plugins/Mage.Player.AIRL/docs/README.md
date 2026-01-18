# RL Player AI Documentation

## Quick Links

- **[Curriculum Usage Guide](curriculum_usage.md)** - How to use the adaptive curriculum
- **[Commands Reference](commands.md)** - Training, evaluation, and build commands
- **[Hysteresis Explained](hysteresis_explained.md)** - Why opponent switching is controlled with hysteresis
- **[Opponent Analysis](opponent_analysis.md)** - Technical details and design decisions
- **[Changelog](CHANGELOG_curriculum.md)** - What changed in the implementation

## Overview

This RL agent uses an **adaptive curriculum** to train efficiently against progressively harder opponents. The curriculum automatically adjusts based on the agent's current performance (rolling 100-game winrate).

## Training Progression

```
WeakBot          MediumBot         StrongBot         Self-Play
(instant)        (skill=1)         (skill=3)         (co-evolution)
   |                |                  |                  |
   |-- 40% WR ----->|-- 55% WR ------>|-- 65% WR ------->|
   |                |                  |                  |
Episodes 0-500   500-2000          2000-4000          4000+
(~1.2 eps/s)     (~0.8 eps/s)      (~0.5 eps/s)      (~0.7 eps/s)
```

## Key Features

1. **Automatic Difficulty Scaling** - No manual episode thresholds
2. **Fast Early Training** - Starts with weak opponent (~3x faster episodes)
3. **Hysteresis Control** - 5% gaps prevent opponent oscillation (upgrade at 40%, downgrade at 35%)
4. **Self-Play Graduation** - Transitions at 65% winrate for advanced learning
5. **Strong Evaluation** - Uses skill=6 for benchmarking (not training)
6. **Comprehensive Logging** - Tracks winrate, opponent type, transitions

## Quick Start

### Default Training (Recommended)
```powershell
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

### Custom Thresholds
```powershell
$env:THRESHOLD_WEAK_MEDIUM="0.35"
$env:THRESHOLD_MEDIUM_STRONG="0.50"
$env:THRESHOLD_STRONG_SELFPLAY="0.60"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

### Evaluation
```powershell
$env:EVAL_OPPONENT_SKILL="6"
$env:EVAL_EPISODES="100"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=eval"
```

## Configuration Reference

### Adaptive Curriculum (Enabled by Default)

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTIVE_CURRICULUM` | 1 | Enable (1) or disable (0) |
| `WINRATE_WINDOW` | 100 | Rolling window size |
| **Upgrade Thresholds** | | |
| `THRESHOLD_WEAK_MEDIUM` | 0.40 | Upgrade to medium opponent |
| `THRESHOLD_MEDIUM_STRONG` | 0.55 | Upgrade to strong opponent |
| `THRESHOLD_STRONG_SELFPLAY` | 0.65 | Switch to self-play |
| **Downgrade Thresholds (Hysteresis)** | | |
| `THRESHOLD_MEDIUM_WEAK` | 0.35 | Downgrade to weak (5% gap prevents oscillation) |
| `THRESHOLD_STRONG_MEDIUM` | 0.50 | Downgrade to medium (5% gap) |
| `THRESHOLD_SELFPLAY_STRONG` | 0.60 | Downgrade to strong (5% gap) |

### Evaluation

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_OPPONENT_SKILL` | 6 | Skill level for eval (1-10) |
| `BENCHMARK_OPPONENT_SKILL` | 6 | Skill level for benchmark |
| `EVAL_EPISODES` | 5 | Number of eval games |

### Legacy Fixed Schedule

| Variable | Default | Description |
|----------|---------|-------------|
| `FIXED_WEAK_UNTIL` | 500 | Weak opponent episodes |
| `FIXED_MEDIUM_UNTIL` | 3000 | Medium opponent episodes |
| `FIXED_STRONG_UNTIL` | 8000 | Strong opponent episodes |

## Expected Performance

### Training Speed Comparison

| Configuration | Eps/Sec | Training Time (10K eps) | Speedup |
|--------------|---------|-------------------------|---------|
| Old (fixed skill=3) | 0.43 | 6.5 hours | 1.0x |
| **New (adaptive)** | **0.80** | **3.5 hours** | **1.86x** |

### Learning Quality
- **Better fundamentals** - Smooth difficulty progression
- **Faster convergence** - Efficient use of training time
- **Superior final policy** - Self-play exploration phase

## Monitoring

### Log Output
```
[INFO] Opponent: WEAK-ComputerPlayer (bootstrap phase, 25/100 games)
[INFO] Episode 523 summary: turns=14, reward=1.050, opponent=CP_WEAK, winrate=0.398 (100 games)
[INFO] Opponent upgraded to: MEDIUM-CP7(skill=1) (winrate=0.402 over 100 games, episode=524)
[INFO] Episode 1847 summary: turns=16, reward=-1.050, opponent=CP7, winrate=0.548 (100 games)
[INFO] Opponent upgraded to: STRONG-CP7(skill=3) (winrate=0.557 over 100 games, episode=1848)
[INFO] Episode 3291 summary: turns=22, reward=1.050, opponent=CP7, winrate=0.643 (100 games)
[INFO] Opponent upgraded to: SELFPLAY (winrate=0.652 over 100 games, episode=3292)
```

### CSV Statistics
```csv
episode,turns,final_reward,opponent_type,winrate,episode_seconds
1,12,1.050,CP_WEAK,0.000,1.89
100,15,-1.050,CP_WEAK,0.450,1.72
500,18,1.050,CP7,0.556,2.34
1000,20,-1.050,CP7,0.489,2.71
3000,22,1.050,SELFPLAY,0.667,2.15
```

## Troubleshooting

### Stuck at Low Winrate
- Check model training (losses decreasing)
- Verify deck quality (balanced, playable)
- Consider lowering thresholds temporarily
- Check for game errors in logs

### Upgrades Too Fast/Slow
- Increase/decrease `WINRATE_WINDOW`
- Adjust threshold values
- Check opponent think times (may be bottleneck)

### Training Too Slow
- Monitor which opponent is active
- WeakBot should be fastest (~1.2 eps/s)
- Self-play should be fast (~0.7 eps/s)
- Skill=3 is slowest (~0.5 eps/s)

## Best Practices

1. **Use default settings first** - They're well-tuned
2. **Monitor first 1000 episodes** - Should see clear progression
3. **Expect self-play by 3000-5000** - Depends on decks/matchups
4. **Use skill=6 for evaluation** - Good benchmark balance
5. **Track opponent transitions** - Indicates learning progress

## Architecture

### Key Components

- **`recordGameOutcome()`** - Thread-safe winrate tracking
- **`getCurrentWinrate()`** - Rolling average calculation
- **`createAdaptiveOpponent()`** - Opponent selection logic
- **`recentWins`** - Concurrent circular buffer
- **`winCount`** - Atomic counter for efficiency

### Thread Safety

All curriculum components are thread-safe for parallel training:
- `ConcurrentLinkedQueue` for win buffer
- `AtomicInteger` for win counting
- Synchronized logging for transitions

## Further Reading

- [Curriculum Usage Guide](curriculum_usage.md) - Detailed usage examples
- [Opponent Analysis](opponent_analysis.md) - Design rationale and performance data
- [Changelog](CHANGELOG_curriculum.md) - Complete implementation details
