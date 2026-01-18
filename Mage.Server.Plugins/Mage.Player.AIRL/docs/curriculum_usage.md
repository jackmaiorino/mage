# Adaptive Curriculum Usage Guide

## Quick Start

The adaptive curriculum is **enabled by default**. Just run training normally:

```bash
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java \
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" \
  "-Dexec.args=train"
```

## What You'll See

### Startup
```
[INFO] Opponent: WEAK-ComputerPlayer (bootstrap phase, 15/100 games)
```

### First Upgrade (~40% winrate)
```
[INFO] Episode 523 summary: turns=14, reward=1.050, opponent=CP_WEAK, winrate=0.398 (100 games)
[INFO] Opponent upgraded to: MEDIUM-CP7(skill=1) (winrate=0.402 over 100 games, episode=524)
```

### Second Upgrade (~55% winrate)
```
[INFO] Episode 1847 summary: turns=16, reward=-1.050, opponent=CP7, winrate=0.548 (100 games)
[INFO] Opponent upgraded to: STRONG-CP7(skill=3) (winrate=0.557 over 100 games, episode=1848)
```

### Self-Play Transition (~65% winrate)
```
[INFO] Episode 3291 summary: turns=22, reward=1.050, opponent=CP7, winrate=0.643 (100 games)
[INFO] Opponent upgraded to: SELFPLAY (winrate=0.652 over 100 games, episode=3292)
```

## Configuration Options

### Adjust Transition Thresholds

Make curriculum more/less aggressive:

```bash
# Make it easier (upgrade sooner)
export THRESHOLD_WEAK_MEDIUM=0.35      # Default: 0.40
export THRESHOLD_MEDIUM_STRONG=0.50    # Default: 0.55
export THRESHOLD_STRONG_SELFPLAY=0.60  # Default: 0.65

# Make it harder (upgrade later)
export THRESHOLD_WEAK_MEDIUM=0.50
export THRESHOLD_MEDIUM_STRONG=0.60
export THRESHOLD_STRONG_SELFPLAY=0.75
```

### Adjust Winrate Window

Change how many recent games are used to calculate winrate:

```bash
# Faster adaptation (more volatile)
export WINRATE_WINDOW=50

# Slower adaptation (more stable)
export WINRATE_WINDOW=200

# Default
export WINRATE_WINDOW=100
```

### Evaluation Opponents

Control eval/benchmark difficulty:

```bash
# Easier evaluation (faster, weaker)
export EVAL_OPPONENT_SKILL=3
export BENCHMARK_OPPONENT_SKILL=3

# Harder evaluation (slower, stronger)
export EVAL_OPPONENT_SKILL=8
export BENCHMARK_OPPONENT_SKILL=8

# Default (balanced)
export EVAL_OPPONENT_SKILL=6
export BENCHMARK_OPPONENT_SKILL=6
```

### Disable Adaptive Curriculum

Use fixed episode-based schedule instead:

```bash
export ADAPTIVE_CURRICULUM=0
export FIXED_WEAK_UNTIL=500
export FIXED_MEDIUM_UNTIL=3000
export FIXED_STRONG_UNTIL=8000
```

## PowerShell Examples

```powershell
# Quick training with adaptive curriculum (default)
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"

# Aggressive curriculum (upgrades faster)
$env:THRESHOLD_WEAK_MEDIUM="0.35"
$env:THRESHOLD_MEDIUM_STRONG="0.50"
$env:THRESHOLD_STRONG_SELFPLAY="0.60"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"

# Evaluation with tough opponent
$env:EVAL_OPPONENT_SKILL="8"
$env:EVAL_EPISODES="100"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=eval"
```

## Monitoring Tips

### Watch Training Progress
```bash
tail -f Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/models/training_stats.csv
```

### Analyze Winrate Progression
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_stats.csv')
df['winrate_ma'] = df['winrate'].rolling(window=50).mean()

plt.figure(figsize=(12, 6))
plt.plot(df['episode'], df['winrate_ma'], label='Winrate (50-ep MA)')
plt.axhline(y=0.40, color='g', linestyle='--', label='Weak→Medium')
plt.axhline(y=0.55, color='orange', linestyle='--', label='Medium→Strong')
plt.axhline(y=0.65, color='r', linestyle='--', label='Strong→SelfPlay')
plt.xlabel('Episode')
plt.ylabel('Winrate')
plt.title('Training Progression with Curriculum')
plt.legend()
plt.show()
```

### Track Opponent Changes
```bash
grep "Opponent upgraded" logs.txt
```

Output:
```
Opponent upgraded to: MEDIUM-CP7(skill=1) (winrate=0.402 over 100 games, episode=524)
Opponent upgraded to: STRONG-CP7(skill=3) (winrate=0.557 over 100 games, episode=1848)
Opponent upgraded to: SELFPLAY (winrate=0.652 over 100 games, episode=3292)
```

## Troubleshooting

### Stuck at Low Winrate
If winrate stays below threshold for too long:
- Check model is training (losses decreasing)
- Verify deck quality (balanced mana, playable cards)
- Consider lowering thresholds temporarily
- Check for game engine errors/warnings

### Upgrades Too Fast/Slow
Adjust window size:
- Too fast → Increase `WINRATE_WINDOW` (e.g., 200)
- Too slow → Decrease `WINRATE_WINDOW` (e.g., 50)

### Training Too Slow
Check which opponent is active:
- `CP_WEAK` → Fast (~1.2 eps/s)
- `CP7 (skill=1)` → Medium (~0.8 eps/s)
- `CP7 (skill=3)` → Slower (~0.5 eps/s)
- `SELFPLAY` → Fast (~0.7 eps/s)

If stuck on slow opponent, lower upgrade threshold.

## Best Practices

1. **Start with defaults** - They're tuned for good balance
2. **Monitor first 500 episodes** - Should upgrade to medium by ~500-1000
3. **Self-play by 3000-5000** - If not, thresholds might be too high
4. **Use skill=6 for eval** - Provides good benchmark without being too slow
5. **Track winrate trends** - Plateau indicates curriculum working correctly
