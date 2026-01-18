# Training Commands Reference

Quick reference for common training, evaluation, and build commands with the new adaptive curriculum system.

## Table of Contents
- [Training Commands](#training-commands)
- [Evaluation Commands](#evaluation-commands)
- [Build Commands](#build-commands)
- [Environment Variables Reference](#environment-variables-reference)

---

## Training Commands

$env:RESET_EPISODE_COUNTER="0"  # Don't reset!
$env:TOTAL_EPISODES="10000"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"

### 1. Basic Training (Adaptive Curriculum - Default)

```powershell
# Simple training with adaptive curriculum (most common)
$env:MODE="train"
$env:TOTAL_EPISODES="2000"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

**What it does:**
- Starts with WeakBot opponents (~1.2 eps/s)
- Upgrades to Medium at 40% winrate
- Upgrades to Strong at 55% winrate
- Switches to Self-Play at 65% winrate
- Uses 100-game rolling window for winrate

### 2. Full Training Run (16 Workers)

```powershell
# Full training with multiple workers and monitoring
$env:MODE="train"
$env:TOTAL_EPISODES="2000"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
$env:RESET_EPISODE_COUNTER="1"
$env:NUM_GAME_RUNNERS="16"
$env:PY_BATCH_MAX_SIZE="768"
$env:PY_BATCH_TIMEOUT_MS="50"
$env:METRICS_PORT="9091"
$env:RL_ACTIVATION_DIAG="1"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

**Or using the PowerShell script:**

```powershell
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
$env:RESET_EPISODE_COUNTER="1"
$env:PY_BATCH_MAX_SIZE="768"
$env:PY_BATCH_TIMEOUT_MS="50"
$env:METRICS_PORT="9091"
$env:RL_ACTIVATION_DIAG="1"

powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\rl-train.ps1" `
  -TotalEpisodes 2000 `
  -NumGameRunners 16 `
  -MetricsPort 9091
```

### 3. Aggressive Curriculum (Upgrades Faster)

```powershell
# Upgrades opponents at lower winrate thresholds
$env:MODE="train"
$env:TOTAL_EPISODES="2000"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
$env:THRESHOLD_WEAK_MEDIUM="0.35"      # Upgrade at 35% (default: 40%)
$env:THRESHOLD_MEDIUM_STRONG="0.50"    # Upgrade at 50% (default: 55%)
$env:THRESHOLD_STRONG_SELFPLAY="0.60"  # Self-play at 60% (default: 65%)

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

### 4. Conservative Curriculum (Stays Longer on Each Level)

```powershell
# More gradual progression, higher winrate requirements
$env:MODE="train"
$env:TOTAL_EPISODES="2000"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
$env:THRESHOLD_WEAK_MEDIUM="0.50"      # Upgrade at 50% (default: 40%)
$env:THRESHOLD_MEDIUM_STRONG="0.60"    # Upgrade at 60% (default: 55%)
$env:THRESHOLD_STRONG_SELFPLAY="0.75"  # Self-play at 75% (default: 65%)
$env:WINRATE_WINDOW="200"              # Larger window = more stable (default: 100)

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

### 4b. Custom Hysteresis Gaps

```powershell
# Adjust hysteresis to control when opponents downgrade
$env:MODE="train"
$env:TOTAL_EPISODES="2000"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"

# Wide hysteresis (10% gap) - very stable, rarely downgrades
$env:THRESHOLD_WEAK_MEDIUM="0.40"      # Upgrade at 40%
$env:THRESHOLD_MEDIUM_WEAK="0.30"      # Downgrade at 30% (10% gap)
$env:THRESHOLD_MEDIUM_STRONG="0.55"    # Upgrade at 55%
$env:THRESHOLD_STRONG_MEDIUM="0.45"    # Downgrade at 45% (10% gap)

# Narrow hysteresis (2% gap) - more responsive
$env:THRESHOLD_WEAK_MEDIUM="0.40"      # Upgrade at 40%
$env:THRESHOLD_MEDIUM_WEAK="0.38"      # Downgrade at 38% (2% gap)
$env:THRESHOLD_MEDIUM_STRONG="0.55"    # Upgrade at 55%
$env:THRESHOLD_STRONG_MEDIUM="0.53"    # Downgrade at 53% (2% gap)

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

### 5. Legacy Fixed Schedule (Comparison/Testing)

```powershell
# Use old episode-based curriculum for comparison
$env:MODE="train"
$env:TOTAL_EPISODES="2000"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
$env:ADAPTIVE_CURRICULUM="0"           # Disable adaptive
$env:FIXED_WEAK_UNTIL="500"
$env:FIXED_MEDIUM_UNTIL="3000"
$env:FIXED_STRONG_UNTIL="8000"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

### 6. Continue Training (Don't Reset Counter)

```powershell
# Continue from existing episode count
$env:MODE="train"
$env:TOTAL_EPISODES="5000"             # New target
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
$env:RESET_EPISODE_COUNTER="0"         # Don't reset (or omit this line)

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

---

## Evaluation Commands

### 1. Basic Evaluation (Default: skill=6)

```powershell
# Evaluate against strong opponent (skill=6)
$env:MODE="eval"
$env:EVAL_EPISODES="100"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=eval"
```

### 2. Quick Evaluation (Fewer Games)

```powershell
# Fast evaluation with 10 games
$env:MODE="eval"
$env:EVAL_EPISODES="10"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=eval"
```

### 3. Evaluation vs Medium Opponent

```powershell
# Test against weaker opponent (skill=3)
$env:MODE="eval"
$env:EVAL_EPISODES="100"
$env:EVAL_OPPONENT_SKILL="3"           # Weaker than default (6)
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=eval"
```

### 4. Evaluation vs Very Strong Opponent

```powershell
# Test against strongest opponent (skill=8+)
$env:MODE="eval"
$env:EVAL_EPISODES="100"
$env:EVAL_OPPONENT_SKILL="8"           # Much stronger
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=eval"
```

### 5. Comprehensive Evaluation (Multiple Workers)

```powershell
# Faster evaluation with parallel games
$env:MODE="eval"
$env:EVAL_EPISODES="200"               # Total episodes
$env:NUM_GAME_RUNNERS="8"              # 8 workers = 25 games each
$env:EVAL_OPPONENT_SKILL="6"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"

mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=eval"
```

---

## Build Commands

### 1. Clean Compile

```powershell
# Full rebuild
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests clean compile
```

### 2. Quick Compile (No Clean)

```powershell
# Incremental compile (faster)
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

### 3. Compile with Tests

```powershell
# Compile and run tests
mvn -pl Mage.Server.Plugins/Mage.Player.AIRL -am compile test
```

### 4. Full Build (All Modules)

```powershell
# Build entire project
mvn clean install -DskipTests
```

---

## Environment Variables Reference

### Adaptive Curriculum

| Variable | Default | Description |
|----------|---------|-------------|
| `ADAPTIVE_CURRICULUM` | 1 | Enable (1) or disable (0) adaptive curriculum |
| `WINRATE_WINDOW` | 100 | Rolling window size for winrate calculation |
| `MIN_GAMES_PER_DIFFICULTY` | 100 | **Minimum games at each difficulty before allowing transitions** |
| **Upgrade Thresholds** | | |
| `THRESHOLD_WEAK_MEDIUM` | 0.40 | Winrate to upgrade weak → medium |
| `THRESHOLD_MEDIUM_STRONG` | 0.55 | Winrate to upgrade medium → strong |
| `THRESHOLD_STRONG_SELFPLAY` | 0.65 | Winrate to upgrade strong → self-play |
| **Downgrade Thresholds (Hysteresis)** | | |
| `THRESHOLD_MEDIUM_WEAK` | 0.35 | Winrate to downgrade medium → weak (5% gap) |
| `THRESHOLD_STRONG_MEDIUM` | 0.50 | Winrate to downgrade strong → medium (5% gap) |
| `THRESHOLD_SELFPLAY_STRONG` | 0.60 | Winrate to downgrade self-play → strong (5% gap) |

### Fixed Curriculum (Legacy)

| Variable | Default | Description |
|----------|---------|-------------|
| `FIXED_WEAK_UNTIL` | 500 | Episodes with weak opponent |
| `FIXED_MEDIUM_UNTIL` | 3000 | Episodes with medium opponent |
| `FIXED_STRONG_UNTIL` | 8000 | Episodes with strong opponent |

### Training Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | train | Mode: train, eval, benchmark, learner, worker |
| `TOTAL_EPISODES` | 10000 | Total training episodes |
| `NUM_GAME_RUNNERS` | CPU-1 | Parallel game workers |
| `RESET_EPISODE_COUNTER` | 0 | Reset episode counter to 0 |
| `DECK_LIST_FILE` | "" | Path to deck list file |
| `DECKS_DIR` | ... | Directory containing deck files |

### Evaluation Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_EPISODES` | 5 | Number of evaluation games per worker |
| `EVAL_OPPONENT_SKILL` | 6 | Opponent skill for evaluation (1-10) |
| `BENCHMARK_OPPONENT_SKILL` | 6 | Opponent skill for benchmark runs |

### Model & Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | models/model.pt | Path to model file |
| `PY_BATCH_MAX_SIZE` | 512 | Max batch size for Python bridge |
| `PY_BATCH_TIMEOUT_MS` | 100 | Batch timeout in milliseconds |

### Logging & Monitoring

| Variable | Default | Description |
|----------|---------|-------------|
| `RL_ACTIVATION_DIAG` | 0 | Enable detailed activation diagnostics |
| `RL_VERBOSE_DECISIONS` | 0 | Enable verbose decision logging |
| `TRAIN_DIAG` | 0 | Enable training diagnostics |
| `TRAIN_DIAG_EVERY` | 50 | Diagnostic frequency (episodes) |
| `TRAIN_LOG_EVERY` | 10 | Progress log frequency (episodes) |
| `TRAIN_HEARTBEAT_SEC` | 30 | Heartbeat interval (seconds) |
| `METRICS_PORT` | 9090 | Prometheus metrics port |

---

## Quick Reference Card

**Start fresh training:**
```powershell
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
$env:RESET_EPISODE_COUNTER="1"
$env:TOTAL_EPISODES="2000"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train"
```

**Quick eval:**
```powershell
$env:EVAL_EPISODES="10"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=eval"
```

**Compile:**
```powershell
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests clean compile
```

---

## Understanding Hysteresis

### What is Hysteresis?

Hysteresis means **different thresholds for upgrading vs downgrading** to prevent rapid oscillation:

```
Without Hysteresis (BAD):
38% → 40% → 39% → 40% → 39% → 40% ...
 Weak → Med → Weak → Med → Weak → Med  (bouncing every game!)

With Hysteresis (GOOD):
38% → 40% → 39% → 38% → 42% → 56% ...
 Weak → Med --------> Med → Med → Strong  (stable blocks)
```

### Default Hysteresis Gaps (5%)

| Transition | Upgrade At | Downgrade At | Gap |
|-----------|-----------|--------------|-----|
| Weak ↔ Medium | 40% | 35% | 5% |
| Medium ↔ Strong | 55% | 50% | 5% |
| Strong ↔ Self-Play | 65% | 60% | 5% |

### Minimum Games Protection

In addition to hysteresis, the system requires **at least 100 games** (configurable via `MIN_GAMES_PER_DIFFICULTY`) at each difficulty before allowing any level change.

**Why this matters:**
- Prevents winrate contamination from previous difficulty levels
- Example without this protection:
  - Beat WEAK 100 times (100% winrate)
  - Upgrade to MEDIUM
  - Lose 1 game → Rolling winrate = 99/100 = 99%
  - System might promote again based on contaminated data!
- With minimum games requirement:
  - Must play 100 games vs MEDIUM before considering next promotion
  - Winrate calculation is reset when difficulty changes
  - Clean data for curriculum decisions

### Why This Helps Training

1. **Reduces noise** - Consistent opponent over multiple episodes
2. **Better consolidation** - Agent fully adapts to each difficulty
3. **Cleaner gradients** - Model sees stable training signal
4. **Faster convergence** - Less time wasted on transitions
5. **More efficient** - Every episode contributes to mastery
6. **Prevents contamination** - Clean winrate data at each level

### Tuning Hysteresis

**Wide gaps (8-10%)** - Very stable, slower to downgrade:
```powershell
$env:THRESHOLD_WEAK_MEDIUM="0.40"
$env:THRESHOLD_MEDIUM_WEAK="0.30"      # 10% gap
```
- Use when: Training is noisy, lots of variance
- Effect: Agent stays at each level longer

**Narrow gaps (2-3%)** - More responsive:
```powershell
$env:THRESHOLD_WEAK_MEDIUM="0.40"
$env:THRESHOLD_MEDIUM_WEAK="0.38"      # 2% gap
```
- Use when: Want faster adaptation to performance changes
- Effect: Quicker downgrades if struggling

**No hysteresis (0% gap)** - Old behavior:
```powershell
$env:THRESHOLD_WEAK_MEDIUM="0.40"
$env:THRESHOLD_MEDIUM_WEAK="0.40"      # Same threshold
```
- Not recommended - causes oscillation

---

## Tips

1. **First run?** Use `RESET_EPISODE_COUNTER="1"` to start from episode 0
2. **Slow training?** Check which opponent is active in logs (WeakBot is fastest)
3. **Monitor progress:** Watch for "Opponent upgraded/downgraded to:" messages in logs
4. **CSV tracking:** Stats are saved to `models/training_stats.csv` with winrate column
5. **Adjust thresholds:** If agent gets stuck at low winrate, lower `THRESHOLD_*` values
6. **Multiple decks:** Use `DECK_LIST_FILE` to specify a subset, or `DECKS_DIR` for all decks in a folder
7. **Parallel training:** Increase `NUM_GAME_RUNNERS` for faster throughput (default: CPU cores - 1)
8. **Memory issues?** Reduce `PY_BATCH_MAX_SIZE` if OOM errors occur
9. **Oscillating opponents?** Increase hysteresis gaps (e.g., set `THRESHOLD_MEDIUM_WEAK="0.30"` for 10% gap)
10. **Faster curriculum progression?** Reduce `MIN_GAMES_PER_DIFFICULTY` to 50-75, but be aware of potential winrate contamination
11. **More stable curriculum?** Increase `MIN_GAMES_PER_DIFFICULTY` to 150-200 for better statistical confidence

---

## Example Training Session

```powershell
# 1. Clean compile
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests clean compile

# 2. Start fresh training (2000 episodes)
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
$env:RESET_EPISODE_COUNTER="1"
$env:TOTAL_EPISODES="2000"
$env:NUM_GAME_RUNNERS="8"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"

# 3. Evaluate after training
$env:MODE="eval"
$env:EVAL_EPISODES="100"
$env:EVAL_OPPONENT_SKILL="6"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=eval"

# 4. Continue training if needed
$env:MODE="train"
$env:RESET_EPISODE_COUNTER="0"
$env:TOTAL_EPISODES="5000"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```
