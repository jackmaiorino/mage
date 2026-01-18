# Hysteresis in Adaptive Curriculum - Explained

## The Problem: Oscillation Without Hysteresis

### What Was Happening (Before Hysteresis)

Your agent was bouncing between opponent types every few episodes:

```
Episode 948: WR=39% → WeakBot
Episode 949: WIN → WR=40% → MediumBot (upgraded)
Episode 950: LOSS → WR=39% → WeakBot (downgraded)
Episode 951: WIN → WR=40% → MediumBot (upgraded)
Episode 952: LOSS → WR=39% → WeakBot (downgraded)
...continues forever...
```

**Why this is bad:**
- Agent never adapts to either opponent
- Wasted training signal (inconsistent difficulty)
- Noisy gradients confuse the model
- Slower learning overall

## The Solution: Hysteresis

### What is Hysteresis?

**Hysteresis** = Different thresholds for going up vs going down

Think of it like a thermostat:
- Heat turns ON at 68°F
- Heat turns OFF at 72°F
- **4° gap** prevents constantly switching on/off

### How It Works in Training

```
Current Level: WEAK
├─ Upgrade to MEDIUM at 40% winrate
└─ Already at WEAK, can't downgrade further

Current Level: MEDIUM
├─ Upgrade to STRONG at 55% winrate
└─ Downgrade to WEAK at 35% winrate (5% gap)

Current Level: STRONG  
├─ Upgrade to SELFPLAY at 65% winrate
└─ Downgrade to MEDIUM at 50% winrate (5% gap)

Current Level: SELFPLAY
├─ Already at max level
└─ Downgrade to STRONG at 60% winrate (5% gap)
```

## Visual Example

### Without Hysteresis (Oscillating)

```
Winrate:  38% | 40% | 39% | 40% | 38% | 40% | 39% | 41% | 39% | 40%
          ───────────────────────────────────────────────────────────
Opponent: WEAK| MED | WEAK| MED | WEAK| MED | WEAK| MED | WEAK| MED
          
Result: 10 episodes, 8 opponent switches, no stable learning!
```

### With Hysteresis (Stable)

```
Winrate:  38% | 40% | 39% | 38% | 37% | 41% | 42% | 43% | 56% | 57%
          ───────────────────────────────────────────────────────────
Opponent: WEAK| MEDIUM ---------------------→ MEDIUM| STRONG------→
          
Upgrade:      ↑ (40%)                                ↑ (55%)
Downgrade:    Would need <35% to go back to WEAK    Would need <50% to go back

Result: 10 episodes, 2 opponent switches, stable learning at each level!
```

## Default Hysteresis Gaps

| Level | Upgrade At | Downgrade At | Gap | Stability Zone |
|-------|-----------|--------------|-----|----------------|
| **WEAK → MEDIUM** | 40% | 35% | 5% | 35-40% stays at current |
| **MEDIUM → STRONG** | 55% | 50% | 5% | 50-55% stays at current |
| **STRONG → SELFPLAY** | 65% | 60% | 5% | 60-65% stays at current |

## Additional Protection: Minimum Games Per Difficulty

### The Contamination Problem

Even with hysteresis, there's another issue: **winrate contamination from previous difficulty levels**.

**Example without minimum games requirement:**

```
Episode 1-100: Play WEAK opponent
  ├─ Win all 100 games
  └─ Winrate = 100%

Episode 101: Play MEDIUM opponent
  ├─ Lose the game
  ├─ Rolling winrate = 99/100 = 99%
  └─ System might immediately promote again!
```

**The rolling window mixes games from different difficulties!**

### The Solution: Minimum Games Requirement

The system now requires **at least 100 games** (configurable via `MIN_GAMES_PER_DIFFICULTY`) at each difficulty before allowing any level change.

```
Episode 1-100: Play WEAK opponent → 100% winrate
  └─ Upgrade to MEDIUM (meets threshold)

Episode 101: Switch to MEDIUM, reset winrate tracking
  ├─ Winrate window cleared
  ├─ Games counter reset to 0
  └─ Must play 100 games vs MEDIUM before next transition

Episodes 101-200: Play MEDIUM opponent
  ├─ Can't upgrade/downgrade until 100 games complete
  ├─ Winrate only includes MEDIUM games (clean data)
  └─ After 100 games: check thresholds with clean winrate
```

**Key benefits:**
- Clean winrate data at each difficulty level
- No contamination from previous opponents
- More reliable curriculum progression decisions
- Better statistical confidence (100 games = reasonable sample size)

### Configuration

```powershell
# Default: require 100 games before allowing transitions
$env:MIN_GAMES_PER_DIFFICULTY="100"

# Faster progression (less stable, smaller sample)
$env:MIN_GAMES_PER_DIFFICULTY="50"

# More conservative (more stable, larger sample)
$env:MIN_GAMES_PER_DIFFICULTY="200"
```

## Real Training Example

### Episode 900-1000 (Without Hysteresis - Bad)

```
Episode 900: WR=39%, Opp=WEAK
Episode 901: WR=40%, Opp=MEDIUM (upgrade)
Episode 902: WR=39%, Opp=WEAK (downgrade)
Episode 905: WR=40%, Opp=MEDIUM (upgrade)
Episode 908: WR=39%, Opp=WEAK (downgrade)
...
Episode 1000: Still bouncing, only 40% winrate
```

**Result:** 100 episodes wasted on oscillation

### Episode 900-1000 (With Hysteresis - Good)

```
Episode 900: WR=39%, Opp=WEAK
Episode 920: WR=40%, Opp=MEDIUM (upgrade at 40%)
Episode 921: WR=39%, Opp=MEDIUM (stays - need <35% to downgrade)
Episode 930: WR=38%, Opp=MEDIUM (stays)
Episode 950: WR=42%, Opp=MEDIUM (improving)
Episode 980: WR=46%, Opp=MEDIUM (learning)
Episode 1000: WR=52%, Opp=MEDIUM (almost ready for STRONG)
```

**Result:** 80 episodes of stable learning, winrate climbing steadily

## Configuration Examples

### Default (Recommended)

5% hysteresis gaps - good balance:

```powershell
# These are defaults, no need to set them
$env:THRESHOLD_WEAK_MEDIUM="0.40"
$env:THRESHOLD_MEDIUM_WEAK="0.35"       # 5% gap
$env:THRESHOLD_MEDIUM_STRONG="0.55"
$env:THRESHOLD_STRONG_MEDIUM="0.50"     # 5% gap
$env:THRESHOLD_STRONG_SELFPLAY="0.65"
$env:THRESHOLD_SELFPLAY_STRONG="0.60"   # 5% gap
```

### Wide Hysteresis (Very Stable)

10% gaps - agent stays at each level much longer:

```powershell
$env:THRESHOLD_WEAK_MEDIUM="0.40"
$env:THRESHOLD_MEDIUM_WEAK="0.30"       # 10% gap
$env:THRESHOLD_MEDIUM_STRONG="0.55"
$env:THRESHOLD_STRONG_MEDIUM="0.45"     # 10% gap
$env:THRESHOLD_STRONG_SELFPLAY="0.65"
$env:THRESHOLD_SELFPLAY_STRONG="0.55"   # 10% gap
```

**Use when:**
- High variance in game outcomes
- Decks have wide power level differences
- Want very stable training blocks

### Narrow Hysteresis (Responsive)

2% gaps - more reactive to performance changes:

```powershell
$env:THRESHOLD_WEAK_MEDIUM="0.40"
$env:THRESHOLD_MEDIUM_WEAK="0.38"       # 2% gap
$env:THRESHOLD_MEDIUM_STRONG="0.55"
$env:THRESHOLD_STRONG_MEDIUM="0.53"     # 2% gap
$env:THRESHOLD_STRONG_SELFPLAY="0.65"
$env:THRESHOLD_SELFPLAY_STRONG="0.63"   # 2% gap
```

**Use when:**
- Low variance in outcomes
- Consistent deck matchups
- Want faster adaptation to performance drops

### No Hysteresis (Not Recommended)

Same thresholds - causes oscillation:

```powershell
$env:THRESHOLD_WEAK_MEDIUM="0.40"
$env:THRESHOLD_MEDIUM_WEAK="0.40"       # No gap - will oscillate!
$env:THRESHOLD_MEDIUM_STRONG="0.55"
$env:THRESHOLD_STRONG_MEDIUM="0.55"     # No gap - will oscillate!
```

## Monitoring Hysteresis

### Good Training Pattern (Stable)

```
Episode 500: opponent=CP_WEAK, winrate=0.380
Episode 600: opponent=CP_WEAK, winrate=0.400
Opponent upgraded to: MEDIUM-CP7 (winrate=0.400, episode=601)
Episode 650: opponent=CP7, winrate=0.390 (stays - above 0.35 threshold)
Episode 700: opponent=CP7, winrate=0.420
Episode 800: opponent=CP7, winrate=0.480
Episode 900: opponent=CP7, winrate=0.520
Episode 950: opponent=CP7, winrate=0.550
Opponent upgraded to: STRONG-CP7 (winrate=0.550, episode=951)
```

**Indicators of good hysteresis:**
- 50-200+ episodes between transitions
- Winrate trends upward within each level
- Rare downgrades (only on real performance issues)

### Bad Training Pattern (Oscillating)

```
Episode 900: opponent=CP_WEAK, winrate=0.390
Episode 901: opponent=CP7, winrate=0.400
Opponent upgraded to: MEDIUM-CP7 (winrate=0.400, episode=901)
Episode 902: opponent=CP_WEAK, winrate=0.390
Opponent downgraded to: WEAK-ComputerPlayer (winrate=0.390, episode=902)
Episode 903: opponent=CP7, winrate=0.400
Opponent upgraded to: MEDIUM-CP7 (winrate=0.400, episode=903)
```

**Indicators of insufficient hysteresis:**
- Opponent switches every 1-5 episodes
- Winrate hovering at threshold boundary
- Frequent "upgraded" immediately followed by "downgraded"

**Fix:** Increase hysteresis gap from 5% to 8-10%

## Hysteresis Gap Recommendations

| Scenario | Gap Size | Rationale |
|----------|----------|-----------|
| **Default training** | 5% | Good balance for most cases |
| **High variance decks** | 8-10% | Deck power differences cause swings |
| **Similar decks** | 3-5% | Consistent matchups = less variance |
| **Early training (<1000 eps)** | 5-8% | Model unstable, needs stability |
| **Late training (>5000 eps)** | 3-5% | Model stable, can be responsive |

## Expected Behavior After Fix

### Training Log with Hysteresis

```
Episode 900: opponent=CP_WEAK, winrate=0.380
Episode 950: opponent=CP_WEAK, winrate=0.390
Episode 1000: opponent=CP_WEAK, winrate=0.420
Opponent upgraded to: MEDIUM-CP7(skill=1) (winrate=0.420, episode=1001)
Episode 1050: opponent=CP7, winrate=0.380 (stays, above 0.35 downgrade threshold)
Episode 1100: opponent=CP7, winrate=0.400
Episode 1200: opponent=CP7, winrate=0.480
Episode 1300: opponent=CP7, winrate=0.520
Episode 1400: opponent=CP7, winrate=0.560
Opponent upgraded to: STRONG-CP7(skill=3) (winrate=0.560, episode=1401)
Episode 1500: opponent=CP7, winrate=0.510 (stays, above 0.50 downgrade threshold)
Episode 1800: opponent=CP7, winrate=0.620
Episode 2000: opponent=CP7, winrate=0.660
Opponent upgraded to: SELFPLAY (winrate=0.660, episode=2001)
```

**Notice:**
- Long stable blocks at each level (100-500 episodes)
- Only 3 transitions in 1100 episodes (was ~200 transitions before!)
- Winrate steadily improves within each block
- Downgrades only occur on significant performance drops

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
10. **Check hysteresis working:** Count transitions - should see <10 transitions per 1000 episodes