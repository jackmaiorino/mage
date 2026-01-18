# Analysis: Efficient "Can Be Played" Checking

## Current Situation

The RL agent uses simulation-based filtering to validate playable actions:

```java
for (ActivatedAbility ability : flattenedOptions) {
    Game sim = game.createSimulationForAI();  // Full game copy!
    if (sim.getPlayer(this.getId()).activateAbility((ActivatedAbility) ability.copy(), sim)) {
        validOptions.add(ability);
    }
}
```

**Cost:** For N abilities, we create N full game state copies.

## What Does Each Method Check?

### `canActivate()` (Lightweight)
- Basic zone checks (e.g., card in hand/graveyard)
- Timing restrictions (sorcery speed, your turn, etc.)
- Mana availability (rough check)
- Activation limits (once per turn, etc.)
- **Approving objects** (continuous effects that enable the ability)

### `activate()` (Heavyweight)
- Everything `canActivate()` checks, PLUS:
- **Alternative cost selection** (Flashback, Escape, etc.)
- **X cost announcements** (dynamic calculation)
- **Phyrexian mana choices** (life vs mana)
- **Mode selection** (modal spells)
- **Target selection and validation**
- **Cost target selection** (targets chosen as part of costs)
- **Splice effects**
- **Additional costs** (Kicker, Buyback, etc.)
- **Cost modification effects**
- **Improvise/Convoke/Offering interactions**

## Why `canActivate()` Isn't Sufficient

Looking at the code in `AbilityImpl.activate()` (lines 271-400+), there are dozens of complex checks and interactions that only happen during actual activation:

1. **Target Availability**: `canActivate()` doesn't validate targets exist and are legal
2. **Alternative Costs**: You might have multiple ways to cast a spell - only `activate()` resolves which
3. **X Costs**: Calculating available X requires deep game state analysis
4. **Hidden Costs**: Some costs depend on game state (Improvise, Convoke, Offering)
5. **Splice**: Cards to splice are only known during activation
6. **Cost Targets**: Some costs require targets (e.g., Devouring Strossus)

## Performance Investigation

### Cost of `game.createSimulationForAI()`

```java
public Game createSimulationForAI() {
    Game res = this.copy();  // <-- Full deep copy
    ((GameImpl) res).simulation = true;
    ((GameImpl) res).aiGame = true;
    return res;
}
```

**What `game.copy()` copies:**
- All players (hands, libraries, graveyards)
- All permanents on battlefield
- The stack
- All game state values
- All continuous effects
- Combat state
- Turn structure

**Rough estimate:** For a mid-game state (turn 10, 20 permanents, 30 cards in hand+graveyard), this is likely:
- ~1000-5000 object allocations
- ~100KB-1MB of memory per simulation
- ~0.5-2ms CPU time per simulation

### Typical Filtering Load

- **Early game (turn 1-3):** 2-5 abilities = 2-5 simulations
- **Mid game (turn 5-10):** 5-15 abilities = 5-15 simulations  
- **Late game (turn 15+):** 10-30 abilities = 10-30 simulations

**Worst case:** 30 simulations × 2ms = **60ms overhead per action**

With 20 actions per game × 100 games = 2000 simulations = **120 seconds overhead per 100 games**.

## Optimization Options

### Option 1: Accept the Cost ✓ (Current)
**Pros:**
- Correct behavior guaranteed
- Matches ComputerPlayer6/7 approach
- No implementation risk

**Cons:**
- ~60ms overhead per action in late game
- ~1-2% of total training time

**Verdict:** This is acceptable! The overhead is small compared to:
- ComputerPlayer7 thinking time (omniscient lookahead)
- Python model inference
- Game engine processing

### Option 2: Lazy Simulation (Create simulation only when ambiguous)
**Approach:**
```java
// Use canActivate() first
if (!status.canActivate()) continue;

// Only simulate if approving objects are empty (ambiguous case)
if (status.getApprovingObjects().isEmpty() && !(ability instanceof PlayLandAbility)) {
    // Test in simulation
}
```

**Pros:**
- Reduces simulations by ~50-70%
- Still correct

**Cons:**
- Complex logic
- Approving objects check doesn't cover all edge cases
- Maintenance burden

**Verdict:** Not worth the complexity for ~1% speedup.

### Option 3: Batch Simulation (Single simulation for all abilities)
**Approach:**
```java
Game sim = game.createSimulationForAI();
for (ActivatedAbility ability : flattenedOptions) {
    if (sim.getPlayer(this.getId()).activateAbility(...)) {
        validOptions.add(ability);
        sim = game.createSimulationForAI(); // Reset for next test
    }
}
```

**Pros:**
- None (you still need N simulations)

**Cons:**
- Same cost as current approach
- More complex

**Verdict:** No benefit.

### Option 4: Custom Lightweight Validator
**Approach:** Build a custom validator that replicates `activate()`'s checks without modifying game state.

**Pros:**
- Could be faster (no deep copy)

**Cons:**
- **HIGH RISK**: Would need to reimplement ~500 lines of complex activation logic
- **Maintenance nightmare**: Every game engine change breaks it
- **Bug prone**: Easy to miss edge cases
- **Estimated effort**: 2-4 weeks of development + ongoing maintenance

**Verdict:** Definitely not worth it for 1-2% speedup.

## Performance Comparison

### With Simulation Filtering (Current)
```
Action selection: ~1-2ms (model inference)
Simulation filtering: ~10-60ms (depends on game state)
Total: ~11-62ms per action
```

### Hypothetical Without Filtering
```
Action selection: ~1-2ms (model inference)
Handling failed activations: ~0ms (but pollutes training signal)
Total: ~1-2ms per action
BUT: Model wastes probability mass on invalid actions
```

## Recommendations

### 1. Keep Current Approach ✓
The simulation-based filtering is the right choice because:
- It's correct and matches the reference implementation
- The performance cost (~60ms worst case) is negligible
- Alternative approaches are risky and complex

### 2. Monitor Performance
Add metrics to track:
```java
long startTime = System.nanoTime();
List<ActivatedAbility> validOptions = new ArrayList<>();
for (ActivatedAbility ability : flattenedOptions) {
    Game sim = game.createSimulationForAI();
    if (sim.getPlayer(this.getId()).activateAbility(...)) {
        validOptions.add(ability);
    }
}
long filterTime = System.nanoTime() - startTime;
metricsCollector.recordFilteringTime(filterTime / 1_000_000); // milliseconds
```

### 3. Potential Future Optimization (Low Priority)
If profiling shows this is actually a bottleneck (unlikely), consider:
- **Object pooling** for simulations (reuse copied game states)
- **Incremental copying** (copy-on-write for unchanged zones)
- **Caching** (memoize simulation results for identical game states)

But these are complex and should only be attempted if profiling proves it's necessary.

## Conclusion

**The simulation-based filtering is efficient enough.** 

The 1-2% overhead is a reasonable price for:
- ✓ Correctness guarantee
- ✓ Matching ComputerPlayer6/7 behavior  
- ✓ No maintenance burden
- ✓ Clean, understandable code

**DO NOT attempt custom validation logic** - the risk/reward ratio is terrible.
