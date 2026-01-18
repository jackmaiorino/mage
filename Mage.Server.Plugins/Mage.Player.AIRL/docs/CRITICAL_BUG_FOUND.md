# CRITICAL BUG: Agent Could Not Play Lands

## Discovery Date
2026-01-17

## Severity
**CATASTROPHIC** - Invalidates all training prior to this fix.

## Summary
The RL agent trained for 10,000 episodes (2,000 initially + 8,000 continuation) **without ever being able to play lands**. The agent's only available actions were:
- Pass
- Cast spells (if it happened to have mana from other sources)
- Activate abilities

This explains the 6.3% final winrate - the agent was playing without lands against opponents who could play lands.

## Root Cause

In `ComputerPlayerRL.calculateRLAction()` at line 1273-1276:

```java
.filter(ability -> {
    ActivatedAbility.ActivationStatus status = ability.canActivate(this.getId(), game);
    return status != null && status.canActivate() && !status.getApprovingObjects().isEmpty();
})
```

**The bug:** The filter requires `!status.getApprovingObjects().isEmpty()`, but `PlayLandAbility` doesn't populate approving objects (lands are played directly without approval). This caused all land plays to be filtered out.

## Evidence

From evaluation game logs (Episode after 10k training):
```
Turn 22, Main Phase 1
PlayerRL1 hand: [Mistvault Bridge; Mistvault Bridge; Galvanic Blast; Krark-Clan Shaman; 
                 Mistvault Bridge; Silverbluff Bridge; Galvanic Blast]
PlayerRL1 permanents: []
Playable options: [Pass]
```

The agent had 4 lands in hand, zero permanents on battlefield, and could only pass.

## The Fix

```java
.filter(ability -> {
    ActivatedAbility.ActivationStatus status = ability.canActivate(this.getId(), game);
    if (status == null || !status.canActivate()) {
        return false;
    }
    // PlayLandAbility doesn't use approving objects - allow it through
    if (ability instanceof PlayLandAbility) {
        return true;
    }
    // For other abilities, require at least one approving object
    return !status.getApprovingObjects().isEmpty();
})
```

## Impact on Training Results

### Before Fix (10,000 episodes)
- **6.3% winrate** against MEDIUM-CP7(skill=1)
- Agent could not play lands
- Winrate **declining** from 12.8% → 6.3% (episode 9944 → 10000)
- Both poor mulligan logic AND land-play bug active

### After Fix (Expected)
Should see dramatic improvement as agent can now:
1. Play lands every turn
2. Cast spells with mana from those lands
3. Build board presence

## Action Required

**START FRESH TRAINING** - All previous training data is invalid.

```powershell
# Reset episode counter and start new training
$env:RESET_EPISODE_COUNTER="1"
$env:TOTAL_EPISODES="5000"
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" `
  "-Dexec.args=train"
```

## Related Issues

1. **Mulligan bug** (also fixed): Base `ComputerPlayer` stops mulliganing after hand size < 6, leading to 0-land keeps
2. **Perfect information**: CP7 opponent sees RL agent's hand during simulations (omniscient opponent)

## Lessons Learned

1. Always validate that AI can perform **basic game actions** before long training runs
2. Print detailed game state during eval to catch obvious issues
3. Unit test that getPlayable() returns lands when expected
