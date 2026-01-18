# Project Guidelines for AI Assistants

## Critical Rules

1. **DO NOT create or modify documentation files unless explicitly requested**
   - No README files
   - No markdown docs
   - No explanatory comments
   - User will ask if they want documentation

2. **Be minimally verbose**
   - Short, direct responses
   - No excessive explanation
   - Code changes speak for themselves
   - Don't over-explain every change

3. **Focus on implementation over discussion**
   - Make changes, don't just suggest them
   - Show code, not lengthy descriptions
   - User can read the diff

## Project Overview

**XMage RL Player** - Reinforcement learning AI for Magic: The Gathering (XMage fork)

### Key Components

- **ComputerPlayerRL.java** - RL agent that uses neural network for decision-making
- **RLTrainer.java** - Training loop with adaptive curriculum
- **StateSequenceBuilder.java** - Game state encoding
- **PythonMLBridge.java** - Java ↔ Python ML model bridge
- **Adaptive curriculum** - Opponent difficulty scales with agent winrate

### Architecture

```
Java (XMage game engine)
  ↓ game state
ComputerPlayerRL 
  ↓ encoded state + candidates
PythonMLBridge (Py4J)
  ↓ batch inference
Python (PyTorch model)
  ↓ policy + value
ComputerPlayerRL
  ↓ action selection
Java (execute action)
```

### Training Flow

1. Games run in parallel (16 workers default)
2. Agent collects state/action/reward tuples
3. Episode ends → calculate returns
4. Model updated via Python bridge
5. Opponent difficulty adapts based on rolling winrate

### Current Focus

- Improving training speed (curriculum learning implemented)
- Reducing activation failures (better action filtering)
- Handling complex card interactions
- Self-play training at high skill levels

## Common Tasks

### Fix a bug
- Read relevant files
- Make changes
- Verify compilation
- Done. No essay needed.

### Add a feature
- Implement it
- Test it works
- Move on

### Investigate an issue
- Check logs/code
- Explain findings concisely
- Propose fix if appropriate

### Refactor code
- Make changes
- Explain why in 1-2 sentences if non-obvious
- Don't write a dissertation

## What User Values

- **Efficiency** - Quick, accurate changes
- **Brevity** - Get to the point
- **Action** - Implement, don't just discuss
- **Honesty** - Disagree when wrong (per user rules)
- **No fluff** - No unnecessary documentation

## Code Style

- Java 8+ features OK
- Follow existing patterns in codebase
- Thread-safe for parallel execution
- Minimal comments (code should be clear)
- No over-engineering

## Tech Stack

- **Java 8+** - Main codebase (Maven build)
- **Python 3.8+** - ML model (PyTorch)
- **Py4J** - Java/Python bridge
- **XMage** - MTG game engine (forked)
- **Maven** - Build system

## File Locations

```
Mage.Server.Plugins/Mage.Player.AIRL/
├── src/mage/player/ai/
│   ├── ComputerPlayerRL.java       # Main RL agent
│   └── rl/
│       ├── RLTrainer.java          # Training loop
│       ├── StateSequenceBuilder.java  # State encoding
│       ├── PythonMLBridge.java     # Python bridge
│       └── MLPythonCode/           # Python model code
│           ├── model.py
│           ├── trainer.py
│           └── inference_server.py
└── docs/                            # Only modify if asked!
    ├── README.md
    ├── commands.md
    └── ...
```

## Build Commands

**Compile:**
```powershell
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile
```

**Train:**
```powershell
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train"
```

## Environment Variables

Key vars for training:
- `ADAPTIVE_CURRICULUM=1` - Enable adaptive opponents (default)
- `WINRATE_WINDOW=100` - Rolling window size
- `THRESHOLD_*` - Curriculum thresholds (see docs/commands.md)
- `NUM_GAME_RUNNERS=16` - Parallel workers
- `TOTAL_EPISODES=2000` - Training episodes

## Remember

**Less is more. User will ask if they want more.**
