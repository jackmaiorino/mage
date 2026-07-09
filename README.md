# XMage RL Player

A reinforcement-learning agent that teaches itself to play Magic: The Gathering.

This is a fork of [XMage](https://github.com/magefree/mage), the open-source MTG engine. Everything RL lives in a new Maven module, [`Mage.Server.Plugins/Mage.Player.AIRL`](Mage.Server.Plugins/Mage.Player.AIRL), plus supporting training infrastructure (`scripts/`, `k8s/`, `monitoring/`). Changes outside that module are game-engine improvements needed to legally mask the action space for the model. For XMage itself (the game client/server), see the [upstream README](https://github.com/magefree/mage#readme).

## What it does

The agent plays full games of Magic through the real XMage engine. No simplified environment. A transformer policy/value network scores legal actions from an encoded game state, trained with PPO through self-play and an adaptive curriculum. Population-based training (PBT) scales it out, locally and on UMD's Zaratan HPC cluster.

**Current focus:** PBT across a pool of Pauper decks (Spy Combo, Wildfire, Rally, Affinity), with teacher-distillation experiments on top of the PPO base to push past plateaus ([experiment logs](docs/mtgrl)). The long-standing benchmark problem is the Spy Combo line: Balustrade Spy targets its own controller (the deck runs zero lands) to mill the entire library, then Flashback Dread Return reanimates Lotleth Giant for lethal damage. A long forced sequence with sparse reward — a hard credit-assignment problem.

## Architecture

```text
Game State -> StateSequenceBuilder (Java) -> Transformer -> Action Selection (Java)
```

- **State encoding:** the full game state (players, hands, battlefield, graveyards, stack) becomes a token sequence; each legal action becomes a candidate feature vector.
- **Model:** transformer encoder with a value head and per-decision-type candidate-scoring heads (action / target / card select). Mulligan decisions started as a separate Q-network and were later folded into the main model.
- **Inference vs. training:** inference runs in-process through Java ONNX Runtime (CUDA, FP16), which replaced GIL-bottlenecked Python inference for a 6.5x throughput gain. Training stays in Python: game runners stream episodes over a batched TCP protocol to a shared GPU service (`gpu_service_host.py`) running PPO updates. Pure-Python and Py4J modes remain for single-process runs.
- **Curriculum:** opponents scale automatically from a weak bot up through XMage's heuristic AI to self-play, driven by rolling winrate with hysteresis to prevent oscillation.
- **PBT:** multiple profiles train with different hyperparameters; the orchestrator periodically copies the winner's weights to the losers and perturbs hyperparameters by +/-20%.

Multi-node layout on HPC: one GPU head node runs the shared GPU service in split-device mode (cuda:0 inference, cuda:1 training) while CPU satellite nodes each run multiple JVMs of game runners connecting back over TCP. Measured throughput: 5.7 episodes/sec across 640 parallel game runners (10 satellite nodes).

Full details: [project timeline](docs/PROJECT_TIMELINE.md), [module docs](Mage.Server.Plugins/Mage.Player.AIRL/docs/) ([architecture](Mage.Server.Plugins/Mage.Player.AIRL/docs/architecture.md), [curriculum](Mage.Server.Plugins/Mage.Player.AIRL/docs/curriculum_usage.md), [commands](Mage.Server.Plugins/Mage.Player.AIRL/docs/commands.md), [model versions](Mage.Server.Plugins/Mage.Player.AIRL/docs/model_versions.md)), and [experiment logs](docs/mtgrl).

## Progress

| When | Milestone |
|------|-----------|
| Jun 2025 | v1: first end-to-end training runs; ~29% winrate vs XMage's heuristic AI |
| Jul 2025 | v1.2: heuristic targeting + retrain; ~44% winrate |
| Jan – Feb 2026 | v2: full refactor; hashed card-ID embeddings replace text embeddings; ~50% winrate after 50k episodes (1-2 days local training) |
| Feb 2026 | v3: RL takes over spell/ability targeting, the decisions the heuristics were getting wrong |
| Mar 2026 | Multi-node HPC pipeline on Zaratan (Slurm): GPU head + CPU satellites, 5.7 eps/sec measured; in-process Java ONNX inference (6.5x over Python) |
| Apr 2026 | Multi-deck Pauper PBT (Wildfire, Rally, Affinity, Elves alongside Spy); dual-sided training doubles effective throughput; mulligan model merged into main model |
| May – Jul 2026 | Teacher-distillation experiments (tree-search and heuristic-AI teachers), cloud GPU rental tooling |

Winrates are vs XMage's built-in heuristic AI (ComputerPlayer7).

## Running it

Local PBT (single GPU, ~3 eps/sec on an RTX 4070 Super):

```bash
py -3.12 scripts/run_local_pbt.py
```

Single profile, legacy path:

```powershell
$env:DECK_LIST_FILE="Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.spy_combo.txt"
$env:MODEL_PROFILE="spy"
mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile exec:java `
  "-Dexec.mainClass=mage.player.ai.rl.RLTrainer" "-Dexec.args=train"
```

HPC (Slurm): build a runtime bundle with `scripts/hpc/build_rl_runtime_bundle.ps1`, submit a GPU head via `sbatch scripts/hpc/gpu_head.sbatch`, then attach CPU satellites with `scripts/hpc/add_satellites.sh`. See the [commands reference](Mage.Server.Plugins/Mage.Player.AIRL/docs/commands.md).

## Background

This started as a way to turn a Magic habit into an ML education: first open-source project, first AI project, first Java at this scale. Magic is a brutal environment to learn in. The [comprehensive rules](https://media.wizards.com/2025/downloads/MagicCompRules%2020250207.pdf) run to hundreds of pages, the action space is enormous and mostly illegal at any given moment, and rewards are sparse. That is also what makes it interesting.

Issues and pointers welcome.
