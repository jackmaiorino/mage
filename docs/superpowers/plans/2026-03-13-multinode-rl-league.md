# Multi-Node RL-vs-RL League Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-node CP7-bot training with a two-node Slurm job running pure RL-vs-RL self-play across 5 pauper deck archetypes with eval-based PBT.

**Architecture:** GPU node (4x H100) runs gpu_service_host.py for inference/training. CPU node (128+ cores) runs 10 Java RLTrainer processes. Orchestrator on GPU node manages both via srun. PBT uses periodic eval games vs CP7 heuristic bots instead of training rolling winrate.

**Tech Stack:** Java 8+ (XMage engine), Python 3.8+ (PyTorch GPU service), Slurm (het-group jobs), Bash

**Spec:** `docs/superpowers/specs/2026-03-13-multinode-rl-league-design.md`

---

## Chunk 1: Registry and Java Changes

### Task 1: Update PBT Registry

**Files:**
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json`

- [ ] **Step 1: Replace registry contents**

Keep Spy-Combo-A (lines 2-35) and Spy-Combo-B (lines 36-69). Remove all other spy profiles (C through T, lines 70-681). Convert the 4 frozen opponent-pool entries for Elves, Affinity, Wildfire, Rally into trainable profile pairs. Remove the other 4 frozen entries (Caw-Gates, Faeries, Terror, Burn -- these stay as eval opponents only, not in registry).

New registry (10 profiles total):

```json
[
  {
    "profile": "Pauper-Spy-Combo-A",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Spy Combo.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "spy-combo",
    "seed": 1001,
    "train_env": {
      "ENTROPY_START": "0.30",
      "ENTROPY_END": "0.02",
      "RL_ACTION_EPS_START": "0.05",
      "RL_FULL_TURN_RANDOM_START": "0.15",
      "TEMPERATURE_FLOOR": "0.30",
      "ACTOR_LR": "3e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Spy combo A"
  },
  {
    "profile": "Pauper-Spy-Combo-B",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Spy Combo.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "spy-combo",
    "seed": 1002,
    "train_env": {
      "ENTROPY_START": "0.25",
      "ENTROPY_END": "0.03",
      "RL_ACTION_EPS_START": "0.08",
      "RL_FULL_TURN_RANDOM_START": "0.20",
      "TEMPERATURE_FLOOR": "0.25",
      "ACTOR_LR": "2e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Spy combo B"
  },
  {
    "profile": "Pauper-Elves-A",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Elves.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "elves",
    "seed": 2001,
    "train_env": {
      "ENTROPY_START": "0.30",
      "ENTROPY_END": "0.02",
      "RL_ACTION_EPS_START": "0.05",
      "RL_FULL_TURN_RANDOM_START": "0.15",
      "TEMPERATURE_FLOOR": "0.30",
      "ACTOR_LR": "3e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Elves A"
  },
  {
    "profile": "Pauper-Elves-B",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Elves.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "elves",
    "seed": 2002,
    "train_env": {
      "ENTROPY_START": "0.25",
      "ENTROPY_END": "0.03",
      "RL_ACTION_EPS_START": "0.08",
      "RL_FULL_TURN_RANDOM_START": "0.20",
      "TEMPERATURE_FLOOR": "0.25",
      "ACTOR_LR": "2e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Elves B"
  },
  {
    "profile": "Pauper-Rally-A",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono Red Rally.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "rally",
    "seed": 3001,
    "train_env": {
      "ENTROPY_START": "0.30",
      "ENTROPY_END": "0.02",
      "RL_ACTION_EPS_START": "0.05",
      "RL_FULL_TURN_RANDOM_START": "0.15",
      "TEMPERATURE_FLOOR": "0.30",
      "ACTOR_LR": "3e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Rally A"
  },
  {
    "profile": "Pauper-Rally-B",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono Red Rally.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "rally",
    "seed": 3002,
    "train_env": {
      "ENTROPY_START": "0.25",
      "ENTROPY_END": "0.03",
      "RL_ACTION_EPS_START": "0.08",
      "RL_FULL_TURN_RANDOM_START": "0.20",
      "TEMPERATURE_FLOOR": "0.25",
      "ACTOR_LR": "2e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Rally B"
  },
  {
    "profile": "Pauper-Affinity-A",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Grixis Affinity.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "affinity",
    "seed": 4001,
    "train_env": {
      "ENTROPY_START": "0.30",
      "ENTROPY_END": "0.02",
      "RL_ACTION_EPS_START": "0.05",
      "RL_FULL_TURN_RANDOM_START": "0.15",
      "TEMPERATURE_FLOOR": "0.30",
      "ACTOR_LR": "3e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Affinity A"
  },
  {
    "profile": "Pauper-Affinity-B",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Grixis Affinity.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "affinity",
    "seed": 4002,
    "train_env": {
      "ENTROPY_START": "0.25",
      "ENTROPY_END": "0.03",
      "RL_ACTION_EPS_START": "0.08",
      "RL_FULL_TURN_RANDOM_START": "0.20",
      "TEMPERATURE_FLOOR": "0.25",
      "ACTOR_LR": "2e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Affinity B"
  },
  {
    "profile": "Pauper-Wildfire-A",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Jund Wildfire.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "wildfire",
    "seed": 5001,
    "train_env": {
      "ENTROPY_START": "0.30",
      "ENTROPY_END": "0.02",
      "RL_ACTION_EPS_START": "0.05",
      "RL_FULL_TURN_RANDOM_START": "0.15",
      "TEMPERATURE_FLOOR": "0.30",
      "ACTOR_LR": "3e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Wildfire A"
  },
  {
    "profile": "Pauper-Wildfire-B",
    "deck_path": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Jund Wildfire.dek",
    "active": true,
    "train_enabled": true,
    "target_winrate": 0.6,
    "priority": 20,
    "population_group": "wildfire",
    "seed": 5002,
    "train_env": {
      "ENTROPY_START": "0.25",
      "ENTROPY_END": "0.03",
      "RL_ACTION_EPS_START": "0.08",
      "RL_FULL_TURN_RANDOM_START": "0.20",
      "TEMPERATURE_FLOOR": "0.25",
      "ACTOR_LR": "2e-4",
      "GAME_TIMEOUT_SEC": "900",
      "USE_GAE": "1",
      "GAE_AUTO_ENABLE": "1",
      "GAE_LAMBDA_HIGH": "0.99",
      "GAE_LAMBDA_LOW": "0.95",
      "GAE_LAMBDA_DECAY_STEPS": "30000",
      "PPO_GAMMA": "0.995"
    },
    "pbt_mutable_env": [
      "ENTROPY_START", "ENTROPY_END", "RL_ACTION_EPS_START",
      "RL_FULL_TURN_RANDOM_START", "TEMPERATURE_FLOOR", "ACTOR_LR"
    ],
    "notes": "Wildfire B"
  }
]
```

- [ ] **Step 2: Verify deck files exist**

Run: `ls Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck\ -\ Elves.dek Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck\ -\ Mono\ Red\ Rally.dek Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck\ -\ Grixis\ Affinity.dek Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck\ -\ Jund\ Wildfire.dek`
Expected: All 4 files listed, no "No such file" errors.

- [ ] **Step 3: Commit**

```bash
git add Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/league/pauper_spy_pbt_registry.json
git commit -m "feat: update PBT registry for 5-deck RL league (10 profiles)"
```

---

### Task 2: Add LEAGUE_MODE to RLTrainer.java

**Files:**
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/RLTrainer.java`

- [ ] **Step 1: Add LEAGUE_MODE env var declaration**

After line 151 (OPPONENT_SAMPLER declaration), add:

```java
    private static final String LEAGUE_MODE = EnvConfig.str("LEAGUE_MODE", ""); // "rl_only" = no CP7 fallback
```

- [ ] **Step 2: Modify createLeagueOpponent to support rl_only**

Replace the CP7 fallback logic in `createLeagueOpponent` (lines 4106-4117). Currently:

```java
                if (pick.qualified && pick.snapshotPath != null) {
                    String policyKey = "snap:" + pick.snapshotPath.toString();
                    lastOpponentType = String.format(java.util.Locale.US,
                            "META-RL(profile=%s,ep=%d,wr=%.3f,promoted=%s)",
                            pick.profile, pick.episode, pick.baselineWr, pick.promoted);
                    return new ComputerPlayerRL("MetaSnapshotOpp", RangeOfInfluence.ALL, sharedModel, false, false, policyKey);
                }
                int skill = Math.max(1, LEAGUE_POST_HEURISTIC_SKILL);
                lastOpponentType = String.format(java.util.Locale.US,
                        "META-H(profile=%s,skill=%d,wr=%.3f,promoted=%s)",
                        pick.profile, skill, pick.baselineWr, pick.promoted);
                return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
```

Replace with:

```java
                boolean rlOnly = "rl_only".equalsIgnoreCase(LEAGUE_MODE);
                if (pick.qualified && pick.snapshotPath != null) {
                    String policyKey = "snap:" + pick.snapshotPath.toString();
                    lastOpponentType = String.format(java.util.Locale.US,
                            "META-RL(profile=%s,ep=%d,wr=%.3f,promoted=%s)",
                            pick.profile, pick.episode, pick.baselineWr, pick.promoted);
                    return new ComputerPlayerRL("MetaSnapshotOpp", RangeOfInfluence.ALL, sharedModel, false, false, policyKey);
                }
                if (rlOnly) {
                    // No snapshot available yet -- use current training policy (random weights at cold start)
                    String policyKey = "profile:" + pick.profile;
                    lastOpponentType = String.format(java.util.Locale.US,
                            "META-RL-LIVE(profile=%s,wr=%.3f)",
                            pick.profile, pick.baselineWr);
                    return new ComputerPlayerRL("LiveRLOpp", RangeOfInfluence.ALL, sharedModel, false, false, policyKey);
                }
                int skill = Math.max(1, LEAGUE_POST_HEURISTIC_SKILL);
                lastOpponentType = String.format(java.util.Locale.US,
                        "META-H(profile=%s,skill=%d,wr=%.3f,promoted=%s)",
                        pick.profile, skill, pick.baselineWr, pick.promoted);
                return new ComputerPlayer7("Bot-Skill" + skill, RangeOfInfluence.ALL, skill);
```

- [ ] **Step 3: Compile**

Run: `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`
Expected: BUILD SUCCESS

- [ ] **Step 4: Commit**

```bash
git add Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/RLTrainer.java
git commit -m "feat: add LEAGUE_MODE=rl_only for pure RL-vs-RL training"
```

---

### Task 3: Add Eval Mode to RLTrainer.java

**Files:**
- Modify: `Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/RLTrainer.java`

The existing `runEvaluation` method (called at line 1671) evaluates against a default opponent. We need a new mode `league_bench` that accepts opponent deck and skill from env vars, plays N games, and writes results to a file.

- [ ] **Step 1: Add league_bench env vars and mode**

Add env var declarations near line 151:

```java
    private static final String EVAL_OPPONENT_DECK = EnvConfig.str("EVAL_OPPONENT_DECK", "");
    private static final int EVAL_OPPONENT_SKILL = EnvConfig.i32("EVAL_OPPONENT_SKILL", 1);
    private static final int EVAL_NUM_GAMES = EnvConfig.i32("EVAL_NUM_GAMES", 50);
    private static final String EVAL_RESULTS_FILE = EnvConfig.str("EVAL_RESULTS_FILE", "");
```

- [ ] **Step 2: Add league_bench mode in main**

After the `league_eval` case (line 1676), add:

```java
            } else if ("league_bench".equalsIgnoreCase(mode)) {
                new RLTrainer().runLeagueBench();
```

- [ ] **Step 3: Implement runLeagueBench method**

Add method to RLTrainer class:

```java
    private void runLeagueBench() {
        if (EVAL_OPPONENT_DECK.isEmpty()) {
            logger.error("league_bench mode requires EVAL_OPPONENT_DECK");
            return;
        }
        logger.info(String.format("league_bench: %d games vs CP7-Skill%d deck=%s",
                EVAL_NUM_GAMES, EVAL_OPPONENT_SKILL, EVAL_OPPONENT_DECK));

        initCardDatabase();
        int wins = 0;
        int total = 0;
        for (int i = 0; i < EVAL_NUM_GAMES; i++) {
            try {
                Player opponent = new ComputerPlayer7("EvalBot", RangeOfInfluence.ALL, EVAL_OPPONENT_SKILL);
                boolean won = playOneEvalGame(opponent, EVAL_OPPONENT_DECK);
                if (won) wins++;
                total++;
            } catch (Exception e) {
                logger.error("Eval game " + i + " failed: " + e.getMessage());
            }
        }
        double winrate = total > 0 ? (double) wins / total : 0.0;
        String result = String.format("EVAL_RESULT: wins=%d total=%d winrate=%.4f profile=%s",
                wins, total, winrate, MODEL_PROFILE_NAME);
        logger.info(result);
        System.out.println(result);

        if (!EVAL_RESULTS_FILE.isEmpty()) {
            try (java.io.FileWriter fw = new java.io.FileWriter(EVAL_RESULTS_FILE)) {
                fw.write(String.format("%d,%d,%.4f,%s\n", wins, total, winrate, MODEL_PROFILE_NAME));
            } catch (Exception e) {
                logger.error("Failed to write eval results file: " + e.getMessage());
            }
        }
    }

    private boolean playOneEvalGame(Player opponent, String opponentDeckPath) {
        // Use existing game runner infrastructure but with fixed opponent and no training
        ComputerPlayerRL agent = new ComputerPlayerRL("EvalAgent", RangeOfInfluence.ALL, sharedModel, false, false, "train");
        GameRunner runner = new GameRunner(agent, opponent, opponentDeckPath, DECK_LIST_FILE);
        GameRunner.GameResult result = runner.playGame();
        return result != null && result.agentWon;
    }
```

Note: `playOneEvalGame` may need adjustment based on how `GameRunner` is actually instantiated in this codebase. Check `GameRunner` constructor and `playGame()` signature when implementing. The key requirement is: play one game with the RL agent (frozen weights) vs CP7 on specified deck, return win/loss.

- [ ] **Step 4: Compile**

Run: `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`
Expected: BUILD SUCCESS

- [ ] **Step 5: Commit**

```bash
git add Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/rl/RLTrainer.java
git commit -m "feat: add league_bench eval mode for PBT eval benchmark"
```

---

## Chunk 2: Orchestrator Changes

### Task 4: Add Eval Harness to Orchestrator

**Files:**
- Modify: `scripts/hpc/run_spy_pbt_native.py`

- [ ] **Step 1: Add eval config to __init__**

After line 267 (`self.trainer_start_stagger_seconds`), add:

```python
        self.eval_results: Dict[str, float] = {}
        self.eval_num_games = env_int("EVAL_NUM_GAMES_PER_OPPONENT", 50)
        self.eval_opponents: List[Dict[str, str]] = [
            {
                "deck": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono Red Rally.dek",
                "skill": "1",
                "label": "Rally-CP7",
            },
            {
                "deck": "Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/Pauper/Deck - Mono-Blue Terror.dek",
                "skill": "1",
                "label": "Terror-CP7",
            },
        ]
```

- [ ] **Step 2: Implement run_eval_benchmark method**

Add method to `NativeOrchestrator` class, after the `stop_trainer` method:

```python
    def run_eval_benchmark(self) -> Dict[str, float]:
        """Stop trainers, run eval games for each profile, return {profile: winrate}."""
        log("[EVAL] Starting eval benchmark")

        # Stop all trainers
        for profile in list(self.trainers.keys()):
            self.stop_trainer(profile, reason="eval_benchmark")
        time.sleep(2)

        results: Dict[str, float] = {}
        for entry in self.active_entries():
            profile = str(entry.get("profile", "")).strip()
            if not profile:
                continue

            total_wins = 0
            total_games = 0
            for opp in self.eval_opponents:
                eval_results_file = str(
                    self.profile_models_dir(profile) / f"eval_{opp['label']}.csv"
                )
                env = dict(os.environ)
                env["MODE"] = "league_bench"
                env["MODEL_PROFILE"] = profile
                env["EVAL_OPPONENT_DECK"] = opp["deck"]
                env["EVAL_OPPONENT_SKILL"] = opp["skill"]
                env["EVAL_NUM_GAMES"] = str(self.eval_num_games)
                env["EVAL_RESULTS_FILE"] = eval_results_file
                env["RL_ARTIFACTS_ROOT"] = str(self.rl_artifacts_root)

                if self.py_service_mode == "shared_gpu":
                    gpu_slot = 0  # eval uses first GPU slot
                    env["PY_SERVICE_MODE"] = "shared_gpu"
                    env["GPU_SERVICE_ENDPOINT"] = (
                        f"{self.gpu_service_bind_host}:{self.gpu_service_port_base + gpu_slot}"
                    )

                command = self.build_command()
                log(f"[EVAL] Running {profile} vs {opp['label']} ({self.eval_num_games} games)")

                try:
                    proc = subprocess.run(
                        command,
                        cwd=str(self.source_repo_root),
                        env=env,
                        timeout=600,  # 10 min timeout per eval batch
                        capture_output=True,
                        text=True,
                    )
                    # Parse results from file
                    if os.path.isfile(eval_results_file):
                        with open(eval_results_file) as f:
                            parts = f.read().strip().split(",")
                            if len(parts) >= 3:
                                total_wins += int(parts[0])
                                total_games += int(parts[1])
                    else:
                        # Fallback: parse from stdout
                        for line in (proc.stdout or "").splitlines():
                            if line.startswith("EVAL_RESULT:"):
                                for tok in line.split():
                                    if tok.startswith("wins="):
                                        total_wins += int(tok.split("=")[1])
                                    elif tok.startswith("total="):
                                        total_games += int(tok.split("=")[1])
                except subprocess.TimeoutExpired:
                    log(f"[EVAL] TIMEOUT: {profile} vs {opp['label']}")
                except Exception as exc:
                    log(f"[EVAL] ERROR: {profile} vs {opp['label']}: {exc}")

            winrate = total_wins / max(1, total_games)
            results[profile] = winrate
            log(f"[EVAL] {profile}: {total_wins}/{total_games} = {winrate:.3f}")

        self.eval_results = results

        # Append to eval CSV
        eval_csv = self.run_dir / "eval_results.csv"
        write_header = not eval_csv.exists()
        try:
            with open(eval_csv, "a") as f:
                if write_header:
                    f.write("timestamp," + ",".join(sorted(results.keys())) + "\n")
                ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                vals = ",".join(f"{results.get(k, 0.0):.4f}" for k in sorted(results.keys()))
                f.write(f"{ts},{vals}\n")
        except Exception as exc:
            log(f"[EVAL] Failed to write eval CSV: {exc}")

        log(f"[EVAL] Benchmark complete: {results}")
        return results
```

- [ ] **Step 3: Commit**

```bash
git add scripts/hpc/run_spy_pbt_native.py
git commit -m "feat: add eval benchmark harness to orchestrator"
```

---

### Task 5: PBT Uses Eval Winrate

**Files:**
- Modify: `scripts/hpc/run_spy_pbt_native.py`

- [ ] **Step 1: Call eval before PBT exploitation**

Find where `invoke_pbt_exploit` is called in the main loop. Add `run_eval_benchmark()` call before it. Search for the call site (likely in the `run()` or main orchestration loop method).

Before the `invoke_pbt_exploit(snapshots, now_dt)` call, add:

```python
                if self.league_mode == "rl_only":
                    self.run_eval_benchmark()
                    # Restart trainers after eval
                    for entry in self.active_entries():
                        profile = str(entry.get("profile", "")).strip()
                        if profile and profile not in self.trainers:
                            self.start_trainer(entry)
```

- [ ] **Step 2: Add league_mode to __init__**

After the `self.gpu_service_bind_host` line (line 255), add:

```python
        self.league_mode = os.getenv("LEAGUE_MODE", "").strip().lower()
```

- [ ] **Step 3: Modify invoke_pbt_exploit to use eval winrate**

In `invoke_pbt_exploit` (line 1331), replace the `rolling_current` reads with eval winrate when in rl_only mode.

Replace lines 1331-1333:
```python
                wr = snap.get("rolling_current")
                if wr is None:
                    continue
```

With:
```python
                if self.league_mode == "rl_only" and self.eval_results:
                    wr = self.eval_results.get(profile)
                else:
                    wr = snap.get("rolling_current")
                if wr is None:
                    continue
```

Replace lines 1340-1342:
```python
            ordered = sorted(
                candidates,
                key=lambda c: (-float(c["snapshot"]["rolling_current"]), str(c["entry"]["profile"])),
            )
```

With:
```python
            def _pbt_sort_key(c):
                p = str(c["entry"]["profile"])
                if self.league_mode == "rl_only" and self.eval_results:
                    wr = self.eval_results.get(p, 0.0)
                else:
                    wr = float(c["snapshot"].get("rolling_current", 0.0))
                return (-wr, p)

            ordered = sorted(candidates, key=_pbt_sort_key)
```

Also replace lines 1361-1362:
```python
                winner_wr = float(winner["snapshot"]["rolling_current"])
                loser_wr = float(loser["snapshot"]["rolling_current"])
```

With:
```python
                wp = str(winner["entry"]["profile"])
                lp = str(loser["entry"]["profile"])
                if self.league_mode == "rl_only" and self.eval_results:
                    winner_wr = self.eval_results.get(wp, 0.0)
                    loser_wr = self.eval_results.get(lp, 0.0)
                else:
                    winner_wr = float(winner["snapshot"]["rolling_current"])
                    loser_wr = float(loser["snapshot"]["rolling_current"])
```

- [ ] **Step 4: Commit**

```bash
git add scripts/hpc/run_spy_pbt_native.py
git commit -m "feat: PBT uses eval winrate in rl_only league mode"
```

---

## Chunk 3: Multi-Node Infrastructure

### Task 6: Node Discovery and Remote Launch in Orchestrator

**Files:**
- Modify: `scripts/hpc/run_spy_pbt_native.py`

- [ ] **Step 1: Add node discovery to __init__**

After `self.league_mode` (added in Task 5), add:

```python
        # Multi-node discovery
        self.gpu_node = ""
        self.cpu_node = ""
        nodelist = os.getenv("SLURM_JOB_NODELIST", "").strip()
        if nodelist:
            expanded = self._expand_slurm_nodelist(nodelist)
            if len(expanded) >= 2:
                self.gpu_node = expanded[0]
                self.cpu_node = expanded[1]
                log(f"Multi-node: gpu_node={self.gpu_node} cpu_node={self.cpu_node}")
            elif len(expanded) == 1:
                self.gpu_node = expanded[0]
                log(f"Single-node: gpu_node={self.gpu_node}")
```

- [ ] **Step 2: Add nodelist expansion helper**

Add static method to NativeOrchestrator:

```python
    @staticmethod
    def _expand_slurm_nodelist(nodelist: str) -> List[str]:
        """Expand Slurm compact nodelist like 'gpu-a6-[5,7]' or 'gpu-a6-5,compute-01'."""
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()
        except Exception:
            pass
        # Fallback: treat as comma-separated
        return [h.strip() for h in nodelist.split(",") if h.strip()]
```

- [ ] **Step 3: Modify start_trainer to use srun for remote node**

In `start_trainer` (around line 800), wrap the subprocess.Popen command to use srun when cpu_node is set.

Before the `process = subprocess.Popen(command, ...)` call (line 800), add:

```python
        if self.cpu_node:
            command = [
                "srun", "--overlap", "--nodelist=" + self.cpu_node,
                "--ntasks=1", "--cpus-per-task=1",
            ] + command
```

- [ ] **Step 4: Update GPU_SERVICE_ENDPOINT for cross-node**

In `start_trainer` (line 785), the endpoint currently uses `self.gpu_service_bind_host`. When multi-node, it should use the GPU node hostname instead.

Replace line 785:
```python
            env["GPU_SERVICE_ENDPOINT"] = f"{self.gpu_service_bind_host}:{self.gpu_service_port_base + gpu_slot}"
```

With:
```python
            endpoint_host = self.gpu_node if self.gpu_node else self.gpu_service_bind_host
            env["GPU_SERVICE_ENDPOINT"] = f"{endpoint_host}:{self.gpu_service_port_base + gpu_slot}"
```

Apply same pattern to line 787 (metrics endpoint):
```python
            env["GPU_SERVICE_METRICS_ENDPOINT"] = (
                f"http://{endpoint_host}:{self.gpu_service_metrics_port_base + gpu_slot}/metrics"
            )
```

- [ ] **Step 5: Commit**

```bash
git add scripts/hpc/run_spy_pbt_native.py
git commit -m "feat: multi-node support - node discovery and remote trainer launch"
```

---

### Task 7: Update Slurm Script for Het-Group

**Files:**
- Modify: `scripts/hpc/run_spy_pbt.sh`

- [ ] **Step 1: Set GPU_SERVICE_BIND_HOST for multi-node**

Near the top of the script (after the repo_root resolution, around line 44), add:

```bash
# Multi-node: bind GPU service to all interfaces so CPU node can connect
if [[ "$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | wc -l)" -gt 1 ]]; then
  export GPU_SERVICE_BIND_HOST="0.0.0.0"
  echo "Multi-node detected: GPU_SERVICE_BIND_HOST=0.0.0.0"
fi
```

- [ ] **Step 2: Set LEAGUE_MODE from environment**

Near the same location, add passthrough:

```bash
# Pass through league mode if set
if [[ -n "${LEAGUE_MODE:-}" ]]; then
  export LEAGUE_MODE
  echo "LEAGUE_MODE=$LEAGUE_MODE"
fi
```

- [ ] **Step 3: Create het-group submission wrapper**

Create new file `scripts/hpc/submit_league_multinode.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
repo_root="$(pwd)"

BUNDLE="${1:-$(ls -1t local-training/hpc/bundles/rl-runtime-*.tar.gz 2>/dev/null | head -1)}"
if [[ -z "$BUNDLE" ]]; then
  echo "ERROR: No bundle found. Pass path as first argument or place in local-training/hpc/bundles/" >&2
  exit 1
fi
echo "Bundle: $BUNDLE"

JOB_NAME="${JOB_NAME:-league-multinode}"
TIME="${TIME:-12:00:00}"
ACCOUNT="${ACCOUNT:-msml603-class}"

sbatch \
  --job-name="$JOB_NAME" \
  --account="$ACCOUNT" \
  --time="$TIME" \
  --output="local-training/hpc/bundles/${JOB_NAME}_%j.out" \
  --error="local-training/hpc/bundles/${JOB_NAME}_%j.err" \
  --partition=gpu-h100 --gres=gpu:h100:4 --cpus-per-task=16 --mem=128G \
  --hetjob \
  --partition=compute --cpus-per-task=128 --mem=0 \
  --export=ALL,HPC_NATIVE_ORCH=1,MAGE_RL_RUNTIME_TARBALL="$BUNDLE",LEAGUE_MODE=rl_only,TRAIN_PROFILES=10,TOTAL_EPISODES=10000000,PY_SERVICE_MODE=shared_gpu,GAME_LOG_FREQUENCY=500,RUNNER_OVERSUBSCRIPTION_FACTOR=20 \
  scripts/hpc/run_spy_pbt.sh

echo "Submitted. Check with: squeue -u \$USER"
```

Note: The `--hetjob` separator syntax may need adjustment for Zaratan's Slurm version. If `--hetjob` is not supported, use the `#SBATCH hetjob` directive inside `run_spy_pbt.sh` instead. Test on the cluster.

- [ ] **Step 4: Make executable**

```bash
chmod +x scripts/hpc/submit_league_multinode.sh
```

- [ ] **Step 5: Commit**

```bash
git add scripts/hpc/run_spy_pbt.sh scripts/hpc/submit_league_multinode.sh
git commit -m "feat: multi-node Slurm het-group submission for RL league"
```

---

### Task 8: Build Bundle and Verify

- [ ] **Step 1: Compile full project**

Run: `mvn -q -pl Mage.Server.Plugins/Mage.Player.AIRL -am -DskipTests compile`
Expected: BUILD SUCCESS

- [ ] **Step 2: Build runtime bundle**

Run: `powershell -File scripts/hpc/build_rl_runtime_bundle.ps1`
Expected: `rl-runtime-*.tar.gz` created in `local-training/hpc/bundles/`

- [ ] **Step 3: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: multi-node RL-vs-RL league training system"
```
