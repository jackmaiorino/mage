# ComputerPlayerRL

**ComputerPlayerRL** is an advanced AI player for the Mage game engine that integrates reinforcement learning (RL) to make in-game decisions. By extending the base `ComputerPlayer` class, this file leverages an RL model to evaluate game states and select optimal actions—from casting spells and activating abilities to choosing attackers, blockers, and targets.

---

## Overview

- **RL Integration:**  
  Uses an `RLModel` to predict Q-value distributions for various decision types. Decisions are based on the current game state, represented by an `RLState`, and the RL model’s output is used to select the best available option.

- **State Buffering:**  
  Maintains a buffer of RL states (`stateBuffer`) to store experience and assist in training the model over time.

- **Simulation Support:**  
  Simulates game states via the `createSimulation` method to test and validate potential actions before executing them.

---

## Key Functionalities

- **Generic Decision-Making:**
    - `genericChoose(int numOptions, RLState.ActionType actionType, Game game, Ability source)`  
      Creates an RL state for a given decision, sets exploration dimensions, and retrieves Q-value predictions from the model.

- **Ability & Action Selection:**
    - `calculateActions(Game game)`  
      Simulates playable options, filters valid abilities, and selects the best action based on predicted Q-values.
    - `act(Game game, ActivatedAbility ability)`  
      Activates the chosen ability, handling target selection and ensuring proper game events are fired.

- **Combat Decisions:**
    - `selectAttackers(Game game, UUID attackingPlayerId)`  
      Evaluates possible attackers and selects targets based on a matrix of Q-values.
    - `selectBlockers(Ability source, Game game, UUID defendingPlayerId)`  
      Filters eligible blockers and assigns them to attackers using RL-driven decisions.

- **Target & Mode Selection:**
    - `chooseMode(Modes modes, Ability source, Game game)`  
      Uses RL predictions to decide among different modes of an ability, filtering out invalid or non-payable options.
    - `chooseTarget(...)` and overrides of `choose(Outcome, Choice, Game)`  
      Handle various target and choice decisions (e.g., selecting cards, choosing replacement effects) by comparing Q-value outputs from the RL model.

- **Variable Cost Decisions:**
    - `announceXMana(int min, int max, String message, Game game, Ability ability)`  
      Determines the optimal value for variable mana costs by comparing Q-values for each possible option.

---

## Helper Classes

- **QValueWithIndex, AttackOption, BlockOption:**  
  Lightweight classes that pair option indices with their corresponding Q-values. These are used to sort and select the best choices for modes, targets, attackers, and blockers.

---

## Usage & Integration

- **Instantiation:**  
  Create an instance by providing a player name, range of influence, and an initialized `RLModel`:
  ```java
  ComputerPlayerRL rlPlayer = new ComputerPlayerRL("AI_Player", RangeOfInfluence.ALL, myRLModel);
