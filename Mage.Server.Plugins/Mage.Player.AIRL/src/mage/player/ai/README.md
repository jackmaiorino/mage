# Projects

# Mage AI Reinforcement Learning

### **Summary**
This project began as a way for me to leverage my passion for Magic The Gathering(MTG) into learning about AI. It is an AI system for the Mage game client(a free open source MTG client), utilizing reinforcement learning (RL) techniques to make decisions. This was my first time working on an open source project, first time working with AI, and first time working with java at this scale. If you see something that you deem heinous to any of these practices, please let me know as this was a massive task to learn. At the time of writing this I have been working on the project for almost 6 months now with varried intensity when life allows.
### **Game Engine Updates**
Magic is an incredibly unforgiving game to code up due to the countless complex interactions. If you are unfamiliar with the game, I encourage you to just glance at the official rules here: [Magic: The Gathering Comprehensive Rules (2025)](https://media.wizards.com/2025/downloads/MagicCompRules%2020250207.pdf) to get a sense of the complexity. This meant that I ended up spending a lot of time improving the game runner itself to better legally mask the action space to my model. Any of my changes you see outside of the Mage.Player.AIRL dir are just game engine improvements.

## Key Components

### 1. **BatchPredictionRequest**
This class manages batch processing of prediction requests using a neural network. It handles the queuing of requests and processes them in batches to optimize performance and resource utilization.

### 2. **ComputerPlayerRL**
An AI player class that extends the capabilities of a standard computer player by integrating reinforcement learning models. It uses the `RLModel` to make decisions during gameplay.

### 3. **RLModel**
The core of the AI's decision-making process, this class encapsulates a neural network that predicts the best actions based on the current game state. It supports both training and inference modes, allowing the model to learn from past games and improve over time.

### 4. **EmbeddingManager**
Handles the creation and management of text embeddings for game elements, using OpenAI's API. This component is crucial for converting game text into a format that the neural network can process.

### 5. **NeuralNetwork**
Defines the architecture and configuration of the neural network used in the RLModel. It includes layers for batch normalization, dropout, and dense connections, optimized with the Adam optimizer.

### 6. **RLState**
Represents the state of the game at any given time, including player stats, card features, and game actions. This class is essential for feeding data into the neural network for predictions.

### 7. **RLTrainer**
Manages the training process of the RLModel, coordinating multiple game simulations to refine the AI's strategies. It uses a multi-threaded approach to run numerous game instances in parallel, accelerating the learning process.

## Features

- **Reinforcement Learning**: The AI uses RL to continuously improve its gameplay strategies by learning from past experiences.
- **Batch Processing**: Efficiently handles multiple prediction requests to optimize performance.
- **Neural Network Integration**: Utilizes a sophisticated neural network architecture for decision-making.
- **OpenAI Embeddings**: Leverages OpenAI's API for generating embeddings, enhancing the AI's understanding of game elements.
- **Multi-threaded Training**: Accelerates the training process by running multiple game simulations concurrently.

##  Win Rate Over Time

| Date       | Win Rate | Notes |
|------------|---------|-------|
| ??? | ?%   | ???? |
