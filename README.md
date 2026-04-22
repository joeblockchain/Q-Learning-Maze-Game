# Q-Learning Maze Game 

## Overview

This project implements a grid-based maze navigation game using the Q-learning algorithm. An autonomous agent learns to navigate from a fixed start position to a goal position while avoiding walls and traps. The environment is visualized with a Pygame-based graphical user interface (GUI), and the training process is also reproduced in Google Colab for analysis and plotting.

This repository is created for Reinforcement Learning Game Design.

## Game Design

- **Environment**: 6×6 grid-world maze.
- **Cell types**:
  - Start cell
  - Goal cell
  - Walls (impassable)
  - Traps (terminal with strong negative reward)
  - Empty cells
- **Episode termination**:
  - The agent reaches the goal.
  - The agent steps into a trap.
  - Maximum step limit is reached.

- **State space**:
  - Each state corresponds to the agent’s position \((x, y)\) on the 6×6 grid.
  - Encoded as a single integer index: `state = y * width + x`.
- **Action space**:
  - Four discrete actions: up, right, down, left.

- **Reward function**:
  - Reach goal: **+10**
  - Step into trap: **−10**
  - Hit wall / move outside grid: **−5**
  - Valid move to empty cell: **−1**

This reward design encourages the agent to reach the goal quickly, avoid traps, and reduce unnecessary or invalid movements.

## Q-Learning Implementation

- **Algorithm**: Tabular Q-learning.
- **Q-table**: 2D array of shape `n_states × n_actions`.
- **Update rule**:
  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr]
  \]
- **Hyperparameters**:
  - Learning rate \(\alpha = 0.1\)
  - Discount factor \(\gamma = 0.99\)
  - Exploration rate \(\epsilon\) starts at 1.0, decays to 0.05.
- **Exploration strategy**: \(\epsilon\)-greedy with multiplicative decay.
- **Training setup**:
  - 800 episodes
  - Maximum 50 steps per episode

During training, the total reward per episode is recorded to evaluate learning performance.

## Q-Table Files

After training, the learned Q-table is saved so that the policy can be reused without retraining.

- **Local (Anaconda/Pygame) version**:
  - The Q-table is automatically saved in NumPy format as:
    - `q_table_trained.npy`
  - Location: same directory as `q_learning_maze.py` (e.g., your Desktop).
- **Colab version**:
  - The Q-table is saved as:
    - `q_table.npy` (NumPy binary format)
    - `q_table.csv` (human-readable CSV format)

These files are included in the repository so that the trained values \(Q(s, a)\) can be inspected or reloaded later.

YouTube Link：https://youtu.be/lTd9camEzLM?si=0ip7JIaEmwAPGg1l



