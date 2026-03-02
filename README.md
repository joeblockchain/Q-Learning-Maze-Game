# Q-Learning Maze Game (CDS524 Assignment 1)

## Overview

This project implements a grid-based maze navigation game using the Q-learning algorithm. An autonomous agent learns to navigate from a fixed start position to a goal position while avoiding walls and traps. The environment is visualized with a Pygame-based graphical user interface (GUI), and the training process is also reproduced in Google Colab for analysis and plotting.

This repository is created for CDS524 Assignment 1 – Reinforcement Learning Game Design.

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

### Loading the Q-table (example)

import numpy as np

# After creating the agent instance:
agent.q_table = np.load("q_table.npy")
This allows you to run demo or evaluation with the previously learned policy.
User Interface and Interaction
The Pygame-based GUI provides several modes:
Main Menu:
Train Agent
Watch Agent
Human Play
Quit
Training Mode:
Runs Q-learning training episodes.
Shows grid, agent position, episode, step, last action, reward, total reward, and epsilon.
After training is finished, the Q-table is saved and the game returns to the menu.
Demo Mode:
Uses the trained Q-table with a greedy policy (
ϵ
=
0
ϵ=0).
The agent repeatedly demonstrates the learned path from start to goal.
Press M to return to the main menu.
Human Play Mode:
The user controls the agent with arrow keys (up/right/down/left).
The same reward rules apply.
Press M to return to the main menu.
Assignment 1 – Your Name/
├─ q_learning_maze.py          # Main Pygame game and Q-learning implementation (local)
├─ CDS524_Qlearning_Maze.ipynb # Colab notebook: training and evaluation (no GUI)
├─ q_table_trained.npy         # Trained Q-table from local run
├─ q_table.npy                 # Trained Q-table from Colab
├─ q_table.csv                 # Q-table in CSV format (Colab)
├─ report.pdf                  # Written report (1000–1500 words)
├─ README.md                   # This file
└─ other assets (screenshots, etc.)
Running the Game (Anaconda / Local)
Create and activate the Conda environment:
conda create -n rl_game python=3.10 -y
conda activate rl_game
pip install pygame numpy
Go to the directory where q_learning_maze.py is located (for example, Desktop):
cd %USERPROFILE%\Desktop
python q_learning_maze.py
Use the main menu to:
Train the agent
Watch the trained agent
Play the game manually
After training, the file q_table_trained.npy will appear in the same directory.
Running the Colab Notebook
Open CDS524_Qlearning_Maze.ipynb in Google Colab.
Run the cells in order:
Install dependencies
Define the environment and Q-learning agent
Train the agent and record rewards
Plot the training reward curve
Save q_table.npy and q_table.csv
Download the Q-table files and upload them to this repository (if needed).
References
Q-learning Tank War example:
https://github.com/KI-cheng/QlearningTankWar
Tetris deep Q-learning example:
https://github.com (course-provided reference)
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.

