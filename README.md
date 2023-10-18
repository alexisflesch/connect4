# Connect 4 DDQN

This is an attempt at training an agent with DDQN to play Connect 4. The agent is trained using a neural network, a replay buffer and a target network.

# Files

## train.py
Main python script to train the agent.

## buffer.py
Contains the ReplayBuffer class to store the agent's experiences.

## mdp.py
Contains the MDP class to represent the Connect 4 game.

## plot_loss.py
A utility script to watch the training loss.