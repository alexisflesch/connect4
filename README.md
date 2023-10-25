# Connect 4 DDQN

This is an attempt at training an agent with DDQN to play Connect 4. The agent is trained using a neural network, a replay buffer and a target network. Details and demo available here:
https://hire.alexisfles.ch/connect4.html

# Files

## train.py
Main python script to train the agent.

## buffer.py
Contains the ReplayBuffer class to store the agent's experiences.

## mdp.py
Contains the MDP class to represent the Connect 4 game.

## plot_loss.py
A utility script to watch the training loss.

## neural_network.py
A bunch of neural networks to try out.

## opponent.py
A few basic algorithm for the environment's response to the agent's actions.

## check_perf.py
To estimate the agent's performance against a few opponents.


# Notes

## my_DDQN (Grogu)
Simple network with 1 layer with fixed weights :
counts the sums of all alignments of 4  cells (diagonally, horizontally and vertically).

Result : approximately 75% win against dumb opponent (looks for an easy win otherwise plays randomly).

Trained on 50,000 episodes. Training further won't help.


## my_DDQN_2
Same network but with Heaviside activation function after the new cells :
looks where four aligned cells have a sum of +3 or -3.

Same results as my_DDQN (Grogu).


## DDQN3

### First try
Huber loss instead of MSE.
92% win against dumb opponent.

### Push it further by decreasing the learning rate (starting from last weights)
Huber loss again
96% win against dumb opponent.


## DDQN5 (Yoda)

Took 1,000,000 episodes to get a 75% winning rate against smartdumb opponent :
- Half of the time it looks for an easy win and if it can't, it plays randomly
- Half of the time it looks for an easy win, then an immediate threat, then plays randomly.