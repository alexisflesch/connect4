"""
Where the training happens
"""

import numpy as np
import torch
import torch.optim as optim
from mdp import connect4
import neural_networks
from buffer import replay_buffer
from opponent import smartdumb as opponentAI
import copy
import os
import json
import check_perf


# Hyperparameters
PARAMS = {
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "buffer_size": 1000,
    "batch_size": 8,
    "epsilon": .6069455178992388,
    "epsilon_decay": 0.999995,
    "min_epsilon": 0.1,
    "n_episodes": 900000,
    "tau": 0.01,
    "update_frequency": 1,
    "nn": neural_networks.DDQN5
}

# Initialize networks
q_network = PARAMS['nn']()
target_network = PARAMS['nn']()
target_network.load_state_dict(q_network.state_dict())
target_network.train(False)

# Optimizer
q_optimizer = optim.Adam(q_network.parameters(), lr=PARAMS['learning_rate'])

# Instance of the game
game = connect4()

# Create buffer
buffer = replay_buffer(PARAMS['buffer_size'])


def create_log_and_model():
    """
    Creates a log file with the parameters and returns a model name
    """
    # Create log file with parameters
    nlogs = len(os.listdir('logs'))
    with open(os.path.join("logs", "log_" + str(nlogs) + ".log"), 'w') as f:
        dict_json = copy.deepcopy(PARAMS)
        dict_json['nn'] = str(dict_json['nn'])
        params_json = json.dumps(dict_json, indent=4)
        f.write(params_json + '\n\n')
        f.write("loss,epsilon,perf\n")
    return "model_" + str(nlogs) + ".pth", nlogs


def epsilon_greedy(network):
    """
    Using epsilon-greedy policy, take an action
    """
    if np.random.rand() < PARAMS['epsilon']:
        return np.random.randint(0, 7)
    else:
        with torch.no_grad():
            q_values = network(torch.tensor(game.state, dtype=torch.float32))
            return torch.argmax(q_values).item()


def step(network, player):
    """
    Take an action in game.state for player. Observe :
        - action
        - reward
        - done (is the game over ?)
    """

    action = epsilon_greedy(network)

    # If the action is illegal, stop
    if game.illegal_move(action):
        return action, -1, True

    # Otherwise, play the action
    game.drop_piece(action, player)

    # Check if the game is over
    reward = game.reward(player)
    if reward is not None:
        return action, reward, True

    # If not, the opponent plays
    # At first, opponent is dumb but it gets smarter over time
    opponent = -player
    opponent_agent = opponentAI
    opponent_action = opponent_agent(game, opponent, network)
    game.drop_piece(opponent_action, opponent)

    # Check if the game is over
    reward = game.reward(player)
    if reward is not None:
        return action, reward, True

    return action, 0, False


def log(episode, loss, nlogs):
    """
    Very basic log function to keep track of what's going on
    """
    perf = str(check_perf.perf(q_network, opponentAI))
    print(f"Performance : {perf}")
    print(f"Episode : {episode}")
    print(f"Loss : {loss}")
    print(f"Epsilon : {PARAMS['epsilon']}")
    print("-"*20)
    # Create a new log file in subfolder logs
    with open(os.path.join("logs", "log_"+str(nlogs)+".log"), 'a') as f:
        f.write(str(loss))
        f.write(',')
        f.write(str(PARAMS['epsilon']))
        f.write(',')
        f.write(perf)
        f.write('\n')


def train(q_network, target_network, nlogs):
    """
    Train the agent using DDQN
    """
    loss = torch.tensor(0)

    for episode in range(PARAMS['n_episodes']):
        game.reset()
        done = False

        # Learn to play as player1 and player2
        if np.random.rand() < 0.5:
            player = 1
        else:
            game.drop_piece(np.random.randint(0, 7), 1)
            player = -1

        while not done:
            # Keep a copy of current state for the buffer
            state = copy.deepcopy(game.state)

            # Take action and observe environment
            action, reward, done = step(target_network, player)

            # Add experience to buffer : should we do a deepcopy ?! Looks like it
            buffer.add(state, action, reward, game.state, done)

        # Sample a batch from the buffer if it's full
        try:
            batch = buffer.sample(PARAMS['batch_size'])
        except replay_buffer.BufferNotFull:
            continue

        # Convert batch to tensors
        states = torch.tensor(batch['states'], dtype=torch.float32)
        dones = torch.tensor(batch['dones'], dtype=torch.bool)
        next_states = torch.tensor(
            batch['next_states'], dtype=torch.float32)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32)
        actions = torch.tensor(batch['actions'], dtype=torch.int)

        # Compute target
        with torch.no_grad():
            target = torch.where(dones, rewards, rewards + PARAMS['gamma'] * torch.max(
                target_network(next_states), dim=1).values)

        # Compute loss
        q_values = q_network(states)  # q(s,.)
        q_values = q_values[range(PARAMS['batch_size']), actions]  # q(s,a)
        # MSE Loss
        # loss = torch.nn.functional.mse_loss(target, q_values)
        # L1 Loss
        # loss = torch.nn.functional.l1_loss(target, q_values)
        # Huber Loss
        loss = torch.nn.functional.smooth_l1_loss(target, q_values)

        # Optimize
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()

        # Update target network every target_update episodes
        if episode % PARAMS['update_frequency'] == 0:
            for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                target_param.data.copy_(
                    PARAMS['tau'] * param.data + (1 - PARAMS['tau']) * target_param.data)

        # Decay epsilon every episode
        if PARAMS['epsilon'] > PARAMS['min_epsilon']:
            PARAMS['epsilon'] *= PARAMS['epsilon_decay']

        # Log
        if episode % 20 == 0:
            log(episode, loss.item(), nlogs)


if __name__ == '__main__':
    # Create log file and model name
    model, nlogs = create_log_and_model()
    model_file = os.path.join("models", model)
    # Load existing model
    if os.path.isfile(model_file):
        q_network.load_state_dict(torch.load(model_file))
        target_network.load_state_dict(q_network.state_dict())
        target_network.train(False)
        print("Model loaded, training further...")
    train(q_network, target_network, nlogs)
    torch.save(q_network.state_dict(), model_file)
