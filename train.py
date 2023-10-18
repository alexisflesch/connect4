"""
Where the training happens
"""

import numpy as np
import torch
import torch.optim as optim
from mdp import connect4
from neural_network import DDQN
from buffer import replay_buffer
import copy
import os


# Hyperparameters
PARAMS = {
    'learning_rate': 0.001,
    'gamma': 0.99,  # discount factor
    'buffer_size': 1000,
    'batch_size': 16,
    'epsilon': 1,  # exploration rate
    'epsilon_decay': 0.9999,
    'min_epsilon': 0.01,
    'n_episodes': 50000,
    'tau': 0.01,  # soft update factor for target network
}

# Initialize networks
q_network = DDQN()
target_network = DDQN()
target_network.load_state_dict(q_network.state_dict())
target_network.train(False)

# Optimizer
q_optimizer = optim.Adam(q_network.parameters(), lr=PARAMS['learning_rate'])

# Instance of the game
game = connect4()

# Create buffer
buffer = replay_buffer(PARAMS['buffer_size'])

# Keep track of the number of logs to create a new one
nlogs = len(os.listdir('logs'))


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
    opponent = -player
    oppenent_action = np.random.choice(game.action_space())

    # Check if the game is over
    game.drop_piece(oppenent_action, opponent)
    reward = game.reward(player)
    if reward is not None:
        return action, reward, True

    return action, 0, False


def log(episode, loss):
    """
    Very basic log function to keep track of what's going on
    """
    print(f"Episode : {episode}")
    print(f"Loss : {loss}")
    print(f"Epsilon : {PARAMS['epsilon']}")
    print("-"*20)
    # Create a new log file in subfolder logs
    with open(os.path.join("logs", "log_"+str(nlogs)+".txt"), 'a') as f:
        f.write(str(loss))
        f.write(',')
        f.write(str(PARAMS['epsilon']))
        f.write('\n')


def train(q_network, target_network):
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
            game.drop_piece(np.random.randint(0, 7), -1)
            player = -1

        while not done:
            # Keep a copy of current state for the buffer
            state = copy.deepcopy(game.state)

            # Take action and observe environment
            action, reward, done = step(q_network, player)

            # Add experience to buffer : should we do a deepcopy ?! Looks like it
            buffer.add(copy.deepcopy(state), copy.deepcopy(action), copy.deepcopy(
                reward), copy.deepcopy(game.state), copy.deepcopy(done))
            # buffer.add(state, action, reward, game.state, done)

            # Sample a batch from the buffer if it's full
            try:
                batch = buffer.sample(PARAMS['batch_size'])
            except replay_buffer.BufferNotFull:
                continue

            # Compute target
            with torch.no_grad():
                rewards = torch.tensor(batch['rewards'], dtype=torch.float32)
                next_states = torch.tensor(
                    batch['next_states'], dtype=torch.float32)
                dones = torch.tensor(batch['dones'], dtype=torch.bool)

                target = torch.where(dones, rewards, rewards + PARAMS['gamma'] * torch.max(
                    target_network(next_states), dim=1).values)

            # Compute loss
            states = torch.tensor(batch['states'], dtype=torch.float32)
            actions = torch.tensor(batch['actions'], dtype=torch.int)
            q_values = q_network(states)

            loss = torch.nn.functional.mse_loss(
                q_values[range(PARAMS['batch_size']), actions], target)

            # Optimize
            q_optimizer.zero_grad()
            loss.backward()
            q_optimizer.step()

        # Update target network every episode
        for target_param, param in zip(target_network.parameters(), q_network.parameters()):
            target_param.data.copy_(
                PARAMS['tau'] * param.data + (1 - PARAMS['tau']) * target_param.data)

        # Decay epsilon every episode
        if PARAMS['epsilon'] > PARAMS['min_epsilon']:
            PARAMS['epsilon'] *= PARAMS['epsilon_decay']

        # Log
        if episode % 10 == 0:
            log(episode, loss.item())


def test():
    print(game)
    for _ in range(25):
        _ = input("Press a key to continue")
        _, reward, done = step(DDQN(), game.state, 1)
        print(game)
        print(reward, done)


if __name__ == '__main__':
    model = 'model4.pth'
    # Load existing model
    if os.path.isfile(model):
        q_network.load_state_dict(torch.load(model))
        target_network.load_state_dict(q_network.state_dict())
        target_network.train(False)
        print("Model loaded, training further...")
    train(q_network, target_network)
    torch.save(q_network.state_dict(), model)
