"""
Check the performances of an agent against the opponents defined in opponent.py
"""


from mdp import connect4
import numpy as np
import torch
import opponent
import neural_networks
import os
import sys
import matplotlib.pyplot as plt


def game_length(network, agent):
    """
    Play one game of connect-4 between a network and an agent
    """
    # Initialize the game
    game = connect4()

    # Start game with probability .5
    if np.random.random() < 0.5:
        action = agent(game, 1, network)
        game.drop_piece(action, 1)
        network_piece = 1
        agent_piece = -1
    else:
        network_piece = -1
        agent_piece = 1

    # Play until the game is over
    k = 0
    while k < 30:
        k += 1
        # Agent plays
        action = get_move(network, game)
        game.drop_piece(action, agent_piece)

        # Check if the game is over
        reward = game.reward(agent_piece)
        if reward is not None:
            return k, game

        k += 1
        # Opponent plays
        action = agent(game, network_piece, network)
        game.drop_piece(action, network_piece)

        # Check if the game is over
        reward = game.reward(agent_piece)
        if reward is not None:
            return k, game
    return k, game  # Shouldn't happen


def distribution_game_length(network, agent):
    """
    Plot a histogram
    """
    H = []
    for _ in range(1000):
        k, game = game_length(network, agent)
        if k == 3:
            print(game)
        H.append(k)

    categories, valeurs = np.unique(H, return_counts=True)

    # Create a list of colors and labels based on whether the category is even or odd
    colors = ['#149ddd' if cat % 2 == 0 else '#ffb15c' for cat in categories]
    # Create the bar plot with customized colors and labels
    plt.bar(categories, valeurs, color=colors)

    # Add a legend
    plt.title("Game length distribution (Grogu)")
    plt.savefig("game_length_grogu.png")
    plt.show()


def play(network, agent):
    """
    Play one game of connect-4 between a network and an agent
    """
    # Initialize the game
    game = connect4()

    # Start game with probability .5
    if np.random.random() < 0.5:
        action = agent(game, 1, network)
        game.drop_piece(action, 1)
        network_piece = 1
        agent_piece = -1
    else:
        network_piece = -1
        agent_piece = 1

    # Play until the game is over
    k = 0
    while k < 30:
        k += 1
        # Agent plays
        action = get_move(network, game)
        game.drop_piece(action, agent_piece)

        # Check if the game is over
        reward = game.reward(agent_piece)
        if reward is not None:
            return reward

        # Opponent plays
        action = agent(game, network_piece, network)
        game.drop_piece(action, network_piece)

        # Check if the game is over
        reward = game.reward(agent_piece)
        if reward is not None:
            return reward
    return 0  # Shouldn't happen


def perf(network, agent, niter=100):
    """
    Check perf of network against agent
    """
    p = 0
    for _ in range(niter):
        p += play(network, agent)
    return p/niter


def get_move(network, game):
    with torch.no_grad():
        q_values = network(torch.tensor(game.state, dtype=torch.float32))
        valid_actions = game.action_space()
        q_values = q_values.view(-1, 1)
        q_values = q_values[valid_actions]
        return torch.argmax(q_values).item()


def main(network_number, q_network, agent):
    # Create the Q-Network that matches the architecture used for training

    # Load the trained model weights
    q_network.load_state_dict(torch.load(
        os.path.join('models', 'model_' + str(network_number) + '.pth')))
    q_network.eval()  # Set the Q-network to evaluation mode

    # Play 100 games
    wins = 0
    draws = 0
    losses = 0
    for i in range(1000):
        reward = play(q_network, agent)
        if reward == 1:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1

    return wins, draws, losses


if __name__ == '__main__':
    if len(sys.argv) > 1:
        network_number = int(sys.argv[1])
    else:
        network_number = 123
    q_network = neural_networks.my_DDQN()
    # print(main(network_number, q_network,  opponent.smartdumb))
    distribution_game_length(q_network, opponent.dumb)
