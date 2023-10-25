"""
A few utility functions for the opponent model:
- A smarter one : checks for winning and losing moves, otherwise uses the main agent's policy
- A smart one : checks for winning and losing moves, otherwise plays randomly
- A dumb one : only checks for winning moves, otherwise plays randomly
- A dumber one : plays randomly

All functions below can only play legal moves (as the environment will respond to the main agent with one of these moves).
"""


import numpy as np
import torch


def dumber(game, player, network):
    """
    Take a random action from the action space
    """
    return np.random.choice(game.action_space())


def dumb(game, player, network):
    """
    Look for an easy win, otherwise play randomly
    """
    # Look for a winning move
    action = game.winning_move(player)
    if action is not None:
        return action

    return dumber(game, player, network)


def smart(game, player, network):
    """
    Look for an easy win and immediate threat, otherwise play randomly
    """
    # Look for a winning move
    action = game.winning_move(player)
    if action is not None:
        return action

    # Can we block the opponent's winning move ?
    action = game.winning_move(-player)
    if action is not None:
        return action

    return dumber(game, player, network)


def smarter(game, player, network):
    """
    Look for easy win and immediate threat, otherwise use the main agent's policy
    """
    # Look for a winning move
    action = game.winning_move(player)
    if action is not None:
        return action

    # Look for a losing move
    action = game.winning_move(-player)
    if action is not None:
        return action

    # Otherwise, use the main agent's policy
    with torch.no_grad():
        q_values = network(torch.tensor(game.state, dtype=torch.float32))
        valid_actions = game.action_space()
        q_values = q_values.view(-1, 1)
        q_values = q_values[valid_actions]
        return torch.argmax(q_values).item()


def smartdumb(game, player, network):
    if np.random.rand() < 0.5:
        return smart(game, player, network)
    else:
        return dumb(game, player, network)


def random_opponent(game, player, network):
    """
    Pick an opponent at random
    """
    opponent = np.random.choice([dumber, dumb, smart, smarter])
    return opponent(game, player, network)
