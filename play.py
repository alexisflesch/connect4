"""
Quick and dirty script to play against the AI
"""


import torch
import mdp as c4
from neural_network import DDQN


# Create an instance of the connect4 game
game = c4.connect4()

# Create the Q-Network that matches the architecture used for training
q_network = DDQN()

# Load the trained model weights
q_network.load_state_dict(torch.load('model.pth'))
q_network.eval()  # Set the Q-network to evaluation mode


def user_input():
    while True:
        try:
            action = int(input("Enter your move (0-6): "))
            if 0 <= action <= 6:
                return action
            else:
                print("Invalid input. Please enter a number between 0 and 6.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 6.")


def main():
    print(game)  # Display the initial game state

    while True:
        # User's turn
        user_action = user_input()
        # Update state after user's move
        game.drop_piece(user_action, 1)
        reward = game.reward(1)

        if reward is not None:
            print(game)
            if reward == 1:
                print("You win!")
            elif reward == -1:
                print("You lose!")
            else:
                print("It's a draw!")
            break

        # DQN's turn
        # Obtain Q-values for all actions
        q_values = q_network(torch.tensor(game.state, dtype=torch.float32))

        # Get the action space from the MDP
        action_space = game.action_space()

        # Filter out illegal actions (actions not in the action space)
        argmax = float('-inf')
        for action in action_space:
            # Get the value as a float
            action_value = q_values[action].item()
            if action_value > argmax:
                argmax = action_value
                dqn_action = action

        print(q_values)

        game.drop_piece(dqn_action, -1)
        reward = game.reward(-1)

        print(game)
        if reward is not None:
            if reward == 1:
                print("You lose!")
            elif reward == -1:
                print("You win!")
            else:
                print("It's a draw!")
            break


if __name__ == '__main__':
    main()
