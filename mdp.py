"""
Markov Decision Process for the game of connect 4.

- State : list of size 42.
    * 0 for empty cell
    * 1 for player 1's piece
    * -1 for player 2's piece

- Action : integer between 0 and 6 (inclusive - columne to drop piece in).

- Reward :
    * 0 for not losing (yet ?)
    * 0 for drawing
    * 1 for winning
    * -1 for losing
    * -1 for illegal move

- Done : boolean (True if game is over, False otherwise)
"""


import numpy as np
import copy


class connect4():
    def __init__(self):
        self.state = [0 for _ in range(42)]

    def __str__(self):
        """
        Customize the string representation of the game state.
        """
        result = ''
        for row in range(6):
            for col in range(7):
                index = row * 7 + col
                cell = self.state[index]
                if cell == 0:
                    result += ' '
                elif cell == 1:
                    result += 'X'  # You can use 'X' for player 1
                elif cell == -1:
                    result += 'O'  # You can use 'O' for player 2
                if col < 6:
                    result += '|'
            result += '\n'
            if row < 5:
                result += '-------------\n'
        return result

    def reset(self):
        """
        Reset the game to the initial state.
        """
        self.state = [0 for _ in range(42)]

    def whose_turn(self):
        n1, n2 = self.state.count(1), self.state.count(-1)
        if n1 == n2:
            return 1
        else:
            return -1

    def action_space(self):
        """
        Warning : check if the game is over first when training the model
        Player can play in any column where the cell at the top of the column is empty
        """
        return [i for i in range(7) if self.state[i] == 0]

    def drop_piece(self, action, player):
        """
        Warning : modifies the state.
        Drop a piece in the column action
        """
        # Find the index of the first empty slot in column action (0-based index)
        # by going through the column from the bottom to the top
        for row in range(5, -1, -1):
            index = row * 7 + action
            if self.state[index] == 0:
                # Place player's chip in the empty slot
                self.state[index] = player
                break

    def winning_move(self, player):
        """
        Check if a player can win the game
        """
        for action in self.action_space():
            game_copy = copy.deepcopy(self)
            game_copy.drop_piece(action, player)
            if game_copy.check_winner() == player:
                return action
        return None

    def check_winner(self):
        # Check for a win in rows, columns, and diagonals

        def check_line(line):
            for i in range(len(line) - 3):
                if line[i] != 0 and line[i] == line[i+1] == line[i+2] == line[i+3]:
                    return line[i]
            return False

        # Check rows
        for row in range(6):
            for col in range(4):
                u = check_line(self.state[row * 7 + col:row * 7 + col + 4])
                if u:
                    return u

        # Check columns
        for col in range(7):
            u = check_line([self.state[i] for i in range(col, 42, 7)])
            if u:
                return u

        # Check diagonals (bottom-left to top-right)
        for row in range(3):
            for col in range(4):
                diagonal = [self.state[(row + k) * 7 + (col + k)]
                            for k in range(4)]
                u = check_line(diagonal)
                if u:
                    return u

        # Check diagonals (bottom-right to top-left)
        for row in range(3):
            for col in range(3, 7):
                diagonal = [self.state[(row + k) * 7 + (col - k)]
                            for k in range(4)]
                u = check_line(diagonal)
                if u:
                    return u

        return None

    def grid_is_full(self):
        return 0 not in self.state[:7]

    def illegal_move(self, action):
        """
        Check if the action is legal.
        """
        return self.state[action] != 0

    def reward(self, player):
        """
        Check if the game is over and returns the reward :
            - 1 for player 1
            - -1 for player 2
            - 0 for draw
            - None if the game is not over
        """
        u = self.check_winner()
        if u is not None:
            return player*self.check_winner()

        # Is it a draw ?
        if self.grid_is_full():
            return 0

        # Not a draw and not over ?
        return None


if __name__ == '__main__':
    game = connect4()
    print(game)
    game.drop_piece(0, 1)
    game.drop_piece(2, -1)
    game.drop_piece(0, 1)
    game.drop_piece(0, 1)
    game.drop_piece(3, -1)
    game.drop_piece(0, 1)
    game.drop_piece(3, -1)
    game.drop_piece(0, -1)
    game.drop_piece(0, -1)

    print(game)
    print(game.reward(-1))
    print(game.illegal_move(3))
    print(game.action_space())
