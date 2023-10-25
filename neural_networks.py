"""
Definition of the Neural Network used to train the agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class my_DDQN(nn.Module):
    def __init__(self):
        super(my_DDQN, self).__init__()
        self.fc1 = nn.Linear(111, 7)

    def compute_sums(self, states):
        # Reshape the states to batch_size x 6 x 7
        states = states.view(-1, 6, 7)

        # Compute sums of four consecutive elements in a row
        row_sums = torch.zeros(states.shape[0], 24)
        for row in range(6):
            for i in range(4):
                s = torch.sum(states[:, row, i:i + 4], dim=1)
                row_sums[:, row*4 + i] = s

        # Compute sums of four consecutive elements in a column
        col_sums = torch.zeros(states.shape[0], 21)
        for col in range(7):
            for i in range(3):
                s = torch.sum(states[:, i:i + 4, col], dim=1)
                col_sums[:, col*3 + i] = s

        # Sums on downward diagonals
        upward_diag_sums = torch.zeros(states.shape[0], 12)
        for off in range(3):
            for col in range(4):
                # states[:,:,col:] removes the first columns
                # diagonal() extracts the diagonal of the states with the given offset
                diags = torch.diagonal(
                    states[:, :, col:], offset=-off, dim1=1, dim2=2)
                upward_diag_sums[:, off*4 + col] = diags.sum(dim=1)

        downward_diag_sums = torch.zeros(states.shape[0], 12)

        # Sums on upward diagonals
        upward_diag_sums = torch.zeros(states.shape[0], 12)
        for off in range(3):
            for col in range(4):
                # rotated_states = torch.rot90(states, k=1, dims=(1, 2))#Not onnx compatible
                rotated_states = states.flip(1)
                diagonal = rotated_states[:, off:off+4, col:col+4]
                diagonal_sum = diagonal.sum(dim=(1, 2))
                upward_diag_sums[:, off*4 + col] = diagonal_sum

        # Concatenate all sums and reshape into a single column tensor
        sums = torch.cat(
            (row_sums, col_sums, upward_diag_sums, downward_diag_sums), dim=1)

        return sums

    def forward(self, game_state):
        # Compute the sums
        sums = self.compute_sums(game_state)
        input_features = torch.cat((game_state.view(-1, 42), sums), dim=1)

        # Pass input features through the network
        x = self.fc1(input_features)
        return x


class my_DDQN2(nn.Module):
    def __init__(self):
        super(my_DDQN2, self).__init__()
        self.fc1 = nn.Linear(180, 7)

    def compute_sums(self, states):
        # Reshape the states to batch_size x 6 x 7
        states = states.view(-1, 6, 7)

        # Compute sums of four consecutive elements in a row
        row_sums = torch.zeros(states.shape[0], 24)
        for row in range(6):
            for i in range(4):
                s = torch.sum(states[:, row, i:i + 4], dim=1)
                row_sums[:, row*4 + i] = s

        # Compute sums of four consecutive elements in a column
        col_sums = torch.zeros(states.shape[0], 21)
        for col in range(7):
            for i in range(3):
                s = torch.sum(states[:, i:i + 4, col], dim=1)
                col_sums[:, col*3 + i] = s

        # Sums on downward diagonals
        upward_diag_sums = torch.zeros(states.shape[0], 12)
        for off in range(3):
            for col in range(4):
                # states[:,:,col:] removes the first columns
                # diagonal() extracts the diagonal of the states with the given offset
                diags = torch.diagonal(
                    states[:, :, col:], offset=-off, dim1=1, dim2=2)
                upward_diag_sums[:, off*4 + col] = diags.sum(dim=1)

        downward_diag_sums = torch.zeros(states.shape[0], 12)

        # Sums on upward diagonals
        upward_diag_sums = torch.zeros(states.shape[0], 12)
        for off in range(3):
            for col in range(4):
                rotated_states = torch.rot90(states, k=1, dims=(1, 2))
                diagonal = rotated_states[:, off:off+4, col:col+4]
                diagonal_sum = diagonal.sum(dim=(1, 2))
                upward_diag_sums[:, off*4 + col] = diagonal_sum

        # Concatenate all sums and reshape into a single column tensor
        sums = torch.cat(
            (row_sums, col_sums, upward_diag_sums, downward_diag_sums), dim=1)

        # Heaviside activation
        big_sums = torch.where(sums == 3, sums, torch.zeros(sums.shape))
        small_sums = torch.where(sums == -3, sums, torch.zeros(sums.shape))
        return torch.cat((big_sums, small_sums), dim=1)

    def forward(self, game_state):
        # Compute the sums
        sums = self.compute_sums(game_state)
        input_features = torch.cat((game_state.view(-1, 42), sums), dim=1)

        # Pass input features through the network
        x = self.fc1(input_features)
        return x


class DDQN5(nn.Module):
    def __init__(self):
        super(DDQN5, self).__init__()
        self.fc1 = nn.Linear(42, 180)
        self.fc2 = nn.Linear(180, 180)
        self.fc3 = nn.Linear(180, 180)
        self.fc4 = nn.Linear(180, 180)
        self.fc5 = nn.Linear(180, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        out = self.fc5(x)
        return out


class DDQN4(nn.Module):
    def __init__(self):
        super(DDQN4, self).__init__()
        self.fc1 = nn.Linear(42, 111)
        self.fc2 = nn.Linear(111, 7)

    def forward(self, x):
        x = self.fc1(x)
        out = self.fc2(x)
        return out


class DDQN3(nn.Module):
    def __init__(self):
        super(DDQN3, self).__init__()
        self.fc1 = nn.Linear(42, 180)
        self.fc2 = nn.Linear(180, 180)
        self.fc3 = nn.Linear(180, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class DDQN(nn.Module):
    def __init__(self):
        super(DDQN, self).__init__()
        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class DDQN2(nn.Module):
    def __init__(self):
        super(DDQN2, self).__init__()
        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        out = self.fc4(x)
        return out


class DDQN3(nn.Module):
    def __init__(self):
        super(DDQN3, self).__init__()
        self.fc1 = nn.Linear(42, 42*4)
        self.fc2 = nn.Linear(42*4, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class DDQN_hint(nn.Module):
    def __init__(self):
        super(DDQN_hint, self).__init__()
        self.fc1 = nn.Linear(42 + 13, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 7)

    def compute_sums(self, game_state):
        # Compute sums for each line, row, and diagonal
        line_sums = torch.sum(game_state.view(-1, 7, 6), dim=2)
        column_sums = torch.sum(game_state.view(-1, 7, 6), dim=1)

        return torch.cat([line_sums, column_sums], dim=1)

    def forward(self, game_state):
        # Compute the sums
        sums = self.compute_sums(game_state)

        # Append the sums to the input features
        input_features = torch.cat((game_state.view(-1, 42), sums), dim=1)

        # Pass input features through the network
        x = torch.relu(self.fc1(input_features))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
