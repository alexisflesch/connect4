"""
Class to store the replay buffer.
Main attribute :
    - buffer : dictionnary containing the different elements of the buffer
        * states : list 
        * actions : list
        * rewards : list (rewards for taking action in state)
        * next_states : list (next state after taking action in state)
        * dones : list (is game over ?)

Other attributes :
    - buffer : dictionnary containing the different elements of the buffer
    - full : boolean indicating if the buffer is full
    - buffer_size : size of the buffer
    - idx : current index in the buffer

Methods :
    - add : add an experience to the buffer
    - sample : sample a batch of experiences from the buffer
"""

import numpy as np
from copy import deepcopy
from collections import deque


class replay_buffer:
    class BufferNotFull(Exception):
        """
        Exception raised when trying to sample from a buffer that is not full.
        """

        def __init__(self, message="Buffer not full! You can't sample from it."):
            self.message = message
            super().__init__(self.message)

    def __init__(self, buffer_size):
        self.buffer = {
            'states': deque(maxlen=buffer_size),
            'actions': deque(maxlen=buffer_size),
            'rewards': deque(maxlen=buffer_size),
            'next_states': deque(maxlen=buffer_size),
            'dones': deque(maxlen=buffer_size),
        }
        self.full = False  # has the buffer been filled?
        self.buffer_size = buffer_size
        self.idx = 0  # current index

    def add(self, state, action, reward, next_state, done):
        """
        If buffer is not full, add the experience to the buffer.
        Otherwise, change the experience at the current (rolling) index.
        """
        self.buffer['states'].append(deepcopy(state))
        self.buffer['actions'].append(deepcopy(action))
        self.buffer['rewards'].append(deepcopy(reward))
        self.buffer['next_states'].append(deepcopy(next_state))
        self.buffer['dones'].append(deepcopy(done))
        if not self.full:
            self.full = len(self.buffer['states']) == self.buffer_size

    def sample(self, batch_size):
        """
        Returns a batch of experiences sampled from the buffer.
        If the buffer is not full, raise BufferNotFull Error.
        """
        if not self.full:
            raise replay_buffer.BufferNotFull
        else:
            indices = np.random.choice(self.buffer_size, batch_size)
            batch = {
                'states': np.array(self.buffer['states'])[indices],
                'actions': np.array(self.buffer['actions'])[indices],
                'rewards': np.array(self.buffer['rewards'])[indices],
                'next_states': np.array(self.buffer['next_states'])[indices],
                'dones': np.array(self.buffer['dones'])[indices]
            }
            return batch


if __name__ == '__main__':
    B = replay_buffer(4)
    B.add(1, 2, 3, 4, 5)
    B.add(1, 2, 3, 4, 5)
    B.add(1, 2, 3, 4, 5)
    B.add(1, 2, 3, 4, 5)

    try:
        print(B.sample(3))
    except B.BufferNotFull as e:
        print(e)
