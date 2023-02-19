import numpy as np
import torch
from torch import nn
from abstract_actor import AbtractActor

import matplotlib.pyplot as plt

from copy import deepcopy

class DenseQLearningActor(AbtractActor):
    def __init__(self, input_size, layer_sizes, output_size, discount=0.99):
        super().__init__()
        layers = (input_size, *layer_sizes, output_size)

        network = []
        for size_in, size_out in zip(layers[:-1], layers[1:]):
            network.append(nn.Linear(size_in, size_out))
            network.append(nn.ReLU())
        network.pop()

        self.discount = discount
        self.network = nn.Sequential(*network)
        self.target_network = nn.Sequential(*network)
        self.ind = 0
        self.history = []

        self.fig = plt.figure(2)
        self.ax = self.fig.subplots()

    def sample(self, state):
        values = self.network(torch.from_numpy(state))
        return torch.argmax(values).item()

    def parameters(self):
        return self.network.parameters()

    def step(self, experiences):
        if self.ind > 1000:
            self.ind = 0
            self.target_network = deepcopy(self.network)

            self.ax.cla()
            self.ax.plot(np.convolve(self.history, np.ones(100) / 100, 'valid'))
            plt.draw()
            plt.pause(0.001)

        self.ind += 1

        states = torch.tensor(np.array([state for state, _, _, _ in experiences]))
        actions = torch.tensor(np.array([action for _, action, _, _ in experiences]))
        rewards = torch.tensor(np.array([reward for _, _, reward, _ in experiences]), dtype=torch.float32)
        next_states = torch.tensor(np.array([next_state for _, _, _, next_state in experiences]))

        targets = rewards + self.discount * self.target_network(next_states).max(dim=1)[0]
        # targets = rewards + self.discount * self.network(next_states).max(dim=1)[0]
        self.optimizer.zero_grad()
        estimate = self.network(states).take(actions)

        loss = self.criterion(estimate, targets)
        self.history.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.optimizer.step()
