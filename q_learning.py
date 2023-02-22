import numpy as np
import torch
from torch import nn
from abstract_actor import AbstractActor
from actor_wrappers import SoftUpdateModel

import matplotlib.pyplot as plt

class DenseQLearningActor(AbstractActor):
    def __init__(self, input_size, layer_sizes, output_size, discount=0.99):
        super().__init__()
        layers = (input_size, *layer_sizes, output_size)

        network = []
        for size_in, size_out in zip(layers[:-1], layers[1:]):
            network.append(nn.Linear(size_in, size_out))
            network.append(nn.ReLU())
        network.pop()

        self.discount = discount
        self.policy_network = nn.Sequential(*network)
        self.target_network = SoftUpdateModel(self.policy_network, tau=0.4)
        # self.ind = 0
        self.history = []

        self.fig = plt.figure(2)
        self.ax = self.fig.subplots()

    def sample(self, state):
        values = self.policy_network(torch.from_numpy(state))
        return torch.argmax(values).item()

    def parameters(self):
        return self.policy_network.parameters()

    def step(self, experiences):
        # if self.ind > 10:
        #     self.ind = 0
        #     # self.target_network = deepcopy(self.network)
        #
        #     # self.ax.cla()
        #     # self.ax.plot(np.convolve(self.history, np.ones(100) / 100, 'valid'))
        #     # plt.draw()
        #     # plt.pause(0.001)
        #
        # self.ind += 1

        # states = torch.tensor(np.array([state for state, _, _, _ in experiences]))
        # actions = torch.tensor(np.array([action for _, action, _, _ in experiences]))
        # rewards = torch.tensor(np.array([reward for _, _, reward, _ in experiences]), dtype=torch.float32)
        # next_states = torch.tensor(np.array([next_state for _, _, _, next_state in experiences]))
        states, actions, rewards, next_states, non_final_mask = super()._step(experiences)

        with torch.no_grad():
            targets = rewards + self.discount * self.target_network(next_states).max(dim=1)[0]
        # targets = rewards + self.discount * self.network(next_states).max(dim=1)[0]
        estimate = self.policy_network(states).take(actions)

        loss = self.criterion(estimate, targets)
        self.history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        self.optimizer.step()

        self.target_network.update(self.policy_network)
