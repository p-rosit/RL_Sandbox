import numpy as np
import torch
from torch import nn
from abstract_actor import AbtractActor

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
        self.network = torch.nn.Sequential(*network)

    def sample(self, state):
        values = self.network(torch.from_numpy(state))
        return torch.argmax(values)

    def step(self, experiences):
        states = torch.tensor(np.array([state for state, _, _, _ in experiences]))
        actions = torch.tensor(np.array([action for _, action, _, _ in experiences]))
        rewards = torch.tensor(np.array([reward for _, _, reward, _ in experiences]))
        next_states = torch.tensor(np.array([next_state for _, _, _, next_state in experiences]))

        targets = rewards + self.discount * self.network(states).max(dim=1)[0]

        print(targets)
        # print(actions)
        error(':)')
