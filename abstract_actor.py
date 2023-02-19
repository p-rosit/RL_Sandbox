import torch

class AbtractActor(torch.nn.Module):
    def sample(self):
        raise RuntimeError

    def step(self, state, action, reward, next_state):
        raise RuntimeError
