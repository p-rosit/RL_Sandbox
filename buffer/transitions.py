from collections import namedtuple
import torch

Experience = namedtuple('Experience', ('state', 'action', 'reward'))

def batch_trajectories(experiences):
    batch_experiences = Experience(*zip(*experiences))
    states = []
    actions = []
    rewards = []
    masks = []

    state, action, reward = batch_experiences
    max_trajectory = max(len(ac) for ac in action)

    for i in range(max_trajectory + 1):
        states.append(torch.cat(
            [st[i] if len(st) > i else torch.zeros_like(st[0]) for st in state]
        ).unsqueeze(0))

    for i in range(max_trajectory):
        actions.append(torch.cat(
            [ac[i] if len(ac) > i else torch.zeros_like(ac[0]) for ac in action]
        ).reshape(1, -1))
        rewards.append(torch.cat(
            [rw[i] if len(rw) > i else torch.tensor([0.0]) for rw in reward]
        ).reshape(1, -1))
        masks.append(torch.tensor(
            [len(st) > i + 1 for st in state], dtype=torch.bool
        ).reshape(1, -1))

    states = torch.cat(states, dim=0)
    actions = torch.cat(actions, dim=0)
    rewards = torch.cat(rewards, dim=0)
    masks = torch.cat(masks, dim=0)

    return states, actions, rewards, masks

def batch_episodes(experiences):
    states = []
    actions = []
    rewards = []

    for state, action, reward in experiences:
        states.append(torch.cat(state, dim=0))
        actions.append(torch.cat(action, dim=0))
        rewards.append(torch.cat(reward, dim=0))

    return states, actions, rewards