from collections import namedtuple
import torch
import torch.nn.functional as nnf

Experience = namedtuple('Experience', ('state', 'action', 'reward'))

def batch_trajectories(experiences):
    batch_experiences = Experience(*zip(*experiences))

    states, actions, rewards = batch_experiences
    max_trajectory = max(ac.size(0) for ac in actions)

    masks = []
    for state in states:
        mask = torch.zeros((max_trajectory, 1), dtype=torch.bool)
        mask[:state.size(0)-1] = True
        masks.append(mask)

    states = [nnf.pad(state, (0, 0, 0, max_trajectory + 1 - state.size(0))).unsqueeze(0) for state in states]
    actions = [nnf.pad(action, (0, 0, 0, max_trajectory - action.size(0))).unsqueeze(0) for action in actions]
    rewards = [nnf.pad(reward, (0, max_trajectory - reward.size(0))).unsqueeze(1) for reward in rewards]

    states = torch.cat(states, dim=0).swapaxes(0, 1)
    actions = torch.cat(actions, dim=0).swapaxes(0, 1)
    rewards = torch.cat(rewards, dim=1)
    masks = torch.cat(masks, dim=1)

    return states, actions, rewards, masks

def batch_episodes(experiences):
    states = []
    actions = []
    rewards = []

    for state, action, reward in experiences:
        states.append(state)
        actions.append(action)
        rewards.append(reward)

    return states, actions, rewards
