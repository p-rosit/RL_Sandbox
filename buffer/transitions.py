from collections import namedtuple
import torch
import numpy as np

Experience = namedtuple('Experience', ('state', 'action', 'reward'))

def batch_trajectories(experiences):
    batch_experiences = Experience(*zip(*experiences))

    state, action, reward = batch_experiences
    max_trajectory = max(ac.shape[0] for ac in action)

    masks = np.zeros((max_trajectory, len(state)), dtype=bool)
    states = np.zeros((max_trajectory + 1, len(state), state[0].shape[-1]), dtype=state[0].dtype)
    actions = np.zeros((max_trajectory, len(action), 1), dtype=action[0].dtype)
    rewards = np.zeros((max_trajectory, len(reward)), dtype=reward[0].dtype)

    for i, st in enumerate(state):
        states[:st.shape[0], i] = st
        masks[:st.shape[0]-1, i] = True

    for i, (ac, rw) in enumerate(zip(action, reward)):
        actions[:ac.shape[0], i] = ac
        rewards[:rw.shape[0], i] = rw

    states = torch.from_numpy(states)
    actions = torch.from_numpy(actions)
    rewards = torch.from_numpy(rewards)
    masks = torch.from_numpy(masks)

    return states, actions, rewards, masks

def batch_episodes(experiences):
    states = []
    actions = []
    rewards = []

    for state, action, reward in experiences:
        states.append(torch.from_numpy(state))
        actions.append(torch.from_numpy(action))
        rewards.append(torch.from_numpy(reward))

    return states, actions, rewards
