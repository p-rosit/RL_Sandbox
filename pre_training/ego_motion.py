import matplotlib.pyplot as plt
from buffer.transitions import batch_transitions

def pre_train_state_dynamics(net, optimizer, buffer, epochs, batch_size, plot=False):
    loss_history = []
    fig = plt.figure() if plot else None
    ax = fig.subplots() if fig is not None else None

    for _ in range(epochs):
        experiences = buffer.sample(batch_size)
        states, actions, _, next_states, non_final_mask = batch_transitions(experiences)
        states = states[non_final_mask]
        actions = actions[non_final_mask]

        loss = net.pre_loss(states, actions, next_states)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if plot:
            ax.cla()
            ax.plot(loss_history)
            plt.draw()
            plt.pause(0.0001)
