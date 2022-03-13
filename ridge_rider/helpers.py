import matplotlib.pyplot as plt
from torch.distributions import Categorical
import numpy as np
import torch


def get_payoff_values(n_dim):
    """
    Create a matrix n_dim by n_dim with 1.0 on diagonal
    """
    payoff_values = torch.zeros(n_dim, n_dim, requires_grad=False)
    for s in range(n_dim):
        payoff_values[s, s] = 1.0
    return payoff_values


def get_state(p1, p2, joint_policy):
    """
    Convert the actions into one-hot vectors
    Return a concatenation
    Add +1 no actions taken yet (t=0)
    Alternatively 2-hot vector
    """
    a1 = torch.zeros(1, p1.shape[0])
    a1[:, p1.argmax()] = 1
    a2 = torch.zeros(1, p2.shape[0])
    a2[:, p2.argmax()] = 1
    s = torch.cat([a1, a2], dim=1)
    i = torch.argmax(joint_policy.view(-1))
    prob = joint_policy.view(-1)[i]
    return s, prob


def get_expected_return(policy, time_steps, n_dim):
    """
    Get the expected value of the current policy (this is what we take gradients wrt)
    """
    s = torch.zeros(1, n_dim * 2)
    s[0, 0] = 1.0
    payoff_values = get_payoff_values(n_dim)
    r1 = 0
    p_s = 1.
    m = policy.get_m()
    for t in range(time_steps):
        (y, m) = policy(s, m)
        thetas_1 = y.T[:n_dim]
        thetas_2 = y.T[n_dim:]
        assert thetas_1.shape == (n_dim, 1)
        p1 = torch.softmax(thetas_1, 0)
        p2 = torch.softmax(thetas_2, 0)
        assert 1.001 > p1.sum(0) > 0.999
        p_s_comma_a = p_s * p1.matmul(torch.reshape(p2, [1, -1]))
        assert p_s_comma_a.shape == (n_dim, n_dim)
        r1 += (p_s_comma_a * payoff_values).sum()
        s, prob = get_state(p1, p2, p_s_comma_a)
        p_s = p_s * prob
    return r1


def plot_episode(policy_1, policy_2, num_S, time_steps, n_dim=10, beta=1):
    """
    Get the expected value of the current policy (this is what we take gradients wrt)
    """
    s = torch.zeros(1, num_S)
    s[0, 0] = 1.0
    payoff_values = get_payoff_values(n_dim)

    f, ax = plt.subplots(nrows=time_steps, ncols=2, figsize=(6, 6))
    x = np.arange(n_dim)
    p_s = 1.
    m1, m2 = policy_1.get_m(), policy_2.get_m()
    with torch.no_grad():
        for t in range(time_steps):
            (y, m1) = policy_1(s, m1)
            thetas_1 = y.T[:n_dim]
            (y, m2) = policy_2(s, m2)
            thetas_2 = y.T[n_dim:]
            assert thetas_1.shape == (n_dim, 1)
            p1 = torch.softmax(thetas_1 * beta, 0)
            p2 = torch.softmax(thetas_2 * beta, 0)
            p_s_comma_a = p_s * p1.matmul(torch.reshape(p2, [1, -1]))

            ax[t, 0].bar(x, p1.view(-1).numpy())
            ax[t, 1].bar(x, p2.view(-1).numpy())

            p1_H = Categorical(probs=p1.squeeze(1)).entropy()
            p2_H = Categorical(probs=p2.squeeze(1)).entropy()

            ax[t, 0].set_title(
                f'P1: \t t={t}, s={torch.argmax(s).item()}, H={p1_H:.2f}, a $\\sim$ {torch.argmax(p1)}')
            ax[t, 1].set_title(
                f'P2: \t t={t}, s={torch.argmax(s).item()}, H={p2_H:.2f}, a $\\sim$ {torch.argmax(p2)}')

            assert p_s_comma_a.shape == (n_dim, n_dim)

            s, prob = get_state(p1, p2, p_s_comma_a)
            p_s = p_s * prob

    for axisr in ax:
        for axisc in axisr:
            axisc.axis([-0.5, n_dim - 0.5, 0, 1])
            axisc.set_xlabel('action')
            axisc.set_ylabel('probability')
    plt.tight_layout()
    plt.show()
    return
