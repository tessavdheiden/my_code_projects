import torch
from torch.distributions import Categorical


from helpers import get_expected_return, get_payoff_values, get_state


def get_entropy(policy, n_dim, time_steps):
    """
    Calculate Entropy by summation over time
    """
    s = torch.zeros(1, n_dim * 2)
    s[0, 0] = 1.0
    entropy = 0
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
        entropy += Categorical(probs=p_s_comma_a.view(-1)).entropy()

        assert len(entropy.size()) == 0
        s, prob = get_state(p1, p2, p_s_comma_a)
        p_s = p_s * prob
    return entropy


def GetMIS(policy, n_dim, time_steps):
    """
    Maximsing the entropy
    Minimizing the gradient norm exploded
    """
    old_params = list(policy.parameters())[0][0][0].clone()
    for i in range(1000):
        entropy = -get_entropy(policy, n_dim=n_dim, time_steps=time_steps)
        gradient = torch.autograd.grad(entropy, policy.parameters(), create_graph=True)
        with torch.no_grad():
            for (p, g) in zip(policy.parameters(), gradient):
                p.data -= g
    assert not torch.isclose(old_params, list(policy.parameters())[0][0][0])

    return policy


def Exact_EVals_and_EVecs(policy, time_steps, n_dim):
    """
    Size of gradient is:
    sum([p.shape[0]*p.shape[0] if len(p.shape)==2 else p.shape[0] for p in policy.parameters()])
    """
    pos_reward = get_expected_return(policy, time_steps=time_steps, n_dim=n_dim)
    gradient = torch.autograd.grad(pos_reward, policy.parameters(), create_graph=True)
    gradient = torch.cat([g.flatten() for g in gradient])
    Hessian = [torch.autograd.grad(g, policy.parameters(), create_graph=True) for g in gradient]
    Hessian = torch.cat([torch.cat([e.flatten() for e in v]).unsqueeze(0) for v in Hessian])
    assert Hessian.shape == (gradient.shape[0], gradient.shape[0])

    EVals, EVecs = torch.linalg.eig(Hessian)
    EVecs = torch.real(EVecs.T)
    EVals = torch.real(EVals)
    idx = torch.argsort(-EVals)

    EVals = EVals[idx]
    EVecs = EVecs[idx]
    assert EVals[0] >= EVals[-1]
    return (Hessian, EVals, EVecs)


def GetRidges(policy, n_dim, time_steps, Filter=True, ReturnVals=False, min_val=0.000001):
    """
    Calculates the relevant ridges at the current parameters
    """
    _, EVals, EVecs = Exact_EVals_and_EVecs(policy, time_steps=time_steps, n_dim=n_dim)
    if Filter:
        idx = [i for i in range(len(EVals)) if EVals[i] > min_val]
    else:
        idx = [i for i in range(len(EVals))]

    EVals = EVals[idx]
    EVecs = EVecs[idx]

    EVals = torch.cat([EVals, EVals])
    EVecs = torch.cat([EVecs, -EVecs])
    index = torch.as_tensor(idx + idx)

    idx = torch.argsort(index)
    EVals = EVals[idx]
    EVecs = EVecs[idx]
    index = index[idx]

    return index, EVecs, EVals


def UpdateRidge(theta, e_i, lambda_i, n_dim, time_steps):
    """
    Take new eigenvector with highest overlap,
    which is the dot product
    """
    index, EVecs, Evals = GetRidges(theta, n_dim=n_dim, time_steps=time_steps, Filter=False)
    overlap = EVecs @ e_i
    assert overlap.shape[0] == EVecs.shape[0] and len(overlap.shape) == 1
    index = torch.argmax(overlap)

    return EVecs[index], Evals[index], overlap[index]