import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import torch


from helpers import get_expected_return
from policy import Policy


def run_self_play(policy, max_iter, n_dim, time_steps):
    returns = []
    for i in range(max_iter):
        loss = -get_expected_return(policy, n_dim=n_dim, time_steps=time_steps)
        gradient = torch.autograd.grad(
            loss, policy.parameters(), create_graph=True)

        with torch.no_grad():
            for (p, g) in zip(policy.parameters(), gradient):
                p.data -= g
        returns.append(-loss.detach().numpy())

    return returns, policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    config = parser.parse_args()

    args = {
        'n_dims': 4,
        'std': 0.1,
        'min_val': 0,
        'alpha': 1,
        'max_iter': 1000,
        'T': 2}
    n_dim = args['n_dims']
    num_S = n_dim * 2

    pop_size = 10
    # run the algorithm
    all_results = []
    policies = []
    for n in range(pop_size):
        print('Run Number ', n)
        policy = Policy(num_S, n_dim * 2)
        old_params = list(policy.parameters())[0][0][0].clone()

        results, policy = run_self_play(
            policy, args['max_iter'], n_dim, args['T'])

        assert not torch.isclose(
            old_params, list(
                policy.parameters())[0][0][0])

        all_results.append(results)
        policies.append(policy)

    model_dir = Path('./models') / config.model_name
    for i, (result, policy) in enumerate(zip(all_results, policies)):
        folder_name = f'{model_dir}/runs/run{i}/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        torch.save(policy.state_dict(), folder_name + 'policy.pt')
        np.save(folder_name + 'result.npy', result)
    avg_r = np.asarray(all_results).mean(0)
    plt.plot(np.arange(args['max_iter']), avg_r)
    plt.show()
