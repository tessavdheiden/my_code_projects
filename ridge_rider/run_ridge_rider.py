import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import torch
import copy

from helpers import get_expected_return
from policy import Policy
from helpers_rr import GetRidges, UpdateRidge, GetMIS


def run_ridge_rider(policy, max_iter, n_dim, time_steps, n_ridges):
    returns = []
    policies = []

    (index, EVecs, EVals) = GetRidges(policy, n_dim=n_dim, time_steps=time_steps, min_val=0.0005)
    for idx_ridge in range(n_ridges):
        e_i, lambda_i = EVecs[idx_ridge], EVals[idx_ridge]
        shapes = [weight.shape[0] * weight.shape[1] for weight in policy.parameters()]
        assert all([len(weight.shape) == 2 for weight in policy.parameters()])

        theta = copy.deepcopy(policy)
        for i in range(max_iter):
            ridge = torch.split(e_i, shapes)
            with torch.no_grad():
                for (p, r) in zip(theta.parameters(), ridge):
                    p.data += r.view(p.shape)
            e_i, lambda_i, overlap = UpdateRidge(theta, e_i, lambda_i, n_dim, time_steps)

        loss = -get_expected_return(theta, n_dim=n_dim, time_steps=time_steps)
        returns.append(-loss.detach().numpy())
        policies.append(copy.deepcopy(theta))
    return returns, policies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    config = parser.parse_args()

    args = {
        'n_dims': 4,
        'std': 0.1,
        'min_val': 0,
        'alpha': 1,
        'max_iter': 2,
        'T': 2,
        'n_ridges': 2}
    n_dim = args['n_dims']
    num_S = n_dim * 2

    pop_size = 1
    # run the algorithm
    all_results = []
    policies_of_policies = []
    for n in range(pop_size):
        print('Run Number ', n)
        policy = Policy(num_S, n_dim * 2)
        policy = GetMIS(policy, n_dim, time_steps=args['T'])

        results, policies = run_ridge_rider(
            policy, args['max_iter'], n_dim, args['T'], n_ridges=args['n_ridges'])

        all_results.append(results)
        policies_of_policies.append(policies)

    model_dir = Path('./models') / config.model_name
    for i, (result, policies) in enumerate(zip(all_results, policies_of_policies)):
        folder_name = f'{model_dir}/runs/run{i}/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for j, policy in enumerate(policies):

            torch.save(policy.state_dict(), folder_name + f'policy{j}.pt')

        np.save(folder_name + 'result.npy', result)

