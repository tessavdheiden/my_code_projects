from pathlib import Path
import argparse
import torch


from policy import Policy
from helpers import plot_episode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--run_num", type=int)
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

    model_dir = Path('./models') / config.model_name
    folder_name = f'{model_dir}/runs/run{config.run_num}/policy.pt'
    policy = Policy(num_S, n_dim * 2)
    policy.load_state_dict(torch.load(folder_name))
    plot_episode(policy, policy, num_S, time_steps=args['T'], n_dim=n_dim)

