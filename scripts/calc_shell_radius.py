import argparse
import torch

from shell.data import MolDataset
from shell.utils.geometry import identify_intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='qm9')
    parser.add_argument('--root', type=str, default='dataset')
    parser.add_argument('--processed-name', type=str, default='qm9.pt')
    parser.add_argument('--force-reload', action='store_true')
    parser.add_argument('--n-samples', type=int, default=-1)
    parser.add_argument('--num-intervals', type=int, default=5)
    parser.add_argument('--padding', type=float, default=1.0)
    args = parser.parse_args()
    
    dataset = MolDataset(
        name=args.name,
        root=args.root,
        feature_names=['x', 'z', 'pos', 'radius'],
        processed_name=args.processed_name,
        force_reload=args.force_reload,
        n_samples=args.n_samples,
    )
    
    distances = torch.cat([data.radius for data in dataset])
    max_val = float(int(max(distances) + args.padding) + 1)
    
    intervals = identify_intervals(
        args.num_intervals,
        distances,
        max_val=max_val,
    )
    intervals = [round(interval, 2) for interval in intervals]
    print('Intervals:')
    print(intervals)
    
    
if __name__ == '__main__':
    main()