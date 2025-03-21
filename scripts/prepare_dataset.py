import argparse

from shell.data.dataset import TMCDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--consider-metal', action='store_true')
    parser.add_argument('--bonded-oct', action='store_true')
    parser.add_argument('--remove-hydrogens', action='store_true')
    parser.add_argument('--metal', nargs='+', default=None)
    parser.add_argument('--force-reload', action='store_true')
    parser.add_argument('--ligatom', nargs='+', default=None)
    parser.add_argument('--processed-dir', type=str, default='dataset/tmqm/processed')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--max-num-atoms', type=int, default=-1)
    parser.add_argument('--n-samples', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    
    dataset = TMCDataset(
        root='dataset',
        feature_names=['x', 'pos', 'z'],
        processed_path='',
        force_reload=args.force_reload,
        remove_hydrogens=args.remove_hydrogens,
        n_samples=args.n_samples,
        metal_type=args.metal,
        ligatom_type=args.ligatom,
        bonded_oct=args.bonded_oct,
        max_num_atoms=args.max_num_atoms,
        consider_metal_type=args.consider_metal
    )
    if args.task == 'one':
        dataset.to_oneligand_task()
    elif args.task == 'all':
        dataset.to_allligand_task()
    else:
        raise ValueError('Invalid task')
    dataset.split_and_save(args.train_ratio, args.val_ratio, args.processed_dir)


if __name__ == '__main__':
    main()