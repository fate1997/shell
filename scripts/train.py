import argparse
import os
import pathlib
from datetime import datetime
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from shell import ShellFlow

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / 'config/default.yaml'
DEFAULT_OUTDIR = '.shell_output/'


def parse_args():
    config = OmegaConf.load(DEFAULT_CONFIG_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-name', type=str, default='test')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--n-layers', type=int, default=7)
    parser.add_argument('--force-reload', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--processed-name', type=str, 
                        default=config['dataset']['processed_name'])
    parser.add_argument('--model', type=str, default='egnn')
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()
    
    for key, value in vars(args).items():
        for k, v in config.items():
            for kk, vv in v.items():
                if key == kk:
                    config[k][kk] = value
                    print(f'Overwrite {kk} with {value}')
    config['train']['task_name'] = args.task_name
    config['train']['trainer_args']['accelerator'] = args.device
    
    return config


def train(config: DictConfig):
    # 1. Setup output directory
    task_name = config['train']['task_name']
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    outdir_name = f'{task_name}({date})'
    outdir = os.path.join(DEFAULT_OUTDIR, outdir_name)
    os.makedirs(outdir, exist_ok=True)
    
    # 2. Setup wandb
    wandb_config = config['wandb']
    wandb_config['name'] = config['train']['task_name']
    wandb_config['save_dir'] = str(outdir)
    wandb_config['id'] = wandb_config['name']
    wandb_config = OmegaConf.to_container(wandb_config)
    wandb_logger = WandbLogger(
        config=OmegaConf.to_container(config), 
        **wandb_config
    )
    
    # 3. Setup checkpoint
    ckpt_dir = os.path.join(outdir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_config = config['checkpointing']
    checkpoint_config['dirpath'] = str(ckpt_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_config)
    
    # 4. Setup model
    model = ShellFlow(config)
    OmegaConf.save(config, os.path.join(outdir, 'config.yaml'))
    
    # 5. Training
    pbar_callback = TQDMProgressBar(refresh_rate=20)
    trainer_args = config['train']['trainer_args']
    trainer = pl.Trainer(
        **trainer_args,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, pbar_callback],
    )
    trainer.fit(model)


if __name__ == '__main__':
    train(parse_args())