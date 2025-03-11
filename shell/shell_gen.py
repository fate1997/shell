import os
from copy import deepcopy
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean, segment_csr
from tqdm import tqdm

from shell.data import MolDataset
from shell.denoiser import EGNNDenoiser
from shell.analysis.mol_sample import MolSample, MolSampleList
from shell.diffusion.loss import EDMLoss
from shell.diffusion.sample import EDMSampler
from shell.utils.settings import QM9_SHELL_RADIUS
from shell.utils.for_training import LRScheduler


class ShellGen(pl.LightningModule):
    def __init__(
        self, 
        config: Union[DictConfig, str],
    ):
        super().__init__()
        self.config = OmegaConf.load(config) if isinstance(config, str) else config
        self.denoiser = EGNNDenoiser(**self.config['denoiser'])
        device = self.config['train']['trainer_args']['accelerator']
        device = 'cuda' if device == 'gpu' else 'cpu'
        self.loss_fn = EDMLoss(
            denoiser=self.denoiser,
            timesteps=self.config['sample']['timesteps'],
            num_shells=len(QM9_SHELL_RADIUS) - 1,
            # norm_values=(1, 4, 10),
            unique_atom_types=self.config['sample']['unique_atom_nums'],
            device=device
        )
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            dataset = MolDataset(**self.config['dataset'])
            if dataset.unique_atom_nums != self.config['sample']['unique_atom_nums']:
                self.config['sample']['unique_atom_nums'] = dataset.unique_atom_nums

            train_loader, val_loader, test_loader = dataset.get_loaders(
                batch_size=self.config['train']['batch_size'],
                n_train=self.config['train']['n_train'],
                n_val=self.config['train']['n_val'],
                num_workers=self.config['train']['num_workers']
            )
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
    
    def forward(self, batch: Batch) -> torch.Tensor:
        batch_size = batch.batch_size
        focus_shell_id = torch.randint(
            0, len(QM9_SHELL_RADIUS) - 1, (batch_size, 1), device=self.device
        )
        # Remove nodes where the shell_id > focus_shell_id
        mask = batch.shell_id <= focus_shell_id[batch.batch].squeeze(1)
        batch = batch.get_submol(mask)
        loss, loss_part = self.loss_fn(
            x=batch.x,
            p=batch.pos,
            edge_index=batch.edge_index,
            shell_id=batch.shell_id,
            focus_shell_id=focus_shell_id,
            batch=batch.batch,
            return_parts=True
        )
        self.log_dict(loss_part, prog_bar=True, on_step=True, sync_dist=True, batch_size=batch_size)
        return loss
    
    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)
        epoch_exact = self.current_epoch + batch_idx / self.batches_per_epoch
        self.lr_scheduler.step_lr(epoch_exact)
        loss = self(batch)
        self.log('train_total_loss', loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int):
        loss = self(batch)
        self.log(
            'val_total_loss', 
            loss, 
            prog_bar=True, 
            batch_size=batch.batch_size, 
            on_step=True, 
            sync_dist=True
        )

    def configure_optimizers(self):
        config = self.config['lr_scheduler']
        weight_decay = config.get('weight_decay', 0)
        base_lr = config['base_lr']

        optimizer = torch.optim.Adam(self.parameters(), base_lr, weight_decay=weight_decay)
        self.lr_scheduler = LRScheduler(model=self, optimizer=optimizer, **config)
        return optimizer

    @torch.no_grad()
    def sample(
        self,
        num_nodes: torch.Tensor,
        record_traj: bool = False,
        context: torch.Tensor = None,
    ) -> MolSampleList:
        sampler = EDMSampler(
            denoiser=self.denoiser,
            timesteps=self.config['sample']['timesteps'],
            num_shells=len(QM9_SHELL_RADIUS) - 1,
            unique_atom_types=self.config['sample']['unique_atom_nums'],
            device=self.device,
            annealing_rate=self.config['sample']['annealing_rate']
        )
        return sampler.sample(num_nodes, record_traj, context)