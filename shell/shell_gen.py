import os
from copy import deepcopy
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.utils.manifolds import Sphere
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributions import Categorical
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean, segment_csr
from tqdm import tqdm

from shell.analysis.mol_sample import MolSample, MolSampleList
from shell.data import Mol, MolDataset, SphMol
from shell.model import EGNNVectorField, GVPVectorField
from shell.path import SphMolPath
from shell.utils.for_training import LRScheduler
from shell.utils.settings import QM9_SHELL_RADIUS


class ShellFlow(pl.LightningModule):
    def __init__(
        self, 
        config: Union[DictConfig, str],
    ):
        super().__init__()
        self.config = OmegaConf.load(config) if isinstance(config, str) else config

        # Setup denoiser
        if self.config['train']['model'] == 'gvp':
            self.vf = GVPVectorField(**self.config['gvp'])
        elif self.config['train']['model'] == 'egnn':
            self.vf = EGNNVectorField(**self.config['egnn'])
        else:
            raise ValueError(f"Unknown model: {self.config['train']['model']}")
        
        # Setup path and manifold
        self.path = SphMolPath(
            x_scheduler=self.config['path']['x_scheduler'],
            v_scheduler=self.config['path']['v_scheduler'],
            r_scheduler=self.config['path']['r_scheduler']
        )
        self.manifold = Sphere()
        
        # Setup Loss Function
        self.loss_fn = {
            'x': MixturePathGeneralizedKL(self.path.x_path),
            'v': nn.MSELoss(),
            'r': nn.MSELoss()
        }
    
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
    
    def forward(self, mol: Mol) -> torch.Tensor:
        num_atom_types = len(self.config['sample']['unique_atom_nums'])
        batch_size = self.config['train']['batch_size']
        
        sphmol1 = SphMol.from_mol(mol)
        sphmol0 = sphmol1.get_prior(num_atom_types)
        
        t = torch.rand((batch_size, ), device=self.device)[mol.batch]
        sphmolt, dvdt, drdt = self.path.sample(sphmol0, sphmol1, t)
        molt = sphmolt.to_mol()
        x_pred, dvdt_pred, drdt_pred = self.vf(molt, t)
        
        loss_dict = {
            'x': self.loss_fn['x'](x_pred.argmax(-1), sphmol1.x.argmax(-1), sphmolt.x.argmax(-1), t),
            'v': self.loss_fn['v'](dvdt_pred, dvdt),
            'r': self.loss_fn['r'](drdt_pred, drdt)
        }
        
        self.log_dict(loss_dict, prog_bar=True, on_step=True, sync_dist=True, batch_size=batch_size)
        loss = loss_dict['x'] + loss_dict['v'] + loss_dict['r']
        return loss
    
    def training_step(self, mol: Mol, batch_idx: int) -> torch.Tensor:
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)
        epoch_exact = self.current_epoch + batch_idx / self.batches_per_epoch
        self.lr_scheduler.step_lr(epoch_exact)
        loss = self(mol)
        self.log('train_total_loss', loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, mol: Mol, batch_idx: int):
        loss = self(mol)
        self.log(
            'val_total_loss', 
            loss, 
            prog_bar=True, 
            batch_size=mol.batch_size, 
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
        raise NotImplementedError