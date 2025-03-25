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
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean, segment_csr
from tqdm import tqdm
from shell.data.transform import GeometryTransform, AtomNumTransform
from typing import List

from shell.analysis.mol_sample import MolSample, MolSampleList
from shell.data import TMC, TMCDataset
from shell.data.mol import RadiusTransform
from shell.model import EGNNVectorField, GVPVectorField
from shell.model.equiformer_v2._model import EquiformerEncoder
from shell.path import SphMolPath
from shell.utils.for_training import LRScheduler
from shell.utils.settings import QM9_SHELL_RADIUS
from shell.path.solver import SphMolSolver


class ShellFlow(pl.LightningModule):
    def __init__(
        self, 
        config: Union[DictConfig, str],
    ):
        super().__init__()
        self.config = OmegaConf.load(config) if isinstance(config, str) else config

        self._get_loader('test')
        # Setup denoiser
        if self.config['train']['model'] == 'gvp':
            self.vf = GVPVectorField(**self.config['gvp'])
        elif self.config['train']['model'] == 'egnn':
            self.vf = EGNNVectorField(**self.config['egnn'])
        elif self.config['train']['model'] == 'equiformer':
            self.vf = EquiformerEncoder(
                **self.config['equiformer'],
                max_num_elements=len(self.config['sample']['unique_atom_nums'])
            )
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
            'x': nn.CrossEntropyLoss(reduction='none'),
            'v': nn.MSELoss(reduction='none'),
            'r': nn.MSELoss(reduction='none')
        }
    
    def _get_loader(self, split = 'train'):
        processed_path = self.config['dataset']['processed_path']
        dataset = TMCDataset.from_processed(os.path.join(processed_path, f'{split}.pt'))
        if dataset.unique_atom_nums != self.config['sample']['unique_atom_nums']:
            self.config['sample']['unique_atom_nums'] = dataset.unique_atom_nums.tolist()
        loader = DataLoader(
            dataset,
            batch_size=self.config['train']['batch_size'],
            num_workers=self.config['train']['num_workers'],
            shuffle=True if split == 'train' else False
        )
        return loader
        
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_loader = self._get_loader('train')
            self.val_loader = self._get_loader('val')
            self.test_loader = self._get_loader('test')
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader
    
    def test_dataloader(self):
        return self.test_loader
    
    def forward(self, mol: TMC) -> torch.Tensor:
        unique_atom_nums = torch.tensor(self.config['sample']['unique_atom_nums'], device=self.device)
        batch_size = self.config['train']['batch_size']
        mask = (1 - mol.ligand_mask).bool().squeeze(-1)
        
        mol.add_sphere()
        mol0 = mol.build_prior(unique_atom_nums, unchanged_vars=['x', 'r'])
        
        t = torch.rand((batch_size, ), device=self.device)[mol.batch]
        sample_out = self.path.sample(mol0, mol, t, unique_atom_nums, unchanged_vars=['x', 'r'])
        x_pred, dvdt_pred, drdt_pred = self.vf(
            x=sample_out['xt'], 
            pos=GeometryTransform().to_cartes(sample_out['vt'], sample_out['rt']),
            t=t.unsqueeze(-1),
            atom_mask=mol.ligand_mask,
            batch=mol.batch
        )
        dvdt_pred = self.manifold.proju(sample_out['vt'], dvdt_pred)
        dvdt_pred[mask] = 0.0
        drdt_pred[mask] = 0.0
        loss_dict = {
            # 'x': self.loss_fn['x'](x_pred, mol.x.argmax(-1)),
            'v': self.loss_fn['v'](dvdt_pred, sample_out['dvdt']),
            # 'r': self.loss_fn['r'](drdt_pred, sample_out['drdt']) / 5
        }
        for key, value in loss_dict.items():
            if key == 'v':
                value = value * mol.pos.norm(dim=-1, keepdim=True) * 10
            
            value = scatter_mean(value * mol.ligand_mask, mol.batch, dim=0)
            loss_dict[key] = value.mean()
        
        self.log_dict(loss_dict, prog_bar=True, on_step=True, sync_dist=True, batch_size=batch_size)
        loss = loss_dict['v']# + loss_dict['v'] + loss_dict['r']
        return loss
    
    def training_step(self, mol: TMC, batch_idx: int) -> torch.Tensor:
        if not hasattr(self, 'batches_per_epoch'):
            self.batches_per_epoch = len(self.trainer.train_dataloader)
        epoch_exact = self.current_epoch + batch_idx / self.batches_per_epoch
        self.lr_scheduler.step_lr(epoch_exact)
        loss = self(mol)
        self.log('train_total_loss', loss, prog_bar=True, on_step=True, sync_dist=True)
        return loss
    
    def validation_step(self, mol: TMC, batch_idx: int):
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
        data: TMC,
        record_traj: bool = False,
        context: torch.Tensor = None,
        unchanged_vars=['v', 'r']
    ) -> List[TMC]:
        data = data.to(self.device)
        solver = SphMolSolver(
            vf=self.vf,
            x_path=self.path.x_path,
            unique_atom_nums=torch.tensor(self.config['sample']['unique_atom_nums'], device=self.device),
            n_steps=self.config['sample']['timesteps']
        )
        return solver.sample(data, return_traj=record_traj, unchanged_vars=unchanged_vars)