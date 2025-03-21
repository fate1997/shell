import os
import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import torch
from rdkit import Chem

from shell.analysis.bond import DistanceBondBuilder, OBBondBuilder
from shell.utils.writer import get_sdf_str, get_xyz_str
from shell.utils.visualizer import visualize_mol, visualize_traj


@dataclass
class MolSample:
    pos: torch.Tensor
    atom_num: torch.Tensor
    
    bond_index: torch.Tensor = None
    bond_order: torch.Tensor = None
    
    pos_traj: List[torch.Tensor] = None
    atom_traj: List[torch.Tensor] = None
    
    @classmethod
    def from_file(cls, path: str):
        if path.endswith('.sdf'):
            rdmol = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)[0]
            #!TODO
    
    def __post_init__(self):
        if self.pos_traj is not None and self.atom_traj is not None:
            assert len(self.pos_traj) == len(self.atom_traj)
        assert self.pos.shape[0] == self.atom_num.shape[0]
        # self.build_bond()
    
    def __len__(self):
        return self.pos.shape[0]
    
    def __repr__(self):
        return f'MolSample(num_atoms={self.num_atoms()})'
    
    @property
    def num_atoms(self) -> int:
        return self.pos.shape[0]
    
    @property
    def xyz_str(self) -> str:
        return get_xyz_str(self.atom_num, self.pos)

    @property
    def sdf_str(self) -> str:
        return get_sdf_str(
            self.atom_num, self.pos, self.bond_index, self.bond_order
        )
    
    def build_bond(self, bond_from: Literal['distance', 'openbabel'] = 'openbabel'):
        if self.bond_index is not None and self.bond_order is not None:
            warnings.warn('Bonds already built, overwriting')
        if bond_from not in ['distance', 'openbabel']:
            raise ValueError(f'Unsupported bond_from: {bond_from}')
        bond_builder = DistanceBondBuilder() if bond_from == 'distance' \
                       else OBBondBuilder()
        edge_index, orders = bond_builder(self.atom_num, self.pos)
        self.bond_index = edge_index
        self.bond_order = orders
        self.bond_from = bond_from
    
    def get_rdmol(self):
        return Chem.MolFromMolBlock(get_sdf_str(
            self.atom_num, self.pos, self.bond_index, self.bond_order
        ), sanitize=True, removeHs=False)
    
    def view(self, assign_bond: bool = False):
        if assign_bond:
            visualize_mol(self.atom_num, self.pos, self.bond_index, self.bond_order)
            return
        visualize_mol(self.atom_num, self.pos)
    
    def view_trajs(self, time_interval: int = 100):
        if self.pos_traj is None or self.atom_traj is None:
            warnings.warn('No trajectory to visualize')
            return
        
        visualize_traj(self.atom_traj, self.pos_traj, interval=time_interval)
    
    def write(self, path: str) -> str:
        if path.endswith('.sdf'):
            content = get_sdf_str(
                self.atom_num, self.pos, self.bond_index, self.bond_order
            )
        elif path.endswith('.xyz'):
            content = get_xyz_str(self.atom_num, self.pos)
        else:
            format = os.path.splitext(path)[1]
            raise ValueError(f'Unsupported file extension: {format}')
        
        with open(path, 'w') as f:
            f.write(content)
        return content


@dataclass
class MolSampleList:
    lst: List[MolSample] = field(default_factory=list)
    
    @classmethod
    def from_batch(
        cls,
        pos: torch.Tensor,
        atom_num: torch.Tensor,
        batch: torch.Tensor,
        pos_traj: List[torch.Tensor] = None,
        atom_traj: List[torch.Tensor] = None,
    ):
        assert pos.shape[0] == atom_num.shape[0]
        if pos_traj is not None and atom_traj is not None:
            assert len(pos_traj) == len(atom_traj)
        
        lst = []
        for i in range(batch.max() + 1):
            mask = batch == i
            if pos_traj is not None and atom_traj is not None:
                pos_traj_i = [traj[mask] for traj in pos_traj]
                atom_traj_i = [traj[mask] for traj in atom_traj]
            else:
                pos_traj_i = None
                atom_traj_i = None
                
            lst.append(MolSample(
                    pos[mask], 
                    atom_num[mask], 
                    pos_traj=pos_traj_i, 
                    atom_traj=atom_traj_i 
                ))
        return cls(lst)
    
    def __repr__(self):
        return f'MolSampleList(num_samples={len(self)})'
    
    def __len__(self):
        return len(self.lst)
    
    def __add__(self, other: 'MolSampleList'):
        return MolSampleList(self.lst + other.lst)
    
    def __getitem__(self, idx: int) -> MolSample:
        return self.lst[idx]
    
    def __iter__(self):
        return iter(self.lst)
    
    def append(self, sample: MolSample):
        self.lst.append(sample)
        
    def extend(self, samples: 'MolSampleList'):
        self.lst.extend(samples.lst)