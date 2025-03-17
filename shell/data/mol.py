from typing import Optional

import torch
from torch_geometric.data import Data
from dataclasses import dataclass

from shell.utils.settings import MID_RADIUS


class Mol(Data):
    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        desc: Optional[torch.Tensor] = None,
        smiles: Optional[str] = None,
        name: Optional[str] = None,
        z: Optional[torch.Tensor] = None,
        id: Optional[torch.Tensor] = None,
        shell_id: Optional[torch.Tensor] = None,
        radius: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super(Mol, self).__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        
        self.desc = desc 
        self.smiles = smiles
        self.name = name
        self.z = z
        self.y = y
        self.id = id
        self.shell_id = shell_id
        self.radius = radius
    
    def get_submol(
        self,
        mask: torch.Tensor,
    ) -> 'Mol':
        assert self.edge_index is None
        mol = Mol(
            x=self.x[mask],
            y=self.y,
            pos=self.pos[mask],
            desc=None if getattr(self, 'desc', None) is None else self.desc[mask],
            smiles=None if getattr(self, 'smiles', None) is None else self.smiles,
            name=None if getattr(self, 'name', None) is None else self.name,
            z=self.z[mask],
            id=None if getattr(self, 'id', None) is None else self.id,
            shell_id=self.shell_id[mask],
            radius=self.radius[mask]
        )
        if getattr(self, 'batch', None) is not None:
            batch = self.batch[mask]
            # Ensure that the batch is contiguous
            batch = torch.unique(batch, return_inverse=True)[1]
            mol.batch = batch
        return mol
    
    def _get_atom_num(
        self, 
        from_attr: str = 'x',
        unique_atom_nums: Optional[torch.Tensor] = None
    ):
        assert from_attr in ['x', 'z']
        if from_attr == 'x':
            assert unique_atom_nums is not None
            return unique_atom_nums[torch.argmax(self.x, dim=1)]
        else:
            return self.z

    @property
    def num_atoms(self):
        return self.num_nodes
    
    @property
    def num_bonds(self):
        return self.num_edges


class RadiusTransform:
    def __init__(self, mid_radius: float = MID_RADIUS):
        self.mid_radius = mid_radius
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.log(r / self.mid_radius)
    
    def inverse(self, r: torch.Tensor) -> torch.Tensor:
        return self.mid_radius * torch.exp(r)


@dataclass
class SphMol:
    x: torch.Tensor
    v: torch.Tensor
    r: torch.Tensor
    b: torch.Tensor
    
    def __post_init__(self):
        assert self.x.shape[0] == self.v.shape[0] == self.r.shape[0]
        assert self.v.norm(dim=-1).allclose(torch.ones_like(self.v))
        assert (self.r > 0).all()

    @classmethod
    def from_cartesian(
        cls,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        mid_radius: float = MID_RADIUS
    ) -> 'SphMol':
        r = pos.norm(dim=-1, keepdim=True)
        v = pos / r
        r = RadiusTransform(mid_radius).forward(r)
        return cls(x, v, r, batch)

    @classmethod
    def from_mol(cls, mol: Mol, mid_radius: float = MID_RADIUS) -> 'SphMol':
        return cls.from_cartesian(mol.x, mol.pos, mol.batch, mid_radius)
    
    def to_cartesian(self, mid_radius: float = MID_RADIUS) -> torch.Tensor:
        r = RadiusTransform(mid_radius).inverse(self.r)
        pos = self.v * r
        return pos
    
    def to_mol(self, mid_radius: float = MID_RADIUS):
        r = RadiusTransform(mid_radius).inverse(self.r)
        pos = self.v * r
        return Mol(x=self.x, pos=pos, batch=self.b)
    
    def get_prior(self, num_atom_types: int) -> 'SphMol':
        x0 = torch.randint_like(self.x, high=num_atom_types)
        v0 = torch.randn_like(self.v)
        r0 = torch.randn_like(self.r)
        return SphMol(x0, v0, r0, self.b)