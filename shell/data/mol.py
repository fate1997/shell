import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from molSimplify.Classes.ligand import ligand_breakdown
from molSimplify.Classes.mol3D import mol3D
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from shell.analysis.bond import OBBondBuilder
from shell.data.writer import get_sdf_str, get_xyz_str
from shell.utils.constants import BOND_ORDER_MAP
from shell.utils.geometry import remove_partial_mean_with_mask
from shell.utils.settings import MID_RADIUS
from shell.utils.visualizer import visualize_mol


class TMC(Data):
    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        desc: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        z: Optional[torch.Tensor] = None,
        ligand_group: Optional[torch.Tensor] = None,
        ligand_mask: Optional[torch.Tensor] = None,
        atom_num: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super(TMC, self).__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        
        self.desc = desc 
        self.name = name 
        if y is not None and y.dim() == 1:
            self.y = y.unsqueeze(0)
        self.z = z
        self.ligand_group = ligand_group
        self.ligand_mask = ligand_mask
        self.atom_num = atom_num
    
    @classmethod
    def from_metal(
        cls, 
        metal: int, 
        num_nodes: int,
        unique_atom_nums: torch.Tensor,
        add_batch: bool=False,
        y: Optional[torch.Tensor]=None
    ) -> 'TMC':
        x = torch.zeros(
            size=(num_nodes, unique_atom_nums.size(0)), 
            dtype=torch.float
        )
        metal_idx = torch.where(unique_atom_nums == metal)[0]
        x[0, metal_idx] = 1.0
        pos = torch.zeros(size=(num_nodes, 3), dtype=torch.float)
        z = torch.zeros(size=(num_nodes,), dtype=torch.long)
        z[0] = metal
        ligand_mask = torch.ones(size=(num_nodes, 1), dtype=torch.float)
        ligand_mask[0] = 0.0
        if add_batch:
            batch = torch.zeros(num_nodes, dtype=torch.long)
        else:
            batch = None
        data = cls(x=x, pos=pos, z=z, ligand_mask=ligand_mask, y=y)
        data.batch = batch
        return data

    def write(self, path: str=None, fmt: str=None) -> str:
        if path is not None and fmt is None:
            fmt = path.split('.')[-1]
        if fmt == 'sdf':
            if self.edge_index is None:
                edge_index, orders = OBBondBuilder()(self.atom_num, self.pos)
                self.edge_index = edge_index
                self.edge_attr = orders
            content = get_sdf_str(
                self.atom_num, 
                self.pos, 
                self.edge_index, 
                self.edge_attr
            )
        elif fmt == 'xyz':
            content = get_xyz_str(self.atom_num, self.pos)
        else:
            raise ValueError(f'Unsupported file extension: {fmt}')
        
        if path is not None:
            with open(path, 'w') as f:
                f.write(content)
        return content
    
    def remove_hydrogen(self) -> 'TMC':
        notH_mask = self.z != 1
        notH_ids = torch.where(notH_mask)[0]
        if self.edge_index is not None:
            edge_index, edge_attr = subgraph(
                notH_ids, self.edge_index, self.edge_attr, relabel_nodes=True
            )
        else:
            edge_index, edge_attr = None, None

        pos = self.pos[notH_mask]
        z = self.z[notH_mask]
        x = self.x[notH_mask]
        if getattr(self, 'ligand_mask', None) is not None:
            ligand_mask = self.ligand_mask[notH_mask]
        else:
            ligand_mask = None
        if self.ligand_group is not None:
            ligand_group = self.ligand_group[notH_mask]
        else:
            ligand_group = None
        if getattr(self, 'atom_num', None) is not None:
            atom_num = self.atom_num[notH_mask]
        else:
            atom_num = None
        desc = getattr(self, 'desc', None)
        name = getattr(self, 'name', None)
        return TMC(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            pos=pos, 
            z=z, 
            desc=desc, 
            name=name, 
            y=self.y, 
            ligand_group=ligand_group, 
            ligand_mask=ligand_mask,
            atom_num=atom_num
        )
    
    def to_mol3d(self) -> mol3D:
        if (self.atom_num == 0).any():
            return None
        xyz_str = get_xyz_str(self.atom_num, self.pos)
        mol = mol3D()
        mol.readfromstring(xyz_str)
        return mol
    
    def to_rdmol(self, based_on: str = 'xyz') -> Chem.Mol:
        if based_on == 'xyz':
            xyz_str = get_xyz_str(self.atom_num, self.pos)
            mol = Chem.MolFromXYZBlock(xyz_str)
        elif based_on == 'sdf':
            if self.edge_index is None:
                edge_index, orders = OBBondBuilder()(self.atom_num, self.pos)
                self.edge_index = edge_index
                self.edge_attr = orders
            sdf_str = get_sdf_str(
                self.atom_num, self.pos, self.edge_index, self.edge_attr
            )
            mol = Chem.MolFromMolBlock(sdf_str, removeHs=False)
        return mol
    
    def extract_ligands(self) -> List[Chem.Mol]:
        mol3d = self.to_mol3d()
        if mol3d is None:
            return [None]
        liglist, ligdents, ligcons = ligand_breakdown(mol3d)
        lig_rdmols = []
        for lig_ids in liglist:
            ligand = self.extract_ligand(torch.tensor(lig_ids))
            lig_rdmols.append(ligand)
        return lig_rdmols
    
    def extract_ligand(self, ligand_ids: torch.Tensor) -> Chem.Mol:
        atom_num = self.atom_num[ligand_ids]
        if (atom_num == 0).any():
            return None
        pos = self.pos[ligand_ids]
        try:
            bond_index, bond_order = OBBondBuilder()(atom_num, pos)
        except:
            return None

        mol = Chem.RWMol()
        for num in atom_num:
            a = Chem.Atom(int(num.item()))
            mol.AddAtom(a)

        # add bonds to rdkit molecule
        src, dst = bond_index
        for bond_type, src_idx, dst_idx in zip(bond_order, src, dst):
            if src_idx >= dst_idx:
                continue
            src_idx = int(src_idx.item())
            dst_idx = int(dst_idx.item())
            mol.AddBond(src_idx, dst_idx, BOND_ORDER_MAP[int(bond_type.item())])

        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
        except:
            return None

        # Set coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            x, y, z = self.pos[i]
            x, y, z = float(x), float(y), float(z)
            conf.SetAtomPosition(i, Point3D(x,y,z))
        mol.AddConformer(conf)
        return mol

    def reform(self, ligand_size: int, add_batch: bool=True) -> 'TMC':
        new_data = {}
        for attr in self.alive_attrs:
            if attr == 'y':
                new_data[attr] = getattr(self, attr)
                continue
            value = getattr(self, attr)
            if not isinstance(value, torch.Tensor):
                continue
            context_info = value[(self.ligand_mask == 0.0).squeeze(-1)]
        
            if attr in ['ligand_mask', 'z', 'atom_num']:
                ligand_info = torch.ones(
                    size=(ligand_size,), 
                    dtype=value.dtype, 
                    device=value.device
                )
                if attr == 'ligand_mask':
                    ligand_info = ligand_info.unsqueeze(-1)
            elif attr in ['x', 'pos', 'ligand_group']:
                ligand_info = torch.zeros(
                    size=(ligand_size, value.size(1)), 
                    dtype=value.dtype, 
                    device=value.device
                )
            new_data[attr] = torch.cat([context_info, ligand_info], dim=0)
        data = TMC(**new_data)
        if add_batch:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        
        data.pos = data.pos - data.pos[0]
        return data
    
    def update_atom_num(
        self, 
        from_attr: str = 'x',
        unique_atom_nums: Optional[torch.Tensor] = None
    ):
        if getattr(self, 'atom_num', None) is not None:
            atom_num = self.atom_num
        assert from_attr in ['x', 'z', 'xz']
        if from_attr == 'x':
            assert unique_atom_nums is not None
            atom_num = unique_atom_nums[torch.argmax(self.x, dim=1)]
        elif from_attr == 'xz':
            assert unique_atom_nums is not None
            atom_num = unique_atom_nums[torch.argmax(self.x, dim=1)]
            atom_num[0] = self.z[0]
        else:
            atom_num = self.z
        self.atom_num = atom_num

    @property
    def alive_attrs(self):
        alive_attr = []
        all_attr = [
            'x', 
            'edge_index', 
            'edge_attr', 
            'y', 
            'pos',   
            'desc', 
            'name', 
            'z', 
            'ligand_group', 
            'metal_idx', 
            'ligand_mask',
            'atom_num'
        ]
        for attr in all_attr:
            if getattr(self, attr, None) is not None:
                alive_attr.append(attr)
        return alive_attr
    
    def show(self):
        visualize_mol(self.atom_num, self.pos)

    @property
    def num_atoms(self):
        return self.num_nodes
    
    @property
    def num_bonds(self):
        return self.num_edges


class RadiusTransform:
    def __init__(
        self, 
        mid_radius: float = MID_RADIUS,
        scale: float = 5.0
    ):
        self.mid_radius = mid_radius
        self.scale = scale
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.log(r / self.mid_radius) * self.scale
    
    def inverse(self, r: torch.Tensor) -> torch.Tensor:
        return self.mid_radius * torch.exp(r / self.scale)
    
    def deriv_coef(self, r_real: torch.Tensor) -> torch.Tensor:
        return self.scale / r_real


@dataclass
class SphTMC:
    x: torch.Tensor
    v: torch.Tensor
    r: torch.Tensor
    b: torch.Tensor
    
    def __post_init__(self):
        assert self.x.shape[0] == self.v.shape[0] == self.r.shape[0]
        v_norm = self.v.norm(dim=-1)
        if not v_norm.allclose(torch.ones_like(v_norm)):
            self.v = self.v / v_norm.unsqueeze(-1)

    @classmethod
    def from_cartesian(
        cls,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        mid_radius: float = MID_RADIUS
    ) -> 'SphTMC':
        rel_pos = pos - pos[0]
        r = rel_pos.norm(dim=-1, keepdim=True)
        v = rel_pos / r
        r = RadiusTransform(mid_radius).forward(r)
        return cls(x, v, r, batch)

    @classmethod
    def from_tmc(cls, tmc: TMC, mid_radius: float = MID_RADIUS) -> 'SphTMC':
        return cls.from_cartesian(tmc.x, tmc.pos, tmc.batch, mid_radius)
    
    @classmethod
    def from_prior(
        cls, 
        num_nodes: torch.Tensor, 
        num_atom_types: int,
        device: str='cuda'
    ) -> 'SphTMC':
        total_nodes = num_nodes.sum()
        x = torch.randint(num_atom_types, (total_nodes, ), device=device)
        x = F.one_hot(x, num_classes=num_atom_types).float()
        v = torch.randn(total_nodes, 3, device=device)
        v = v / v.norm(dim=-1, keepdim=True)
        r = torch.randn(total_nodes, 1, device=device)
        b = torch.repeat_interleave(torch.arange(len(num_nodes), device=device), num_nodes)
        return cls(x, v, r, b)

    
    def get_cartesian(self, mid_radius: float = MID_RADIUS) -> torch.Tensor:
        r = RadiusTransform(mid_radius).inverse(self.r)
        pos = self.v * r
        return pos
    
    def to_tmc(self, mid_radius: float = MID_RADIUS):
        r = RadiusTransform(mid_radius).inverse(self.r)
        pos = self.v * r
        return TMC(x=self.x, pos=pos, batch=self.b)
    
    def get_prior(self) -> 'SphTMC':
        num_atom_types = self.x.shape[1]
        x0 = torch.randint_like(self.x, high=num_atom_types)
        v0 = torch.randn_like(self.v)
        v0 = v0 - v0[0]
        v0 = v0 / v0.norm(dim=-1, keepdim=True)
        r0 = torch.randn_like(self.r)
        return SphTMC(x0, v0, r0, self.b)
    
    def __len__(self):
        return self.x.shape[0]