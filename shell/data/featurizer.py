from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from rdkit import Chem
from scipy.sparse import coo_matrix

from shell.utils.constants import DEGREE_CHOICE
from shell.utils.settings import QM9_SHELL_RADIUS
from shell.utils.shell import assign_shell_id

FEATURIZER_REGISTRY: Dict[str, 'Featurizer'] = {}


def register_cls(name: str):
    def decorator(cls):
        FEATURIZER_REGISTRY[name] = cls
        return cls
    return decorator


class Featurizer(ABC):
    _avail_features: List[str] = []
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names
        if feature_names is not None:
            self._check_features(feature_names)
    
    def _check_features(self, feature_names: List[str]):
        for feature_name in feature_names:
            assert feature_name in self._avail_features, \
                f'{feature_name} is not available in {self.__class__.__name__}'
    
    @abstractmethod
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        pass


########################################################
############# Add new featurizers below ################
########################################################

@register_cls('x')
class AtomFeaturizer(Featurizer):
    """Featurize atoms in a molecule. Default features include one-hot encoding
    of atomic numbers.
    """
    _avail_features: List[str] = ['degree', 'is_aromatic']
    def __init__(
        self,
        feature_names: List[str] = None,
        unique_atom_nums: List[int] = None,
    ):
        super().__init__(feature_names)
        self.unique_atom_nums = unique_atom_nums
    
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        atom_num = self.get_atom_num(rdmol, self.unique_atom_nums)
        if self.feature_names is None:
            return {'x': atom_num}
        
        x = []
        for atom in rdmol.GetAtoms():
            atom_features = []
            for feature_name in self.feature_names:
                atom_features.append(getattr(self, feature_name)(atom))
            x.append(torch.cat(atom_features))
        feature_exclude_atom_num = torch.stack(x, dim=0)
        return {'x': torch.cat([atom_num, feature_exclude_atom_num], dim=1)}
    
    @staticmethod
    def get_atom_num(
        rdmol: Chem.Mol,
        unique_atom_nums: List[int],
    ) -> torch.Tensor:
        atom_nums = [atom.GetAtomicNum() for atom in rdmol.GetAtoms()]
        indices = [unique_atom_nums.index(num) for num in atom_nums]
        atom_nums, indices = torch.tensor(atom_nums), torch.tensor(indices)
        onehot = torch.zeros(len(atom_nums), len(unique_atom_nums)).to(int)
        onehot.scatter_(1, indices.unsqueeze(1), 1)
        return onehot
    
    def degree(self, atom: Chem.Atom) -> torch.Tensor:
        onehot = torch.zeros(len(DEGREE_CHOICE))
        onehot[DEGREE_CHOICE.index(atom.GetTotalDegree())] = 1
        return onehot
    
    def is_aromatic(self, atom: Chem.Atom) -> torch.Tensor:
        return torch.tensor([int(atom.GetIsAromatic())])


@register_cls('edge')
class BondFeaturizer(Featurizer):
    _avail_features: List[str] = ['fully_connected_edges', 'bond']
    
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        if self.feature_names is None:
            return {}
        
        assert len(self.feature_names) == 1, \
            'Only one feature name is supported for bond featurizer'
        feature_name = self.feature_names[0]
        return getattr(self, feature_name)(rdmol)
    
    def fully_connected_edges(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        n_nodes = rdmol.GetNumAtoms()
        edges = torch.combinations(torch.arange(n_nodes), r=2).T
        reversed_edges = torch.stack([edges[1], edges[0]])
        return {'edge_index': torch.cat([edges, reversed_edges], dim=1)}
    
    def bond(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        adj = Chem.GetAdjacencyMatrix(rdmol)
        coo_adj = coo_matrix(adj)
        edge_index = torch.from_numpy(np.vstack([coo_adj.row, 
                                                 coo_adj.col])).long()
        edge_attr = []
        for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            bond = rdmol.GetBondBetweenAtoms(i, j)
            bond_type = bond.GetBondType()
            bond_type_one_hot_encoding = [
                int(bond_type == Chem.rdchem.BondType.SINGLE),
                int(bond_type == Chem.rdchem.BondType.DOUBLE),
                int(bond_type == Chem.rdchem.BondType.TRIPLE),
                int(bond_type == Chem.rdchem.BondType.AROMATIC)
            ]
            edge_attr.append(torch.tensor(bond_type_one_hot_encoding))
        edge_attr = torch.stack(edge_attr, dim=0)
        return {'edge_index': edge_index, 'edge_attr': edge_attr}


@register_cls('pos')
class PosFeaturizer(Featurizer):
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        if rdmol.GetNumConformers() == 0:
            return {'pos': None}
        pos = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        pos -= pos.mean(dim=0)
        return {'pos': pos}


@register_cls('z')
class AtomNumFeaturizer(Featurizer):
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        atom_nums = [atom.GetAtomicNum() for atom in rdmol.GetAtoms()]
        return {'z': torch.tensor(atom_nums)}


@register_cls('shell_id')
class ShellIdFeaturizer(Featurizer):
    def __call__(
        self,
        rdmol: Chem.Mol,
    ) -> Dict[str, torch.Tensor]:
        pos = rdmol.GetConformer().GetPositions()
        pos = torch.from_numpy(pos).float()
        pos -= pos.mean(dim=0)
        distance = torch.norm(pos, dim=1)
        shell_id = assign_shell_id(distance, torch.tensor(QM9_SHELL_RADIUS))
        return {'shell_id': shell_id}


########################################################
############# End of new featurizers ###################
########################################################

AVAIL_FEATURES = set(FEATURIZER_REGISTRY.keys())
for key, cls in FEATURIZER_REGISTRY.items():
    AVAIL_FEATURES.update(cls._avail_features)


class ComposeFeaturizer:
    def __init__(self, names: List[str], config: dict = None):
        invalid_names = set(names) - set(AVAIL_FEATURES)
        if invalid_names:
            raise ValueError(f'Invalid feature names: {invalid_names}')
        
        if config is None:
            config = {}
        
        featurizers = []
        for key, cls in FEATURIZER_REGISTRY.items():
            feature_names = []
            for name in names:
                if name == key:
                    featurizers.append(cls(**config.get(key, {})))
                elif name in cls._avail_features:
                    feature_names.append(name)
            if feature_names:
                featurizers.append(cls(feature_names, **config.get(key, {})))
        self.featurizers = featurizers
    
    def __call__(self, rdmol: Chem.Mol) -> Dict[str, torch.Tensor]:
        mol_dict = {}
        for featurizer in self.featurizers:
            mol_dict.update(featurizer(rdmol))
        return mol_dict