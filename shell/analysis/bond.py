import tempfile
from typing import Dict, Tuple

import numpy as np
import torch
from openbabel import openbabel
from rdkit import Chem
from scipy.sparse import coo_matrix

from shell.data.writer import get_xyz_str
from shell.utils.constants import (DOUBLE_BOND_DISTANCE, SINGLE_BOND_DISTANCE,
                                   TRIPLE_BOND_DISTANCE)


class DistanceBondBuilder:
    def __init__(
        self,
        margin1: float = 10,
        margin2: float = 5,
        margin3: float = 3,
    ):
        # Margin1, margin2 and margin3 have been tuned to maximize the 
        # stability of the QM9 true samples.
        self.margin1 = margin1
        self.margin2 = margin2
        self.margin3 = margin3
        
        # Get all support pairs.
        support_pairs1 = self.get_support_pairs(SINGLE_BOND_DISTANCE)
        support_pairs2 = self.get_support_pairs(DOUBLE_BOND_DISTANCE)
        support_pairs3 = self.get_support_pairs(TRIPLE_BOND_DISTANCE)
        self.support_pairs = {
            'single': support_pairs1,
            'double': support_pairs2,
            'triple': support_pairs3,
        }
    
    def __call__(
        self,
        atom_num: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = torch.cdist(pos, pos)
        edge_index = torch.zeros((2, 0), device=pos.device)
        orders = []
        for i in range(len(atom_num)):
            for j in range(i + 1, len(atom_num)):
                pair = tuple(sorted((atom_num[i].item(), atom_num[j].item())))
                order = self.infer_order(pair[0], pair[1], dist[i, j])
                if order > 0:
                    edge = torch.tensor([[i, j], [j, i]], device=pos.device)
                    edge_index = torch.cat([edge_index, edge], dim=1).int()
                    orders.extend([order, order])
        orders = torch.tensor(orders, device=pos.device)
        edge_index = edge_index.long()
        return edge_index, orders
    
    def get_support_pairs(self, bond_dict: Dict[str, Dict[str, float]]) -> set:
        support_pairs = set()
        for element1, _ in bond_dict.items():
            for element2 in _:
                support_pairs.add(tuple(sorted((element1, element2))))
        return support_pairs
    
    def infer_order(
        self,
        elem1: int,
        elem2: int,
        distance: float,
    ) -> int:
        distance *= 100
        # Check if elements are in the dictionary.
        if (elem1, elem2) in self.support_pairs['triple']:
            if distance < TRIPLE_BOND_DISTANCE[elem1][elem2] + self.margin3:
                return 3
        if (elem1, elem2) in self.support_pairs['double']:
            if distance < DOUBLE_BOND_DISTANCE[elem1][elem2] + self.margin2:
                return 2
        if (elem1, elem2) in self.support_pairs['single']:
            if distance < SINGLE_BOND_DISTANCE[elem1][elem2] + self.margin1:
                return 1
        
        return 0


class OBBondBuilder:
    def __init__(self):
        pass

    def __call__(
        self,
        atom_num: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        with tempfile.NamedTemporaryFile(suffix='.xyz') as tmp:
            tmp_file = tmp.name
            # Write xyz file
            with open(tmp_file, 'w') as f:
                f.write(get_xyz_str(atom_num, pos))

            # Convert to sdf file with openbabel
            # openbabel will add bonds
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("xyz", "sdf")     
            ob_mol = openbabel.OBMol()
            obConversion.ReadFile(ob_mol, f'{tmp_file}')
            obConversion.WriteFile(ob_mol, f'{tmp_file}.sdf')
            # Read sdf file with RDKit
            tmp_mol = Chem.SDMolSupplier(f'{tmp_file}.sdf', sanitize=False)[0]
        
        adj = Chem.GetAdjacencyMatrix(tmp_mol, useBO=True)
        coo_adj = coo_matrix(adj)
        bond_index = torch.from_numpy(np.array([coo_adj.row, coo_adj.col], dtype=np.int64))
        bond_order = torch.from_numpy(coo_adj.data)
        return bond_index, bond_order