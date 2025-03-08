from typing import Optional

import torch
from torch_geometric.data import Data


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
        **kwargs
    ):
        super(Mol, self).__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        
        self.desc = desc 
        self.smiles = smiles
        self.name = name
        self.z = z
        self.y = y.unsqueeze(0)
        self.id = id
        self.shell_id = shell_id
    
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