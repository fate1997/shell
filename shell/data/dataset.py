import os
import os.path as osp
from typing import List, Literal, Tuple

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from shell.data.featurizer import ComposeFeaturizer
from shell.data.mol import Mol
from shell.data.rdmols import get_rdmols

RDLogger.DisableLog('rdApp.*')


class MolDataset(Dataset):
    
    def __init__(
        self,
        name: Literal['qm9'],
        root: str='dataset',
        feature_names: List[str]=None,
        processed_path: str='',
        force_reload: bool=False,
        remove_hydrogens: bool=True,
        add_smiles: bool=True,
        n_samples: int=-1,
        add_xmask: bool=False
    ):
        super().__init__()
        
        self.name = name
        self.feature_names = feature_names
        
        if osp.exists(processed_path) and not force_reload:
            data = torch.load(processed_path)
            self.data_list = data['data_list']
            self.unique_atom_nums = data['unique_atom_nums']
            self.label_cols = data['label_cols']
        else:
            raw_dir = osp.join(root, name, 'raw')
            rdmols = get_rdmols(name, raw_dir=raw_dir)
            data_list = []
            self.unique_atom_nums = rdmols.unique_atom_nums
            if add_xmask:
                self.unique_atom_nums = self.unique_atom_nums + [0]
            if n_samples == -1:
                sample_ids = range(len(rdmols.rdmols))
            else:
                sample_ids = range(n_samples)
            start_idx = 0
            for j, i in enumerate(tqdm(sample_ids, desc='Featurizing')):
                rdmol = rdmols.rdmols[i]
                if remove_hydrogens:
                    rdmol = Chem.RemoveHs(rdmol, sanitize=False)
                if 'x' in self.feature_names:
                    config = {'x': {'unique_atom_nums': self.unique_atom_nums}}
                else:
                    config = {}
                mol_len = rdmol.GetNumAtoms()
                mol_dict = ComposeFeaturizer(self.feature_names, config)(rdmol)
                mol_dict['name'] = getattr(rdmols, 'names', [None]*mol_len)[i]
                mol_dict['y'] = getattr(rdmols, 'labels', [None]*mol_len)[i]
                mol_dict['id'] = j
                start_idx += mol_len
                if add_smiles:
                    mol_dict['smiles'] = Chem.MolToSmiles(rdmol)
                data_list.append(Mol(**mol_dict))
            self.data_list = data_list
            self.label_cols = rdmols.label_columns
            
            if processed_path:
                os.makedirs(osp.dirname(processed_path), exist_ok=True)
                torch.save({
                    'data_list': self.data_list,
                    'unique_atom_nums': self.unique_atom_nums,
                    'label_cols': self.label_cols
                }, processed_path)
    
    def get(self, idx: int) -> Mol:
        return self.data_list[idx]
    
    def len(self) -> int:
        return len(self.data_list)

    def get_loaders(
        self,
        batch_size: int,
        n_train: int,
        n_val: int,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dataset = self.shuffle(self)
        assert n_train + n_val < len(dataset)
        train_loader = DataLoader(
            dataset[:n_train], batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            dataset[n_train:n_train+n_val], batch_size, num_workers=num_workers
        )
        test_loader = DataLoader(
            dataset[n_train+n_val:], batch_size, num_workers=num_workers
        )
        return train_loader, val_loader, test_loader
    
    def sample_batch(
        self,
        batch_size: int = 64,
    ) -> Batch:
        loader = DataLoader(self, batch_size, shuffle=True)
        return next(iter(loader))