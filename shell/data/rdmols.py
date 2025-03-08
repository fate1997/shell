import os
import os.path as osp
from dataclasses import dataclass
from typing import List, Literal

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import download_url, extract_zip
from tqdm import tqdm

from shell.utils.constants import HAR2EV, KCALMOL2EV
from shell.utils.decorator import register


RDMOLS_REGISTRY = {}
def get_rdmols(
    name: Literal['qm9'], 
    *args,
    **kwargs    
) -> 'RDMols':
    if name not in RDMOLS_REGISTRY:
        raise ValueError(f'Unknown rdmols: {name}')
    return RDMOLS_REGISTRY[name].load(*args, **kwargs)


@dataclass
class RDMols:
    rdmols: List[Chem.Mol]
    
    names: List[str] = None
    labels: torch.Tensor = None
    label_columns: List[str] = None
    
    def __post_init__(self):
        if self.labels is not None:
            assert len(self.rdmols) == len(self.labels)
            if self.label_columns is not None:
                assert self.labels.shape[1] == len(self.label_columns)
        if self.names is not None:
            assert len(self.rdmols) == len(self.names)
        
        if self.names is None:
            self.names = [None] * len(self.rdmols)
        if self.labels is None:
            self.labels = [None] * len(self.rdmols)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({len(self.rdmols)})'
    
    @staticmethod
    def files_exist(files: List[str]) -> bool:
        return len(files) != 0 and all([osp.exists(f) for f in files])
    
    @property
    def unique_atom_nums(self) -> List[int]:
        return list(set([atom.GetAtomicNum() \
                        for mol in self.rdmols \
                        for atom in mol.GetAtoms()]))


@register(name='qm9', registry=RDMOLS_REGISTRY)
class QM9RDMols(RDMols):
    RAW_URL = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    SKIP_URL = 'https://ndownloader.figshare.com/files/3195404'
    
    @classmethod
    def load(cls, raw_dir: str='dataset') -> 'QM9RDMols':
        # 0. Define paths
        raw_path = osp.join(raw_dir, 'gdb9.sdf')
        label_path = osp.join(raw_dir, 'gdb9.sdf.csv')
        skip_path = osp.join(raw_dir, 'uncharacterized.txt')
        
        # 1. Download raw data and skip file
        if not cls.files_exist([raw_path, label_path, skip_path]):
            zip_path = download_url(cls.RAW_URL, raw_dir)
            extract_zip(zip_path, raw_dir)
            os.unlink(zip_path)
            skip_prev_path = download_url(cls.SKIP_URL, raw_dir)
            os.rename(skip_prev_path, skip_path)
        
        # 2. Load skip indices
        with open(skip_path) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]
        
        # 3. load labels
        conversion = torch.tensor([
            1, 1, HAR2EV, HAR2EV, HAR2EV, 1, HAR2EV, HAR2EV, HAR2EV, HAR2EV, 
            HAR2EV, 1, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1, 1, 1
        ])
        with open(label_path) as f:
            label_df = pd.read_csv(f)
            columns = label_df.columns[1:]
            columns = columns[3:].tolist() + columns[:3].tolist()
            y = label_df[columns].values
            y = torch.FloatTensor(y) * conversion.view(1, -1)
        
        # 4. Extract rdmols and labels
        suppl = Chem.SDMolSupplier(raw_path, removeHs=False, sanitize=True)
        rdmols = []
        labels = []
        names = []
        for i, mol in enumerate(tqdm(suppl, desc='Loading QM9 rdmols')):
            if i in skip or mol is None:
                continue
            rdmols.append(mol)
            labels.append(y[i].unsqueeze(0))
            names.append(mol.GetProp('_Name'))
        
        return cls(rdmols, names, torch.cat(labels), columns)


@register(name='geom', registry=RDMOLS_REGISTRY)
class GeomRDMols(RDMols):
    def load(self, raw_dir: str='dataset') -> 'GeomRDMols':
        pass