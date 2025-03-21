import os
import os.path as osp
from copy import deepcopy
from typing import List, Tuple, Union

import pandas as pd
import torch
from molSimplify.Classes.ligand import ligand_breakdown
from molSimplify.Classes.mol3D import mol3D
from rdkit import Chem, RDLogger
from torch.nn import functional as F
from torch_geometric.data import Dataset, download_url, extract_gz
from tqdm import tqdm

from shell.data.featurizer import ComposeFeaturizer
from shell.data.mol import TMC
from shell.utils.constants import ELEMENT2NUM
from shell.utils.geometry import move_atom_to_top

RDLogger.DisableLog('rdApp.*')
TMQM_URL = {
    'data': (
        'https://github.com/uiocompcat/tmQM/blob/master/tmQM/tmQM_X1.xyz.gz?raw=true',
        'https://github.com/uiocompcat/tmQM/blob/master/tmQM/tmQM_X2.xyz.gz?raw=true',
        'https://github.com/uiocompcat/tmQM/blob/master/tmQM/tmQM_X3.xyz.gz?raw=true'
    ),
    'label': 'https://github.com/uiocompcat/tmQM/blob/master/tmQM/tmQM_y.csv?raw=true'
}

class TMCDataset(Dataset):
    
    def __init__(
        self,
        root: str='dataset',
        feature_names: List[str]=None,
        processed_path: str='',
        force_reload: bool=False,
        remove_hydrogens: bool=True,
        add_smiles: bool=True,
        n_samples: int=-1,
        metal_type: List[str]=None,
        ligatom_type: List[str]=None,
        bonded_oct: bool=False,
        max_num_atoms: int=-1,
        consider_metal_type: bool=False
    ):
        super().__init__()
        self.root = root
        self.feature_names = feature_names
        self.remove_hydrogens = remove_hydrogens
        self.add_smiles = add_smiles
        self.n_samples = n_samples
        self.metal_type = metal_type
        self.ligatom_type = ligatom_type
        self.bonded_oct = bonded_oct
        self.max_num_atoms = max_num_atoms
        self.consider_metal_type = consider_metal_type
        
        if osp.exists(processed_path) and not force_reload:
            dataset = torch.load(processed_path)
            self.data_list = dataset['data_list']
            self.unique_atom_nums = dataset['unique_atom_nums']
            self.label_cols = dataset['label_cols']
        else:
            xyz_list, y_list = self._prepare_raw()
            data_list, unique_atom_nums = self._prepare_data(xyz_list, y_list)
            self.data_list: List[TMC] = data_list
            self.unique_atom_nums = torch.Tensor(unique_atom_nums)
            if processed_path:
                os.makedirs(osp.dirname(processed_path), exist_ok=True)
                torch.save({
                    'data_list': self.data_list,
                    'unique_atom_nums': self.unique_atom_nums,
                    'label_cols': self.label_cols
                }, processed_path)
    
    @classmethod
    def from_processed(cls, processed_path: str):
        return cls(processed_path=processed_path, force_reload=False)

    def _prepare_raw(self) -> Tuple[List[str], List[torch.Tensor]]:
        root_dir = osp.abspath(self.root)
        raw_dir = osp.join(root_dir, 'tmqm', 'raw')
        file_names = ['tmQM_X1.xyz', 'tmQM_X2.xyz', 'tmQM_X3.xyz', 'tmQM_y.csv']
        raw_paths = [osp.join(raw_dir, name) for name in file_names]
        
        # 1. Download and extract raw data (xyz and csv files)
        if not self.files_exist(raw_paths):
            for url, raw_path in zip(TMQM_URL['data'], raw_paths[:3]):
                zip_path = download_url(url, raw_dir)
                extract_gz(zip_path, raw_dir)
                os.unlink(zip_path)
            download_url(TMQM_URL['label'], raw_dir)
        
        # 2. Split xyz files to XYZBlock and extract CSD codes
        xyz_list = []
        csd_codes = []
        for raw_path in raw_paths:
            with open(raw_path) as f:
                xyzs = f.read().split('\n\n')[:-1]
                for xyz in xyzs:
                    csd_code = xyz.split('\n')[1][11:17]
                    csd_codes.append(csd_code)
                    xyz_list.append(xyz)
        
        # 3. Extract labels from csv file
        df = pd.read_csv(raw_paths[-1], sep=';')
        df = df.drop(columns=['CSD_years'])
        if 'SMILES' in df.columns:
            df = df.drop(columns=['SMILES'])
        self.label_cols = df.columns[1:].tolist()
        df.set_index('CSD_code', inplace=True)
        y_list = df.loc[csd_codes].values.tolist()

        return xyz_list, y_list

    def _prepare_data(
        self, 
        xyz_list: List[str], 
        y_list: List[torch.Tensor]
    ) -> Tuple[List[TMC], List[int]]:
        n_samples = len(xyz_list) if self.n_samples == -1 else self.n_samples
        sample_ids = list(range(len(xyz_list)))[:n_samples]

        skip_registry = {
            'none_rdmol': 0,
            'metal_count': 0,
            'metal_type': 0,
            'ligatom_type': 0,
            'floating_components': 0
        }
        unique_atom_nums = set()
        rdmols, ligand_groups, names, ys = [], [], [], []
        for i in tqdm(sample_ids, desc='Loading TMCs'):
            # 1. Read xyz string to rdmol and mol3d
            xyz = xyz_list[i]
            mol = Chem.MolFromXYZBlock(xyz)
            if mol is None:
                skip_registry['none_rdmol'] += 1
                continue
            mol3d = mol3D()
            mol3d.readfromstring(xyz)
            metal = mol3d.findMetal(True)
            
            if len(metal) != 1:
                skip_registry['metal_count'] += 1
                continue
            if self.metal_type is not None:
                if mol.GetAtomWithIdx(metal[0]).GetSymbol() not in self.metal_type:
                    skip_registry['metal_type'] += 1
                    continue
            new_xyz = move_atom_to_top(xyz, metal[0])
            mol = Chem.MolFromXYZBlock(new_xyz)
            mol3d = mol3D()
            mol3d.readfromstring(new_xyz)
            
            # 2. Extract ligand groups
            liglist, _, _ = ligand_breakdown(
                mol3d, transition_metals_only=True, BondedOct=self.bonded_oct
            )
            n = mol.GetNumAtoms()
            ligand_group = torch.zeros((n, len(liglist)), dtype=torch.long)
            uniqe_atom_symbol = set()
            lig_ids = []
            for j, lig in enumerate(liglist):
                ligand_group[lig, j] = 1
                symbols = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in lig]
                uniqe_atom_symbol.update(symbols)
                lig_ids.extend(lig)
            if set(range(1, n)).difference(set(lig_ids)):
                skip_registry['floating_components'] += 1
                continue
            unique_atom_num = list(map(ELEMENT2NUM.get, uniqe_atom_symbol))

            # 3. Whether to consider metal type for atomic number one-hot encoding
            if self.consider_metal_type:
                unique_atom_num += [mol.GetAtomWithIdx(0).GetAtomicNum()]
            if self.ligatom_type is not None:
                if not uniqe_atom_symbol.issubset(self.ligatom_type):
                    skip_registry['ligatom_type'] += 1
                    continue
            
            # 4. Update
            ys.append(y_list[i])
            rdmols.append(mol)
            ligand_groups.append(ligand_group)
            names.append(xyz.split('\n')[1])
            unique_atom_nums.update(unique_atom_num)
        print(f'Skip registry: {skip_registry}')
        
        # 5. Featurize TMCs
        unique_atom_nums = list(sorted(unique_atom_nums))
        print(f'Unique atom numbers: {unique_atom_nums}')
        config = {'x': {'unique_atom_nums': unique_atom_nums}}
        featurizer = ComposeFeaturizer(self.feature_names, config)
        data_list = []
        bar = tqdm(total=len(rdmols), desc='Featurizing TMCs')
        max_num_ligands = max([l.size(1) for l in ligand_groups])
        for mol, ligand_group, name, y in zip(rdmols, ligand_groups, names, ys):
            mol_dict = featurizer(mol)
            mol_dict['pos'] = mol_dict['pos'] - mol_dict['pos'][0]
            mol_dict['name'] = name
            mol_dict['y'] = torch.tensor(y, dtype=torch.float)
            pad_size = max_num_ligands - ligand_group.size(1)
            mol_dict['ligand_group'] = F.pad(ligand_group, (0, pad_size))
            tmc = TMC(**mol_dict)
            if self.remove_hydrogens:
                tmc = tmc.remove_hydrogen()
            if self.max_num_atoms == -1 or tmc.num_nodes <= self.max_num_atoms:
                data_list.append(tmc)
            bar.update(1)
        bar.close()
        print(f'Number of TMCs: {len(data_list)}')
        return data_list, unique_atom_nums
    
    @staticmethod
    def files_exist(files: List[str]) -> bool:
        return len(files) != 0 and all([osp.exists(f) for f in files])
    
    def get(self, idx: int) -> TMC:
        return self.data_list[idx]
    
    def len(self) -> int:
        return len(self.data_list)
    
    def to_allligand_task(self):
        data_list = []
        for data in self.data_list:
            ligand_mask = torch.ones((data.num_nodes, 1))
            ligand_mask[0] = 0
            train_data = deepcopy(data)
            train_data.ligand_mask = ligand_mask
            train_data.pos = train_data.pos - train_data.pos[0]
            data_list.append(train_data)
        self.data_list = data_list
        
    def split_and_save(
        self, 
        n_train: Union[int, float], 
        n_val: Union[int, float], 
        save_dir: str
    ):
        dataset = self.shuffle(self)
        if isinstance(n_train, float):
            assert n_train > 0 and n_train < 1
            n_train = int(n_train * len(dataset))
        if isinstance(n_val, float):
            assert n_val > 0 and n_val < 1
            n_val = int(n_val * len(dataset))
        assert n_train + n_val < len(dataset)
        train_data = dataset[:n_train]
        val_data = dataset[n_train:n_train+n_val]
        test_data = dataset[n_train+n_val:]
        os.makedirs(save_dir, exist_ok=True)
        basic_dict = {
            'unique_atom_nums': self.unique_atom_nums,
            'label_cols': self.label_cols
        }
        for data, name in zip([train_data, val_data, test_data], 
                              ['train', 'val', 'test']):
            torch.save({
                'data_list': data,
                **basic_dict
            }, osp.join(save_dir, f'{name}.pt'))