from collections import Counter
from typing import Tuple

from rdkit import Chem
import numpy as np
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from shell.analysis.mol_sample import MolSample, MolSampleList
from shell.utils.constants import ALLOWED_BONDS


def analyze(
    mols: MolSampleList,
):
    n_atoms = 0
    n_stable_atoms = 0
    n_stable_molecules = 0
    n_valid_molecules = 0
    n_molecules = len(mols)
    
    for mol in tqdm(mols, desc='Analyzing molecules'):
        n_atoms += mol.num_atoms
        n_stable_atoms_this_mol, mol_stable = check_stability(mol)
        n_valid, frag_fracs, num_components = check_validity(mol)
        n_stable_atoms += n_stable_atoms_this_mol
        n_stable_molecules += int(mol_stable)
        n_valid_molecules += int(n_valid)
    frac_atoms_stable = n_stable_atoms / n_atoms
    frac_mols_stable_valence = n_stable_molecules / n_molecules
    frac_mols_valid = n_valid_molecules / n_molecules

    return {
        'atom_stability': frac_atoms_stable,
        'mol_stability': frac_mols_stable_valence,
        'mol_validity': frac_mols_valid
    }


def check_validity(mol: MolSample):
    valid = False
    error_message = Counter()
    rdmol = mol.get_rdmol()
    frag_fracs = 0.0
    num_components = 1
    if rdmol is not None:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
            num_components = len(mol_frags)
            if len(mol_frags) > 1:
                error_message[4] += 1
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            largest_mol_n_atoms = largest_mol.GetNumAtoms()
            frag_fracs = largest_mol_n_atoms / mol.num_atoms
            valid = True
            error_message[-1] += 1
        except Chem.rdchem.AtomValenceException:
            error_message[1] += 1
            # print("Valence error in GetmolFrags")
        except Chem.rdchem.KekulizeException:
            error_message[2] += 1
            # print("Can't kekulize molecule")
        except Chem.rdchem.AtomKekulizeException or ValueError:
            error_message[3] += 1
    return valid, frag_fracs, num_components


def check_stability(mol: MolSample) -> Tuple[int, bool]:
    adj_matrix = to_dense_adj(mol.bond_index, edge_attr=mol.bond_order)
    nr_bonds = adj_matrix.sum(dim=1).tolist()[0]

    n_stable_atoms = 0
    mol_stable = True
    for atom_num_i, nr_bonds_i in zip(mol.atom_num, nr_bonds):
        possible_bonds = ALLOWED_BONDS.get(atom_num_i.item())
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        n_stable_atoms += int(is_stable)
        if not is_stable:
            mol_stable = False
    return n_stable_atoms, mol_stable