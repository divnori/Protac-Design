'''
Filters data by max_n_nodes. Randomly splits into 80% test, 15% test, 5% valid

python split_filter_protac.py --orig_smi path/to/file.smi --new_smi path/to/file.smi --threshold num
'''
import random

import argparse
from utils import load_molecules
from rdkit import Chem

# define the argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 add_help=False)

# define two potential arguments to use when drawing SMILES from a file
parser.add_argument("--orig_smi",
                    type=str,
                    default="data/gdb13_1K/train.smi",
                    help="path to original SMILES file.")
parser.add_argument("--new_smi",
                    type=str,
                    default="data/gdb13_1K/train.smi",
                    help="path to new SMILES file.")
parser.add_argument("--threshold",
                    type=int,
                    default=100,
                    help="All molecules with >n atoms will be filtered out.")
args = parser.parse_args()

def filter(threshold, original_smi_file, filtered_smi_file):
    #filter from original.smi to new smi file filtered by size
    #threshold num must be an integer less than 139

    molecules = load_molecules(path=original_smi_file)

    with open(filtered_smi_file, 'w+') as f:
        for mol in molecules:
            n_atoms = mol.GetNumAtoms()
            if n_atoms <= threshold:
                atoms = mol.GetAtoms()
                atom_types = set(atoms)
                atom_symbols = [Chem.Atom(atom).GetSymbol() for atom in atom_types]
                if 'I' not in atom_symbols and 'P' not in atom_symbols:
                    smi = Chem.MolToSmiles(mol)
                    f.write("{}\n".format(smi))

    

def split(filtered_smi_file):

    orig_smiles = []
    with open(filtered_smi_file, 'r') as f:
        for line in f.readlines():
            words = line.split()
            orig_smiles.append(words[0])

    test_sz = int(.15*len(orig_smiles))
    valid_sz = int(.05*len(orig_smiles))
    train_sz = len(orig_smiles) - test_sz - valid_sz

    random.shuffle(orig_smiles)
    train_smiles = orig_smiles[:train_sz]
    test_smiles = orig_smiles[train_sz:len(orig_smiles)-valid_sz]
    valid_smiles = orig_smiles[len(orig_smiles)-valid_sz:]
    
    with open('data/pre-training/protac_db_subset_70/train.smi', 'w+') as f:
        for smi in train_smiles:
            f.write("{}\n".format(smi))
    with open('data/pre-training/protac_db_subset_70/test.smi', 'w+') as f:
        for smi in test_smiles:
            f.write("{}\n".format(smi))
    with open('data/pre-training/protac_db_subset_70/valid.smi', 'w+') as f:
        for smi in valid_smiles:
            f.write("{}\n".format(smi))

if __name__ == "__main__":
    filter(threshold=args.threshold, original_smi_file = args.orig_smi, filtered_smi_file=args.new_smi)
    split(filtered_smi_file = args.new_smi)
    print("Done.", flush=True)