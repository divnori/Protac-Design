"""
Create train.csv containing summary stats on training set

To use script, run:
python graphinvent/training_stats.py
"""
import random
import csv

random.seed(10)

# def write_input_csv(params_dict : dict, filename : str="params.csv") -> None:
#     """
#     Writes job parameters/hyperparameters in `params_dict` to CSV using the specified
#     `filename`.
#     """
#     print("in write input csv")
#     dict_path = params_dict["job_dir"] + filename

#     with open(dict_path, "w") as csv_file:

#         writer = csv.writer(csv_file, delimiter=";")
#         for key, value in params_dict.items():
#             writer.writerow([key, value])

# params = {"compute_train_csv": True, "job_dir": "data/pre-training/MOSES/"}
# write_input_csv(params_dict=params, filename="input.csv")

from rdkit import Chem

from DataProcesser import DataProcesser
import util
import torch
from parameters.constants import constants

from torch.utils.tensorboard import SummaryWriter

#take a random subset of 10000 smiles
orig_smi_file_path = str('data/pre-training/protac_db_subset_50/orig.smi')
list_of_smiles = []
with open(orig_smi_file_path, 'r') as f:
    for line in f.readlines():
        words = line.split()
        list_of_smiles.append(words[0])

train_smiles = list_of_smiles[:92]
test_smiles = list_of_smiles[92:109]
valid_smiles = list_of_smiles[109:]

with open('data/pre-training/protac_db_subset_50/train.smi', 'w+') as f:
    for smi in train_smiles:
        f.write("{}\n".format(smi))
with open('data/pre-training/protac_db_subset_50/test.smi', 'w+') as f:
    for smi in test_smiles:
        f.write("{}\n".format(smi))
with open('data/pre-training/protac_db_subset_50/valid.smi', 'w+') as f:
    for smi in valid_smiles:
        f.write("{}\n".format(smi))


#convert smiles to molecular graphs
mols = [Chem.MolFromSmiles(smi) for smi in subset_of_smiles]
processor = DataProcesser(path = 'data/pre-training/MOSES/train_subset.smi', is_training_set=True, molset = mols)
graphs = [processor.get_graph(mol) for mol in mols]

processor.get_ts_properties(molecular_graphs=graphs, group_size=10000)
print(processor.ts_properties)

#writecsv (from util.py)
# writer = SummaryWriter()
# util.properties_to_csv(processor.ts_properties, 'data/pre-training/MOSES/train.csv', 'Training set', writer)