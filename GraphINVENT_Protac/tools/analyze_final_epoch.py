import math
import random
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

smi_file = "/home/gridsan/dnori/GraphINVENT/output_MOSES_subset/example-job-name/job_0/generation/aggregated_generation/epoch_100.smi"
train_smi = "/home/gridsan/dnori/GraphINVENT/data/pre-training/MOSES_subset/train.smi"
#smi_file = "/home/gridsan/dnori/GraphINVENT/output_protac_db_subset_60/example-job-name/job_0/generation/aggregated_generation/epoch_GEN80.smi"

# # load molecules from file
# mols = SmilesMolSupplier(smi_file, sanitize=True, nameColumn=-1,titleLine=True)

# n_samples = 40
# mols_list = [mol for mol in mols]
# mols_sampled = random.sample(mols_list, n_samples)  # sample 100 random molecules to visualize

# mols_per_row = int(math.sqrt(n_samples))            # make a square grid

# png_filename=smi_file[:-3] + "png"  # name of PNG file to create
# labels=list(range(n_samples))       # label structures with a number

# # draw the molecules (creates a PIL image)
# img = MolsToGridImage(mols=mols_sampled,
#                       molsPerRow=mols_per_row,
#                       legends=[str(i) for i in labels])

# img.save(png_filename)

#calculate regeneration percentage

with open(smi_file, 'r') as f1:
    gen_smi = f1.readlines()

with open (train_smi, 'r') as f2:
    tr_smi = f2.readlines()

#canon all smi in tr_smi:
for i in range(len(tr_smi)):
    mol = Chem.MolFromSmiles(tr_smi[i])
    canon_smi = Chem.MolToSmiles(mol)
    if canon_smi!=tr_smi[i]:
        tr_smi[i]

total = len(gen_smi)
num = 0
for s in gen_smi:
    mol = Chem.MolFromSmiles(s)
    canon_s = Chem.MolToSmiles(mol)
    if canon_s not in tr_smi:
        num+=1

print(f"percentage of new molecules: {num/total}")
