import math
import random
import rdkit
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.rdmolfiles import SmilesMolSupplier

smi_file = "/home/gridsan/dnori/GraphINVENT/output_MOSES_subset/example-job-name/job_0/generation/step61_agent.smi"

# load molecules from file
mols = SmilesMolSupplier(smi_file, sanitize=True, nameColumn=-1,titleLine=True)

n_samples = 8
mols_list = [mol for mol in mols]
mols_sampled = random.sample(mols_list, n_samples)  # sample 100 random molecules to visualize

mols_per_row = int(math.sqrt(n_samples))            # make a square grid

png_filename=smi_file[:-3] + "png"  # name of PNG file to create
#labels=list(range(n_samples))       # label structures with a number

labels = [i for i in range(n_samples)]

# draw the molecules (creates a PIL image)
img = MolsToGridImage(mols=mols_sampled,
                      molsPerRow=mols_per_row,
                      legends=[str(i) for i in labels])

img.save(png_filename)