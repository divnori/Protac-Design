"""
Contains function to score a given SMILES string as a high quality, average quality, or low quality
PROTAC based on protac scoring boosted tree model
"""
import rdkit
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdchem
import numpy as np
import pickle
from parameters import constants
import requests as r

def get_morgan_fp(m):
  fingerprint = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
  array = np.zeros((0,), dtype=np.int8)
  DataStructs.ConvertToNumpyArray(fingerprint, array)
  return array

known_active_degraders = ['CC1=C(C2=CC=C(CNC(=O)[C@@H]3[C@@H](F)[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCOCCOCCNC(=O)C[C@@H]3N=C(C4=CC=C(Cl)C=C4)C4=C(SC(C)=C4C)N4C(C)=NN=C34)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCOCCOCCNC(=O)C3=CC4=C(C=C3CS(C)(=O)=O)C3=CN(C)C(=O)C5=C3C(=C[NH]5)CN4C3=NC=C(F)C=C3F)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCCCCCCCCNC(=O)C3=CC4=C(C=C3CS(C)(=O)=O)C3=CN(C)C(=O)C5=C3C(=C[NH]5)CN4C3=NC=C(F)C=C3F)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)CCCCCCCCCCNC(=O)C3=CC4=C(C=C3CS(C)(=O)=O)C3=CN(C)C(=O)C5=C3C(=C[NH]5)CN4C3=NC=C(F)C=C3F)C(C)(C)C)C(OC3CCNCC3)=C2)SC=N1',
    'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCC3=CC(N)=CC(OCCNC(=O)C4=CC5=C(C=C4CS(C)(=O)=O)C4=CN(C)C(=O)C6=C4C(=C[NH]6)CN5C4=NC=C(F)C=C4F)=C3)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCC3=CC(O)=CC(OCCNC(=O)C4=CC5=C(C=C4CS(C)(=O)=O)C4=CN(C)C(=O)C6=C4C(=C[NH]6)CN5C4=NC=C(F)C=C4F)=C3)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCC#CC#CCNC(=O)C[C@@H]3N=C(C4=CC=C(Cl)C=C4)C4=C(SC(C)=C4C)N4C(C)=NN=C34)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C([C@H](C)NC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCCOCCOC(=O)C[C@@H]3N=C(C4=CC=C(Cl)C=C4)C4=C(SC(C)=C4C)N4C(C)=NN=C34)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C([C@H](C)NC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCCOCCNC(=O)C[C@@H]3N=C(C4=CC=C(Cl)C=C4)C4=C(SC(C)=C4C)N4C(C)=NN=C34)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C([C@H](C)NC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCCOCCNC(=O)C[C@@H]3N=C(C4=CC=C(Cl)C=C4)C4=C(SC(C)=C4C)N4C(C)=NN=C34)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C([C@H](C)NC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCCOCCNC(=O)C[C@@H]3N=C(C4=CC=C(Cl)C=C4)C4=C(SC(C)=C4C)N4C(C)=NN=C34)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCOCCOCCNC(=O)C[C@@H]3N=C(C4=CC=C(Cl)C=C4)C4=C(SC(C)=C4C)N4C(C)=NN=C34)C(C)(C)C)C=C2)SC=N1',
    'CC1=C(C2=CC=C(CNC(=O)[C@@H]3C[C@@H](O)CN3C(=O)[C@@H](NC(=O)COCCOCCOCCOCCNC(=O)C[C@@H]3N=C(C4=CC=C(Cl)C=C4)C4=C(SC(C)=C4C)N4C(C)=NN=C34)C(C)(C)C)C=C2)SC=N1',
    'CC(=O)N1C2=CC=C(C3=CC=C(C(=O)NCCOCCOCCOCC(=O)N[C@H](C(=O)N4C[C@H](O)C[C@H]4C(=O)NCC4=CC=C(C5=C(C)N=CS5)C=C4)C(C)(C)C)C=C3)C=C2[C@H](NC2=CC=C(Cl)C=C2)C[C@@H]1C',
    'CC(=O)N1C2=CC=C(C3=CC=C(C(=O)NCCOCCOCCOCCOCC(=O)N[C@H](C(=O)N4C[C@H](O)C[C@H]4C(=O)NCC4=CC=C(C5=C(C)N=CS5)C=C4)C(C)(C)C)C=C3)C=C2[C@H](NC2=CC=C(Cl)C=C2)C[C@@H]1C']

mols = [Chem.MolFromSmiles(smi) for smi in known_active_degraders]
known_fps = [get_morgan_fp(m) for m in mols]


def predictProteinDegradation(mol : rdkit.Chem.Mol, constants : namedtuple, cellType='SRD15', receptor='O60885', e3Ligase='VHL'):
    '''
     Returns a 0 or 1 based on a protac's molecule protein degradation potential
     1 -> degrades
     0 -> does not degrade
    '''
    try:
        generated_fp = get_morgan_fp(mol)

        euclid_dists = np.sum((generated_fp - known_fps)**2, axis=1)/1024
        max_dist = np.max(euclid_dists)
        if max_dist > 0.12:
            return 1
        else:
            return 0
    except:
        return 0

    # try:
    #     scoring_model = constants.qsar_models["protac_qsar_model"]
    #     features = constants.activity_model_features
    #     Chem.SanitizeMol(mol)

    #     ngrams_array = np.zeros((1,7841), dtype=np.int8)
    #     # baseUrl="http://www.uniprot.org/uniprot/"
    #     # currentUrl=baseUrl+receptor+".fasta"
    #     # response = r.post(currentUrl)
    #     # cData=''.join(response.text)
    #     # i = cData.index('\n')+1
    #     # seq = cData[i:].strip().lower()
    #     seq = "MSAESGPGTRLRNLPVMGDGLETSQMSTTQAQAQPQPANAASTNPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEETEIMIVQAKGRGRGRKETGTAKPGVSTVPNTTQASTPPQTQTPQPNPPPVQATPHPFPAVTPDLIVQTPVMTVVPPQPLQTPPPVPPQPQPPPAPAPQPVQSHPPIIAATPQPVKTKKGVKRKADTTTPTTIDPIHEPPSLPPEPKTTKLGQRRESSRPVKPPKKDVPDSQQHPAPEKSSKVSEQLKCCSGILKEMFAKKHAAYAWPFYKPVDVEALGLHDYCDIIKHPMDMSTIKSKLEAREYRDAQEFGADVRLMFSNCYKYNPPDHEVVAMARKLQDVFEMRFAKMPDEPEEPVVAVSSPAVPPPTKVVAPPSSSDSSSDSSSDSDSSTDDSEEERAQRLAELQEQLKAVHEQLAALSQPQQNKPKKKEKDKKEKKKEKHKRKEEVEENKKSKAKEPPPKKTKKNNSSNSNVSKKEPAPMKSKPPPTYESEEEDKCKPMSYEEKRQLSLDINKLPGEKLGRVVHIIQSREPSLKNSNPDEIEIDFETLKPSTLRELERYVTSCLRKKRKPQAEKVDVIAGSSKMKGFSSSESESSSESSSSDSEDSETEMAPKSKKKGHPGREQKKHHHHHHQQMQQAPAPVPQQPPPPPQQPPPPPPPQQQQQPPPPPPPPSMPQQAAPAMKSSPPPFIATQVPVLEPQLPGSVFDPIGHFTQPILHLPQPELPPHLPQPPEHSTPPHLNQHAVVSPPALHNALPQQPSRPSNRAAALPPKPARPPAVSPALTQTPLLPQPPMAQPPQVLLEDEEPPAPPLTSMQMQLYLQQLQKVQPPTPLLPSVKVQSQPPPPLPPPPHPSVQQQLQQQPPPPPPPQPQPPPQQQHQPPPRPVHLQPMQFSTHIQQPPPPQGQQPPHPPPGQQPPPPQPAKPQQVIQHHHSPRHHKSDPYSTGHLREAPSPLMIHSPQMSQFQSLTHQSPPQQNVQPKKQELRAASVVQPQPLVVVKEEKIHSPIIRSEPFSPSLRPEPPKHPESIKAPVHLPQRPEMKPVDVGRPVIRPPEQNAPPPGAPDKDKQKQEPKTPVAPKKDLKIKNMGSWASLVQKHPTTPSSTAKSSSDSFEQFRRAAREKEEREKALKAQAEHAEKEKERLRQERMRSREDEDALEQARRAHEEARRRQEQQQQQRQEQQQQQQQQAAAVAAAATPQAQSSQPQSMLDQQRELARKREQERRRREAMAATIDMNFQSDLLSIFEENLF".lower()
    #     ngrams = features[1237:]
    #     for i in range(len(ngrams)):
    #         n = seq.count(ngrams[i])
    #         ngrams_array[0][i] = n

    #     fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    #     fp_array = np.zeros((0,), dtype=np.int8)
    #     DataStructs.ConvertToNumpyArray(fingerprint, fp_array)

    #     ct_ind = features.index("ct_"+cellType)
    #     e3_ind = features.index("e3_"+e3Ligase)

    #     input = list(0 for i in range(5)) + list(fp_array) + list(0 for i in range(207))+ list(ngrams_array[0])
    #     input[ct_ind] = 1
    #     input[e3_ind] = 1

    #     output = scoring_model.predict([input])
    #     smi = Chem.MolToSmiles(mol)
    #     letters = smi.replace("[=()-]","")
    #     if output[0][0]-output[0][1] < 0 and len(letters)>=30:
    #         return 1
    #     else:
    #         return 0
    # except Exception as e:
    #     print('EXCEPTION IN PREDICT PROTEIN DEGRADATION',flush=True)
    #     print(e, flush=True)
    #     return 0
