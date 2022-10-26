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

def predictProteinDegradation(mol : rdkit.Chem.Mol, constants : namedtuple, cellType='SRD15', receptor='Q9Y616', e3Ligase='CRBN'):
    '''
     Returns a 0 or 1 based on a protac's molecule protein degradation potential
     1 -> degrades
     0 -> does not degrade
    '''
    try:
        scoring_model = constants.qsar_models["protac_qsar_model"]
        features = constants.activity_model_features
        Chem.SanitizeMol(mol)

        ngrams_array = np.zeros((1,7841), dtype=np.int8)
        # baseUrl="http://www.uniprot.org/uniprot/"
        # currentUrl=baseUrl+receptor+".fasta"
        # response = r.post(currentUrl)
        # cData=''.join(response.text)
        # i = cData.index('\n')+1
        # seq = cData[i:].strip().lower()
        seq = "MAGNCGARGALSAHTLLFDLPPALLGELCAVLDSCDGALGWRGLAERLSSSWLDVRHIEKYVDQGKSGTRELLWSWAQKNKTIGDLLQVLQEMGHRRAIHLITNYGAVLSPSEKSYQEGGFPNILFKETANVTVDNVLIPEHNEKGILLKSSISFQNIIEGTRNFHKDFLIGEGEIFEVYRVEIQNLTYAVKLFKQEKKMQCKKHWKRFLSELEVLLLFHHPNILELAAYFTETEKFCLIYPYMRNGTLFDRLQCVGDTAPLPWHIRIGILIGISKAIHYLHNVQPCSVICGSISSANILLDDQFQPKLTDFAMAHFRSHLEHQSCTINMTSSSSKHLWYMPEEYIRQGKLSIKTDVYSFGIVIMEVLTGCRVVLDDPKHIQLRDLLRELMEKRGLDSCLSFLDKKVPPCPRNFSAKLFCLAGRCAATRAKLRPSMDEVLNTLESTQASLYFAEDPPTSLKSFRCPSPLFLENVPSIPVEDDESQNNNLLPSDEGLRIDRMTQKTPFECSQSEVMFLSLDKKPESKRNEEACNMPSSSCEESWFPKYIVPSQDLRPYKVNIDPSSEAPGHSCRSRPVESSCSSKFSWDEYEQYKKE".lower()
        ngrams = features[1237:]
        for i in range(len(ngrams)):
            n = seq.count(ngrams[i])
            ngrams_array[0][i] = n

        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fp_array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, fp_array)

        ct_ind = features.index("ct_"+cellType)
        e3_ind = features.index("e3_"+e3Ligase)

        input = list(0 for i in range(5)) + list(fp_array) + list(0 for i in range(207))+ list(ngrams_array[0])
        input[ct_ind] = 1
        input[e3_ind] = 1

        output = scoring_model.predict([input])
        smi = Chem.MolToSmiles(mol)
        letters = smi.replace("[=()-]","")
        if output[0][0]-output[0][1] < 0 and len(letters)>=30:
            return 1
        else:
            return 0
    except Exception as e:
        print('EXCEPTION IN PREDICT PROTEIN DEGRADATION',flush=True)
        print(e, flush=True)
        return 0