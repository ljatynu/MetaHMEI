import torch
import pandas as pd
from mol2vec.features import mol2alt_sentence
from rdkit import Chem
import gensim
import numpy as np

def transfor_mol2vec(i):
    if type(i) != type("a"):
        return None
    sentence = mol2alt_sentence(Chem.MolFromSmiles(i), 1)[1::2]
    l = len(sentence)
    if l == 0:
        sentence = mol2alt_sentence(Chem.MolFromSmiles(i), 0)
        sentence.append(sentence[0])
    if l == 1:
        sentence.append(sentence[0])
    if l > 65:
        sentence = sentence[:65]

    return ' '.join(sentence)



word2vec_model = gensim.models.word2vec.Word2Vec.load('dataset/model_300dim.pkl')
dataset = "KDM"
task = 0

data = pd.read_csv("dataset/HME/" + dataset + "/" + str(task + 1) + "/data.csv")
smiles = []
label = []
for i in data['smiles']:
    smiles.append(i)

print(smiles[0])
temp = transfor_mol2vec(smiles[0])

vec = []
for i in temp.split(" "):
    if i in word2vec_model.wv.index2word:
        if vec.__len__() == 0:
            vec = [word2vec_model.wv.get_vector(i)]
        else:
            vec = np.vstack([vec, [word2vec_model.wv.get_vector(i)]])

padding = np.zeros(300)
if vec.shape[0] < 65:
    for i in range(65 - vec.shape[0]):
        vec = np.vstack([vec, [padding]])

m = torch.load("result/KDM_7_model")
device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")
y, emb = m(torch.tensor(vec).unsqueeze(0).to(device))
print(y[0][0])