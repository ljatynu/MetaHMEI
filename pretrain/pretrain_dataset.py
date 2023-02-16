from sklearn import preprocessing
from torch.utils.data import Dataset
import torch
import pandas as pd
import gensim
word2vec_model = gensim.models.word2vec.Word2Vec.load('../dataset/model_300dim.pkl')
from mol2vec.features import mol2alt_sentence
from rdkit import Chem
import numpy as np
class PredictDataset(Dataset):
    def __init__(self, indexs, batch_size, smiles_vec, data):
        self.indexs = indexs
        self.data = data
        self.batch_size = batch_size
        self.smiles_vec = smiles_vec
        self.smiles_vec_matrix = []
        self.labels = []
        self.protein_vec = None




        for c in range(int(self.indexs/batch_size)):
            label = []
            smiles = []
            # upper = len(self.indexs) if (c+1)*batch_size > len(self.indexs) else (c+1)*batch_size
            for i in range(c*batch_size, (c+1)*batch_size):
                t = []
                for u in self.data.iloc[i, 1:]:
                    if np.isnan(u):
                        t.append(-1.0)
                    elif u == 0.0:
                        t.append(0.0)
                    else:
                        t.append(1.0)
                if label.__len__() == 0:
                    label = t
                else:
                    label = np.vstack([label, [t]])

                vec = []
                padding = np.zeros(300)
                s = self.transfor_mol2vec(self.data.iloc[i, 0])
                for c in s.split(" "):
                    if c in word2vec_model.wv.index2word:
                        if vec.__len__() == 0:
                            vec = [word2vec_model.wv.get_vector(c)]
                        else:
                            vec = np.vstack([vec, [word2vec_model.wv.get_vector(c)]])
                if vec.shape[0] < 65:
                    for i in range(65 - vec.shape[0]):
                        vec = np.vstack([vec, [padding]])

                if smiles.__len__() == 0:
                    smiles = [vec]
                else:
                    smiles.append(vec)

            self.labels.append(label)
            self.smiles_vec_matrix.append(np.array(smiles))
        self.smiles_vec_matrix = np.array(self.smiles_vec_matrix)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        output = {"smiles_vec": self.smiles_vec_matrix[item, ::]
                  , "label": self.labels[item]}

        return {key: torch.tensor(value) for key, value in output.items()}


    def transfor_mol2vec(self, i):
        sentence = mol2alt_sentence(Chem.MolFromSmiles(i), 1)[1::2]
        l = len(sentence)
        if l == 0:  # 用来处理'[Ar]'这种元素的情况 mol2alt_sentence出来为空
            sentence = mol2alt_sentence(Chem.MolFromSmiles(i), 0)
            sentence.append(sentence[0])
        if l == 1:  # 用来处理'CI'这种元素的情况 mol2alt_sentence出来为单个元素
            sentence.append(sentence[0])
        if l > 65:
            sentence = sentence[:65]
        l = len(sentence)


        return ' '.join(sentence)

