from torch.utils.data import Dataset
import torch
from mol2vec.features import mol2alt_sentence
from rdkit import Chem
import gensim
word2vec_model = gensim.models.word2vec.Word2Vec.load('dataset/model_300dim.pkl')
import numpy as np
class PredictDataset(Dataset):
    def __init__(self, corpus, label, seq_len, encoding="utf-8", on_memory=True):
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus = corpus
        self.corpus_lines = len(self.corpus)
        self.label = label
        self.encoding = encoding

        self.lines = None

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # print(item)

        s = self.corpus[item]
        vec = []
        padding = np.zeros(300)
        if s == None: #解决HDAC数据集nan的问题
            vec = [padding]
            vec = np.vstack([vec, [padding]])
        else:
            for i in s.split(" "):
                if i in word2vec_model.wv.index2word:
                    if vec.__len__() == 0:
                        vec = [word2vec_model.wv.get_vector(i)]
                    else:
                        vec = np.vstack([vec, [word2vec_model.wv.get_vector(i)]])

        if vec.shape[0] <65:
            for i in range(65-vec.shape[0]):
                vec = np.vstack([vec, [padding]])

        output = {"vec": vec,
                  "label": self.label[item]}

        return {key: torch.tensor(value) for key, value in output.items()}

    def load(self, index):
        self.corpus = [self.transfor_mol2vec(self.corpus[i]) for i in index]
        self.label = [self.label[i] for i in index]
        self.corpus_lines = len(self.corpus)
        return self



    def transfor_mol2vec(self, i):
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
