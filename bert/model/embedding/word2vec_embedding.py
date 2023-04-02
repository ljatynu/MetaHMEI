import torch.nn as nn
import torch


class Word2vecEmbedding(nn.Module):
    def __init__(self, word2vec_model, embedding_size,vocab = None):
        super().__init__()
        self.word2vec_model = word2vec_model
        self.embedding_size = embedding_size
        self.vocab = vocab

    def forward(self, x):
        res = None
        for s in x:
            temp_res = None
            for word in s.tolist():
                if not temp_res == None:
                    if self.vocab.dic.get(word) not in self.word2vec_model:
                        temp_res = torch.cat((temp_res, torch.zeros(self.embedding_size).reshape(1,512)),0)
                    else:
                        temp_res = torch.cat((temp_res, torch.tensor(self.word2vec_model[self.vocab.dic.get(word)]).reshape(1,512)),0)
                else:
                    if self.vocab.dic.get(word) not in self.word2vec_model:
                        temp_res = torch.zeros(self.embedding_size).reshape(1,512)
                    else:
                        temp_res = torch.tensor(self.word2vec_model[self.vocab.dic.get(word)]).reshape(1,512)

            # print(temp_res.shape)
            if not res == None:
                res = torch.cat((res, torch.tensor(temp_res.reshape(1, 30, 512))),0)
            else:
                res = temp_res.reshape(1, 30, 512)

        # print(res.shape)
        return res.cuda()

