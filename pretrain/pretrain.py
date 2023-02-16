import random
import gensim
import pandas as pd
import numpy as np
from pretrain_model import Pretrain_Model
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn, optim
from sklearn.metrics import roc_auc_score
import torch
from pretrain_dataset import PredictDataset
from Loss import Myloss

# 获取所有smiles码的特征向量表示


def main(dataset):
    epoch = 300
    batch_size = 1
    f = pd.read_csv("../dataset/data_for_pretrain_"+dataset+".csv", header=0)
    random.seed(777)
    a = [i for i in range(len(f.iloc[:, 0]))]
    random.shuffle(a)

    pretrain_Model = Pretrain_Model(300)

    train_dataset = PredictDataset(len(f.iloc[:, 0]), 500, [], f)
    #         得到的特征矩阵500 乘 300维
    # pretrain_Model.train()
    regression_crit = Myloss(0.3)
    optimizer = optim.Adam(pretrain_Model.parameters(), lr=0.0005)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    model_best = None
    best_valid_auc = 0

    best_auc = 0
    for e in range(epoch):
        losses = torch.tensor([0.0])
        label_list = []
        predict_list = []
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            pretrain_Model.train()
            labeled_index = np.where(np.array(batch['label']) != -1.0)
            unlabeled_index = np.where(np.array(batch['label']) == 0.0)
            # IMC_model.zero_grad()

            predict = pretrain_Model(batch['smiles_vec'])
            loss = regression_crit(labeled_index, unlabeled_index, batch['label'], predict)
            losses += loss
            predict_list += predict[labeled_index].tolist()
            label_list += batch['label'][labeled_index].tolist()
        auc = roc_auc_score(label_list, predict_list)
        fw = open("pretrain_model/"+dataset+"res.txt", "a")
        fw.write("auc: " + "\t")
        fw.write(str(auc) + "\t")
        fw.write("\n")
        optimizer.zero_grad()
        losses.backward()
        # print([x.grad for x in optimizer.param_groups[0]['params']])
        optimizer.step()
        print("loss:"+str(loss.item()))
        if auc > best_auc:
            model_best = pretrain_Model
            best_auc = auc
        print("best_auc:" + str(best_auc))
        torch.save(pretrain_Model, "pretrain_model/"+dataset+"model"+str(e))

        fw.close()

    torch.save(model_best, "pretrain_model/"+dataset+"_best_model")


if __name__ == "__main__":
    main("HAT")
