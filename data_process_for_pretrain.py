import os
import pandas as pd
from numpy import nan


def process(dataset):
    path = "HME/" + dataset  # 文件夹目录
    folders = os.listdir(path)  # 得到文件夹下的所有文件夹名称
    folders.sort()
    columns = []

    for folder in folders:
        columns.append(folder)

    location = {}
    cont_list = []
    columns = ["smiles"]
    train_task = 0
    if dataset == "KDM":
        train_task = 5
    elif dataset == "HDAC":
        train_task = 7
    elif dataset == "PMT":
        train_task = 6
    elif dataset == "HAT":
        train_task = 3

    smiles_index = {}
    for folder in range(1, train_task + 1):
        columns.append(str(folder))
        folder_path = path + "/" + str(folder)
        files = os.listdir(folder_path)  # 得到文件夹下的所有文件名称
        for file in files:  # 遍历文件夹
            # print(folder_path + "/" + file)
            f = pd.read_csv(folder_path + "/" + file, header=0)
            for i in range(len(f.iloc[:, 0])):
                if not f.iloc[i, 1] == nan and type(f.iloc[i, 0]) == type("a") and len(f.iloc[i, 0].strip()) != 0:
                    if location.__contains__(f.iloc[i, 0]):
                        cont_list[location.get(f.iloc[i, 0])][str(folder)] = f.iloc[i, 1]
                    else:
                        location[f.iloc[i, 0]] = len(cont_list)
                        cont_list.append({"smiles": f.iloc[i, 0], str(folder): f.iloc[i, 1]})

    df = pd.DataFrame(cont_list, columns=columns)
    df.to_csv("../dataset/data_for_pretrain_" + dataset + ".csv", index=False)


process("KDM")
