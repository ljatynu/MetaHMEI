import torch
import random

def obtain_distr_list(dataset):
    if dataset == "HDAC":
        return [[609, 2966], [474, 1906], [306, 1028], [439, 1016], [247, 983], [1118, 613], [1083, 316], [124, 312],
                [117, 298], [303, 233], [214, 202], [214, 197], [144, 162], [371, 158]]
    elif dataset == "HAT":
        return [[7551, 6285], [142, 285], [156, 132], [62, 166], [183, 51], [51, 47]]
    elif dataset == "PMT":
        return [[156, 453], [47, 364], [338, 250], [57, 159], [112, 107], [164, 71], [10, 86], [42, 95], [42, 65],
                [13, 38], [13, 35], [67, 13]]
    elif dataset == "KDM":
        return [[44976, 6619], [35455, 3976], [620, 830], [227, 563], [177, 406], [149, 51], [20, 25], [40, 69], [34, 75], [47, 25]]

def sample_datasets(s_data, q_data, dataset, task, m_support, n_query):
    distri_list = obtain_distr_list(dataset)

    support_list = random.sample(range(0,distri_list[task][0]), m_support)
    support_list += random.sample(range(distri_list[task][0],distri_list[task][0]+distri_list[task][1]), m_support)
    random.shuffle(support_list)
    print("训练集")
    print(support_list)
    l = [i for i in range(0, distri_list[task][0]+distri_list[task][1]) if i not in support_list]
    query_list = random.sample(l, n_query)

    support_dataset = s_data.load(torch.tensor(support_list))
    query_dataset = q_data.load(torch.tensor(query_list))

    return support_dataset, query_dataset

def sample_test_datasets(s_data, q_data, dataset, task, m_support, n_query):
    import random
    random.seed(1)
    distri_list = obtain_distr_list(dataset)
    support_list = random.sample(range(0,distri_list[task][0]), m_support,)
    support_list += random.sample(range(distri_list[task][0],distri_list[task][0]+distri_list[task][1]), m_support)
    random.shuffle(support_list)
    print("测试集")
    print(support_list)
    l = [i for i in range(0, distri_list[task][0]+distri_list[task][1]) if i not in support_list]

    support_dataset = s_data.load(torch.tensor(support_list))
    query_dataset = q_data.load(torch.tensor(l))

    return support_dataset, query_dataset

def sample_select_datasets(s_data):
    distri_list = [149, 51]
    random.seed(2)
    support_list = random.sample(range(0,distri_list[0]), 5)
    support_list += random.sample(range(distri_list[0],distri_list[0]+distri_list[1]), 5)
    random.shuffle(support_list)
    support_dataset = s_data.load(torch.tensor(support_list))


    return support_dataset


