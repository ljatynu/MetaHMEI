import argparse
import torch
import numpy as np
from meta_model import Meta_model

def main(dataset,m_support):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of MetaHIA')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='input batch size for training (default: 32)') 
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dataset', type=str, default = 'sider', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_tasks', type=int, default=12, help = "# of tasks")
    parser.add_argument('--num_train_tasks', type=int, default=9, help = "# of training tasks")
    parser.add_argument('--num_test_tasks', type=int, default=3, help = "# of testing tasks")
    parser.add_argument('--m_support', type=int, default=5, help = "size of the support dataset")
    parser.add_argument('--k_query', type = int, default=256, help = "size of querry datasets")
    parser.add_argument('--meta_lr', type=float, default=0.0001)
    parser.add_argument('--update_lr', type=float, default=0.004) #0.4
    parser.add_argument('--update_step', type=int, default=5) #5
    parser.add_argument('--update_step_test', type=int, default=10) #10

    args = parser.parse_args()

    args.dataset = dataset
    args.m_support = m_support

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset == "HDAC":
        args.num_tasks = 14
        args.num_train_tasks = 7
        args.num_test_tasks = 7
    elif args.dataset == "HAT":
        args.num_tasks = 6
        args.num_train_tasks = 3
        args.num_test_tasks = 3
    elif args.dataset == "KDM":
        args.num_tasks = 10
        args.num_train_tasks = 5
        args.num_test_tasks = 5
    elif args.dataset == "PMT":
        args.num_tasks = 12
        args.num_train_tasks = 6
        args.num_test_tasks = 6
    else:
        raise ValueError("Invalid dataset name.")

    model = Meta_model(args).to(device)
    # model.to(device)

    print(args.dataset)

    best_accs = []
    best_model = []
    for epoch in range(1, args.epochs+1):
        torch.cuda.empty_cache()
        support_grads = model(epoch)

        if epoch % 1 == 0:
            accs, model_list = model.test(support_grads)

            if best_accs != []:
                for acc_num in range(len(best_accs)):
                    if best_accs[acc_num] < accs[acc_num]:
                        best_accs[acc_num] = accs[acc_num]
                        best_model[acc_num] = model_list[acc_num]
            else:
                best_accs = accs
                best_model = model_list

            fw = open("result/" + args.dataset +  "_" + str(args.m_support) + "_" + str(args.update_step) + ".txt", "a")
            fw.write("test: " + "\t")
            for i in accs:
                fw.write(str(i) + "\t")

            fw.write("best: " + "\t")
            for i in best_accs:
                fw.write(str(i) + "\t")
            fw.write("\n")
            fw.close()
    # for j in range(len(best_model)):
    #     torch.save(best_model[j], "result/" + args.dataset + "_" + str(args.num_tasks-j) + "_model")

if __name__ == "__main__":
    main("HDAC", 5)
