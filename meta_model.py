from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.pardir)
import torch
import torch.nn as nn
from samples import sample_datasets, sample_test_datasets
from model import HIA_predict
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from predict_dataset import PredictDataset
from sklearn.metrics import roc_auc_score


from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

class Meta_model(nn.Module):
    def __init__(self, args):
        super(Meta_model,self).__init__()

        self.dataset = args.dataset
        self.num_tasks = args.num_tasks
        self.num_train_tasks = args.num_train_tasks
        self.num_test_tasks = args.num_test_tasks
        self.m_support = args.m_support
        self.k_query = args.k_query

        self.emb_dim = args.emb_dim

        self.device = args.device


        self.batch_size = args.batch_size

        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.criterion = nn.BCEWithLogitsLoss()
        
        self.model = HIA_predict(args.emb_dim, args.dataset)

            
        model_param_group = []

        # for name, param in self.graph_model.gnn.named_parameters(prefix='', recurse=True):
        #     print('参数名字是:', name, '参数形状是:', param.shape)

        model_param_group.append({"params": filter(lambda p: p.requires_grad, self.model.parameters())})
        
        self.optimizer = optim.Adam(model_param_group, lr=args.meta_lr, weight_decay=args.decay)

    def update_params(self, loss, update_lr):
        grads = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, self.model.parameters()))
        return parameters_to_vector(grads), parameters_to_vector(filter(lambda p: p.requires_grad, self.model.parameters())) - parameters_to_vector(grads) * update_lr

    def forward(self, epoch):
        support_loaders = []
        query_loaders = []

        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        self.model.train()

        for task in range(self.num_train_tasks):
        # for task in tasks_list:
            data = pd.read_csv("dataset/HME/"+self.dataset+"/"+str(task+1)+"/data.csv")
            smiles = []
            label = []
            for i in data['smiles']:
                smiles.append(i)

            for i in data['label']:
                label.append(i)


            s_dataset = PredictDataset(smiles, label, 70)
            q_dataset = PredictDataset(smiles, label, 70)
            support_dataset, query_dataset = sample_datasets(s_dataset, q_dataset, self.dataset, task, self.m_support, self.k_query)
            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            support_loaders.append(support_loader)
            query_loaders.append(query_loader)

        for k in range(0, self.update_step):
            # print(self.fi)
            old_params = parameters_to_vector(filter(lambda p: p.requires_grad, self.model.parameters()))

            losses_q = torch.tensor([0.0]).to(device)

            for task in range(self.num_train_tasks):

                losses_s = torch.tensor([0.0]).to(device)

                for step, batch in enumerate(tqdm(support_loaders[task], desc="Iteration")):
                    # batch = batch.to(device)

                    pred, node_emb = self.model(batch['vec'].to(device))
                    y = batch['label'].to(device).view(pred.shape).to(torch.float64)
                    loss = torch.sum(self.criterion(pred.double(), y)) /pred.size()[0]

                    losses_s += loss
                
                new_grad, new_params = self.update_params(losses_s, update_lr = self.update_lr)


                vector_to_parameters(new_params, filter(lambda p: p.requires_grad, self.model.parameters()))

                this_loss_q = torch.tensor([0.0]).to(device)
                for step, batch in enumerate(tqdm(query_loaders[task], desc="Iteration")):

                    pred, node_emb = self.model(batch['vec'].to(device))
                    y = batch['label'].to(device).view(pred.shape).to(torch.float64)

                    loss_q = torch.sum(self.criterion(pred.double(), y))/pred.size()[0]

                    this_loss_q += loss_q

                if task == 0:
                    losses_q = this_loss_q
                else:
                    losses_q = torch.cat((losses_q, this_loss_q), 0)

                vector_to_parameters(old_params, filter(lambda p: p.requires_grad, self.model.parameters()))

            losses_q = torch.sum(losses_q)
            loss_q = losses_q / self.num_train_tasks       
            self.optimizer.zero_grad()
            loss_q.backward()
            self.optimizer.step()
        
        return []

    def test(self, support_grads):
        accs = []
        old_params = parameters_to_vector(filter(lambda p: p.requires_grad, self.model.parameters()))
        for task in range(self.num_test_tasks):
            print(self.num_tasks-task)

            data = pd.read_csv("dataset/HME/"+self.dataset+"/"+str(self.num_tasks-task)+"/data.csv")
            smiles = []
            label = []
            for i in data['smiles']:
                smiles.append(i)

            for i in data['label']:
                label.append(i)

            s_dataset = PredictDataset(smiles, label, 70)
            q_dataset = PredictDataset(smiles, label, 70)
            support_dataset, query_dataset = sample_test_datasets(s_dataset, q_dataset, self.dataset, self.num_tasks-task-1, self.m_support, self.k_query)

            support_loader = DataLoader(support_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)
            query_loader = DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers = 1)

            device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")

            self.model.eval()

            for k in range(0, self.update_step_test):
                loss = torch.tensor([0.0]).to(device)
                for step, batch in enumerate(tqdm(support_loader, desc="Iteration")):

                    pred, node_emb = self.model(batch['vec'].to(device))
                    y = batch['label'].to(device).view(pred.shape).to(torch.float64)

                    loss += torch.sum(self.criterion(pred.double(), y))/pred.size()[0]

                    print(loss)

                new_grad, new_params = self.update_params(loss, update_lr = self.update_lr)

                vector_to_parameters(new_params, filter(lambda p: p.requires_grad, self.model.parameters()))
                

            y_true = []
            y_scores = []
            for step, batch in enumerate(tqdm(query_loader, desc="Iteration")):
                pred, node_emb = self.model(batch['vec'].to(device))
                pred = F.sigmoid(pred)
                y_scores.append(pred)
                y_true.append(batch['label'].to(device).view(pred.shape))
                

            y_true = torch.cat(y_true, dim = 0).cpu().detach().numpy()
            y_scores = torch.cat(y_scores, dim = 0).cpu().detach().numpy()
           
            roc_list = []
            roc_list.append(roc_auc_score(y_true, y_scores))
            acc = sum(roc_list)/len(roc_list)
            accs.append(acc)

            vector_to_parameters(old_params, filter(lambda p: p.requires_grad, self.model.parameters()))

        return accs
    
    def predict(self, vec):
        device = torch.device("cuda:" + str(self.device)) if torch.cuda.is_available() else torch.device("cpu")
        pred, node_emb = self.model(vec.to(device),self.dataset)
        pred = F.sigmoid(pred)
        return pred, node_emb