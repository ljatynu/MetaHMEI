import torch
from transformer.model.transformer import TransformerBlock


class Pretrain_Model(torch.nn.Module):
    def __init__(self, emb_dim):
        super(Pretrain_Model, self).__init__()
        self.emb_dim = emb_dim
        self.pretrain_linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                                   torch.nn.Linear(2 * emb_dim, 4 * emb_dim), torch.nn.ReLU(),
                                                   torch.nn.Linear(4 * emb_dim, emb_dim), torch.nn.ReLU())

        # self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim),torch.nn.ReLU())
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, 4 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(4 * emb_dim, emb_dim), torch.nn.ReLU())

        task_num = 5  # Depending on the number of meta-training tasks
        self.pred_linear = torch.nn.Linear(self.emb_dim, task_num)

        self.transformer = TransformerBlock(300, None, 2, 0)

    def forward(self, bert_input):
        x = self.transformer.forward(bert_input.to(torch.float32), None)
        x = torch.sum(x, dim=2)
        x = self.mlp(x)
        x = self.pretrain_linear(x)
        return self.pred_linear(x)
