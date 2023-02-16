from torch import nn, optim
import torch

class Myloss(nn.Module):
    def __init__(self, alpha):
        super(Myloss, self).__init__()
        self.alpha = alpha

    def forward(self, labeled_index, unlabeled_index, target, predict):
        # loss = nn.MSELoss(reduction='none')
        loss = nn.BCEWithLogitsLoss()
        return loss(predict[labeled_index].float(), target[labeled_index].float())
        # loss_sum = loss(predict.float(), target.float())
        # return (1-self.alpha)*loss_sum[labeled_index].sum()+self.alpha*loss_sum[unlabeled_index].sum()
