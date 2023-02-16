import torch.nn as nn

from .attention import MultiHeadedAttention
from .attention import Attention
from .utils import SublayerConnection, PositionwiseFeedForward
import torch
from torch.nn.parameter import Parameter

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        # self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.attention = Attention()
        self.weight1 = torch.nn.Linear(300, 300,bias=0)
        w1 = torch.Tensor(300, 300)
        w1 = torch.nn.init.eye_(w1)
        self.weight1.weight = Parameter(w1, requires_grad=True)

        self.weight2 = torch.nn.Linear(300, 300,bias=0)
        w2 = torch.Tensor(300, 300)
        w2 = torch.nn.init.eye_(w2)
        self.weight2.weight = Parameter(w2, requires_grad=True)

        self.weight3 = torch.nn.Linear(300, 300,bias=0)
        w3 = torch.Tensor(300, 300)
        w3 = torch.nn.init.eye_(w3)
        self.weight3.weight = Parameter(w3, requires_grad=True)
        # self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        # self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # x = self.output_sublayer(x, self.feed_forward)
        # return self.dropout(x)
        return self.attention.forward(self.weight1(x), self.weight2(x), self.weight3(x), mask=mask)