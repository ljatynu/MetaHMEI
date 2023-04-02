import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding
from .word2vec_embedding import Word2vecEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.5, word2vec_model = None, vocab = None):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.word2vec_embedding = Word2vecEmbedding(word2vec_model=word2vec_model,embedding_size=embed_size,vocab=vocab)
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=embed_size)
        self.segment = SegmentEmbedding(embed_size=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        # print(sequence.shape)
        # print(self.token(sequence).shape)
        # print(self.position(sequence).shape)
        # torch.Size([256, 30])
        # torch.Size([256, 30, 512])
        # torch.Size([1, 30, 512])
        # x = self.word2vec_embedding(sequence) + self.position(sequence) + self.segment(segment_label)
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
