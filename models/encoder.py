import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoder
from .multihead_attention import MultiHeadAttention

'''
Encoder flow
embedings+pe->norm->MHA->res->norm->ffn->res->out
'''

class EncoderBlock(nn.Module):
    def __init__(self,embed_dim, num_heads, ff_dim):
        super().__init__()
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mha=MultiHeadAttention(embed_dim,num_heads)
        self.ffn=nn.Sequential(nn.Linear(in_features=embed_dim, out_features=ff_dim),
                               nn.ReLU(),
                               nn.Linear(in_features=ff_dim, out_features=embed_dim),
        )

    def forward(self,x,mask=None):
        x=x+self.mha(self.norm1(x),mask)
        x=x+self.ffn(self.norm2(x))

        return x

class Encoder(nn.Module):
    """
    This is the class responsible for encoding using embeddings,
    positional encoder, multi head attention, normalization and residual addition.
    """
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.token_embeddings=nn.Embedding(vocab_size,embed_dim)
        self.pe=PositionalEncoder(embed_dim, max_len)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

    def forward(self,input_ids,mask=None):
        x=self.token_embeddings(input_ids)
        x=x+self.pe(input_ids.size(1))
        x=self.dropout(x)

        for block in self.blocks:
            x = block(x,mask=mask)

        return x
