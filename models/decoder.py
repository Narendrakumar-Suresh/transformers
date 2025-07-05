import torch
import torch.nn as nn
import torch.nn.functional as F
from  .positional_encoding import PositionalEncoder
from .multihead_attention import MultiHeadAttention
"""
embedding + PE
→ norm → masked MHA → res
→ norm → cross-attention → res
→ norm → FFN → res
→ linear → softmax
"""

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attention=MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=ff_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=ff_dim, out_features=embed_dim),
        )

    def forward(self,x,enc_output,tgt_mask=None,src_mask=None):
        x = x + self.self_attention(self.norm1(x), mask=tgt_mask)
        x=x+self.cross_attention(self.norm2(x),mask=src_mask,key=enc_output, value=enc_output)

        x=x+self.ffn(self.norm3(x))
        return x

class Decoder(nn.Module):
    """
    This is the class responsible for decoding using embeddings,
    positional encoder, multi head attention, normalization and residual addition.
    """
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers,):
        super().__init__()
        self.token_embeddings=nn.Embedding(vocab_size,embed_dim)
        self.pe=PositionalEncoder(embed_dim, max_len)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(embed_dim, vocab_size)


    def forward(self,input_ids,enc_output,tgt_mask,src_mask):
        x=self.token_embeddings(input_ids)
        x=x+self.pe(input_ids.size(1))
        x=self.dropout(x)

        for block in self.blocks:
            x = block(x, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)
        x = self.output_linear(x)
        return x