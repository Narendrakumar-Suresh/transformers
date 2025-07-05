import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super().__init__()
        assert embed_dim%num_heads==0

        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dims = embed_dim// num_heads

        self.QKV=nn.Linear(embed_dim,3*embed_dim)
        self.out_proj=nn.Linear(embed_dim,embed_dim)

    def forward(self, x, mask=None):
        B,T,D = x.size()
        QKV=self.QKV(x)
        Q,K,V=QKV.chunk(3,dim=-1)

        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.head_dims).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        attn_score=torch.matmul(Q,K.transpose(-2,-1))/(self.head_dims**0.5)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_weight=F.softmax(attn_score,dim=-1)
        attn_output=torch.matmul(attn_weight,V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(attn_output)