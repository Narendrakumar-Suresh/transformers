import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, key=None, value=None):
        if key is None: key = x
        if value is None: value = x

        B, T, _ = x.size()
        _, S, _ = key.size()

        Q = self.q_proj(x)
        K = self.k_proj(key)
        V = self.v_proj(value)

        def split_heads(t):
            return t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)  # (B, H, T, D_head)
        K = split_heads(K)  # (B, H, S, D_head)
        V = split_heads(V)  # (B, H, S, D_head)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, S)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, D_head)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(attn_output)