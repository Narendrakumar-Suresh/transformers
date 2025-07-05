import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self ,vocab_size,max_len, embed_dim=512, num_heads=8, ff_dim=2048,num_layers=6):
        super().__init__()
        self.encoder = Encoder(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers)
        self.decoder = Decoder(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src_ids)
        logits = self.decoder(tgt_ids, enc_output, tgt_mask, src_mask)
        return logits

model = Transformer(vocab_size=10000, max_len=100)
src = torch.randint(0, 10000, (2, 20))
tgt = torch.randint(0, 10000, (2, 20))

out = model(src, tgt)
print(out.shape)  # (2, 20, 10000)
