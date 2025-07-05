import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

def create_padding_mask(seq, pad_token=0):
    # seq: (B, T)
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)  # → (B, 1, 1, T)

def create_look_ahead_mask(size):
    # size: sequence length
    return torch.tril(torch.ones(size, size)).bool()  # (T, T)

def combine_masks(pad_mask, look_ahead_mask):
    # pad_mask: (B, 1, 1, T)
    # look_ahead_mask: (T, T)
    return pad_mask & look_ahead_mask.unsqueeze(0).unsqueeze(1)  # (B, 1, T, T)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_len, embed_dim=512, num_heads=8, ff_dim=2048, num_layers=6):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers)
        self.decoder = Decoder(tgt_vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers)


    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, pad_token=0):
        if src_mask is None:
            src_mask = create_padding_mask(src_ids, pad_token)
        if tgt_mask is None:
            tgt_pad_mask = create_padding_mask(tgt_ids, pad_token)
            look_ahead_mask = create_look_ahead_mask(tgt_ids.size(1)).to(tgt_ids.device)
            tgt_mask = combine_masks(tgt_pad_mask, look_ahead_mask)
        
        enc_output = self.encoder(src_ids, mask=src_mask)
        logits = self.decoder(tgt_ids, enc_output, tgt_mask=tgt_mask, src_mask=src_mask)
        return logits