import torch
from encoder import Encoder

# Dummy config
vocab_size = 100
max_len = 32
embed_dim = 64
num_heads = 8
ff_dim = 128
num_layers = 2
batch_size = 4
seq_len = 16

# Initialize encoder
encoder = Encoder(
    vocab_size=vocab_size,
    max_len=max_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_layers=num_layers
)

# Dummy input (token IDs)
input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # (B, T)

# Optional: create mask (1s for non-pad, 0s for pad)
mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

# Forward pass
output = encoder(input_ids, mask=mask)  # (B, T, D)

print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {output.shape}")
