import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PositionalEncoder(nn.Module):
    def __init__(self,dim_out,max_len=1000):
        super().__init__()
        self.dim_out=dim_out
        self.n=10_000

        pos=torch.arange(max_len).float().unsqueeze(1)
        i = torch.arange(dim_out).float().unsqueeze(0)

        angle_rates = 1 / torch.pow(self.n, (2 * (i // 2)) / self.dim_out)
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        pe = angle_rads.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len]

pe = PositionalEncoder(dim_out=64)
x = pe(seq_len=100)  # (1, 100, 64)

plt.figure(figsize=(12, 6))
plt.imshow(x[0].T.numpy(), cmap='viridis', aspect='auto')
plt.xlabel("Position")
plt.ylabel("Encoding Dimension")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.savefig('PE.png')
