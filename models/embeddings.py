import torch
import torch.nn as nn
import math

# Positional embedding (for including time information)
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        # Frequencies
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        # Time steps
        ts  = torch.arange(total_time_steps, dtype=torch.float32)
        # Form the angles as products time steps * frequencies
        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        # Half for sines, alf for cosines
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            # The positional encodings are in emb
            nn.Embedding.from_pretrained(emb),
            # Linear layer
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            # Activation
            nn.SiLU(),
            # Linear layer
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)