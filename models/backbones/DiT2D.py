import torch
import torch.nn as nn
from .embeddings import SinusoidalPositionEmbeddings


class PatchEmbed(nn.Module):
    """
    Patchify a spatiotemporal tensor (B, C, H, W, T) by treating each frame
    independently: reshape to (B*T, C, H, W), apply Conv2d, then restore T.

    Supports rectangular grids (grid_rows != grid_cols) as long as both are
    divisible by patch_size.
    """
    def __init__(self, grid_rows, grid_cols, patch_size, in_channels, hidden_size):
        super().__init__()

        assert grid_rows % patch_size == 0, \
            f"grid_rows ({grid_rows}) must be divisible by patch_size ({patch_size})"
        assert grid_cols % patch_size == 0, \
            f"grid_cols ({grid_cols}) must be divisible by patch_size ({patch_size})"

        self.patch_size  = patch_size
        self.h_patches   = grid_rows // patch_size
        self.w_patches   = grid_cols // patch_size
        self.num_patches = self.h_patches * self.w_patches
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, T)
        Returns:
            tokens: (B, T, N, D)   where N = num_patches per frame
        """
        B, C, H, W, T = x.shape
        x = x.permute(0, 4, 1, 2, 3)          # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)
        x = self.proj(x)                       # (B*T, D, h_p, w_p)
        x = x.flatten(2).transpose(1, 2)       # (B*T, N, D)
        x = x.reshape(B, T, self.num_patches, -1)   # (B, T, N, D)
        return x

class PatchUnEmbed(nn.Module):
    """
    Reassemble (B, T*N, C*p*p) token sequence → (B, C, H, W, T).
    FinalLayer has already projected hidden_size → C*p*p.
    This module only does the spatial + temporal reshape.
    """
    def __init__(self, grid_rows, grid_cols, patch_size, out_channels):
        super().__init__()
        self.patch_size   = patch_size
        self.h_patches    = grid_rows // patch_size
        self.w_patches    = grid_cols // patch_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """
        Args:
            x: (B, T*N, C*p*p)
            T: Total mprops sequence lenght, P+F=T
        Returns:
            (B, C, H, W, T)
        """
        B    = x.shape[0]
        p    = self.patch_size
        h, w = self.h_patches, self.w_patches
        C    = self.out_channels
        N    = h * w

        x = x.reshape(B, T, N, C*p*p)
        x = x.reshape(B * T, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)         # (B*T, C, h, p, w, p)
        x = x.reshape(B*T, C, h*p, w*p)
        x = x.reshape(B, T, C, h*p, w*p)
        x = x.permute(0, 2, 3, 4, 1)            # (B, C, H, W, T)
        return x

def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads,
                                           dropout=dropout_rate, batch_first=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden, hidden_size), nn.Dropout(dropout_rate),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x_mod = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out
        x_mod = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(x_mod)
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT2D(nn.Module):
    """
    DiT version 1.
    Diffusion Transformer for spatiotemporal macroscopic sequences.

    Drop-in replacement for the sequence UNet.
    Mirrors UNet's forward signature: forward(future, t, past) → future_hat

    Strategy: concatenate past + future along time, patchify all frames,
    run through transformer.

    Positional encoding — two separate learned embeddings that are added:

        spatial_pos_embed  (1, N, D)      — where in the grid  (same for all frames)
        temporal_pos_embed (1, T_max, D)  — when in the sequence (same for all patches)

    Together every token knows both WHERE it is spatially and WHEN it is
    temporally, giving the self-attention full spatiotemporal context.

    Token sequence length: (grid_rows/p) × (grid_cols/p) × (past + future)
    For ATC: (12/4) × (36/4) × 8 = 3 × 9 × 8 = 216 tokens
    """
    def __init__(
        self,
        input_channels:    int   = 4,
        output_channels:   int   = 4,
        grid_rows:         int   = 12,
        grid_cols:         int   = 36,
        patch_size:        int   = 4,
        hidden_size:       int   = 256,
        depth:             int   = 6,
        num_heads:         int   = 4,
        mlp_ratio:         float = 4.0,
        dropout_rate:      float = 0.1,
        time_multiple:     int   = 4,        # same name as UNet
        total_time_steps:  int   = 1000,     # same name as UNet
        condition:         str   = "Past",
        t_max:             int   = 8,        # max total frames (past+future) supported
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.condition       = condition
        self.input_channels  = input_channels
        self.output_channels = output_channels

        # time_emb_dims_exp uses the same formula as UNet
        time_emb_dims_exp = hidden_size * time_multiple

        # Diffusion timestep embedding (same as UNet)
        # this t is the diffusion step — separate from the temporal frames in the sequence below.
        self.time_embeddings = SinusoidalPositionEmbeddings(
            total_time_steps  = total_time_steps,
            time_emb_dims     = hidden_size,
            time_emb_dims_exp = time_emb_dims_exp,
        )
        # Project from time_emb_dims_exp → hidden_size for AdaLN conditioning
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dims_exp, hidden_size),
            nn.SiLU(),
        )

        # Patch embedding
        self.patch_embed = PatchEmbed(grid_rows, grid_cols, patch_size, input_channels, hidden_size)
        N = self.patch_embed.num_patches   # spatial patches per frame

        # Spatial positional encoding
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, N, hidden_size))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # Temporal positional encoding
        self.t_max = t_max
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, t_max, hidden_size))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])

        # Output
        self.final_layer = FinalLayer(hidden_size, patch_size, output_channels)
        self.unpatch     = PatchUnEmbed(grid_rows, grid_cols, patch_size, output_channels)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.view(m.weight.size(0), -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _add_positional_embeddings(self, tokens: torch.Tensor, T: int) -> torch.Tensor:
        """
        Add spatial + temporal positional encodings to token sequence.

        Args:
            tokens : (B, T, N, D)   — patch tokens, still split by frame
            T      : total mprops sequence lenght, P+F=T

        The two embeddings answer orthogonal questions:
            spatial_pos_embed  (1,  1, N, D)  → where in the grid?
            temporal_pos_embed (1,  T, 1, D)  → when in the sequence?

        Broadcasting adds them correctly across all (T, N) combinations:
            (1, 1, N, D) + (1, T, 1, D) = (1, T, N, D)
        so every token at (frame=t, patch=n) gets:
            spatial_pos_embed[n] + temporal_pos_embed[t]
        """
        assert T <= self.t_max, \
            f"T={T} exceeds t_max={self.t_max}. Increase t_max at construction."

        # spatial: (1, N, D) → (1, 1, N, D)  broadcasts over T
        spatial  = self.spatial_pos_embed.unsqueeze(1)

        # temporal: (1, t_max, D) → slice to T → (1, T, 1, D)  broadcasts over N
        temporal = self.temporal_pos_embed[:, :T, :].unsqueeze(2)

        return tokens + spatial + temporal   # (B, T, N, D)

    def forward(self, future: torch.Tensor, t: torch.Tensor, past: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            future : (B, C, H, W, F)  noisy future frames to be predicted
            t : (B,)  long            diffusion timestep in [0, total_time_steps)
            past: (B, C, H, W, P)     noisy past frames (conditioning)
        Returns:
            (B, C, H, W, F)  predicted noise (DDPM) or velocity field (FM)
        """
        # Concatenate past + future along time (mirrors UNet)
        if self.condition == "Past" and past is not None:
            past_len = past.shape[4]
            x = torch.cat([past, future], dim=4)    # (B, C, H, W, P+F)
        else:
            past_len = past.shape[4]
            x = future

        mprops_seq_len = x.shape[4]

        # Diffusion timestep conditioning → (B, D)
        c = self.time_proj(self.time_embeddings(t))         # (B, hidden_size)

        # Patchify
        tokens = self.patch_embed(x)    # (B, T, N, D)

        # Add spatial + temporal positional encodings
        tokens = self._add_positional_embeddings(tokens, mprops_seq_len) # (B, T, N, D)

        # Flatten T and N into one sequence for the transformer
        B, T, N, D = tokens.shape
        tokens = tokens.reshape(B, T*N, D)

        # DiT blocks — full attention over all T*N tokens
        for block in self.blocks:
            tokens = block(tokens, c)

        # Project back to bin space
        tokens = self.final_layer(tokens, c)    # (B, T*N, C*p*p)
        out    = self.unpatch(tokens, mprops_seq_len)  # (B, C, H, W, T)

        # Return only future frames (mirrors UNet's final slice)
        return out[:, :, :, :, past_len:]