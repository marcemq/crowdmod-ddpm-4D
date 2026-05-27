import torch
import torch.nn as nn
from .embeddings import SinusoidalPositionEmbeddings


class PatchEmbed4D(nn.Module):
    """
    Patchify (B, C, H, W, T) using Conv3d with full temporal tube kernel.

    One token per spatial patch position, each encoding the full
    temporal evolution at that location.

    For ATC: (12/4) × (36/4) × 1 = 3 × 9 = 27 tokens
    Each token knows what happens at this grid cell across ALL T_total frames.
    """
    def __init__(self, grid_rows, grid_cols, T_total, patch_size, in_channels, hidden_size):
        super().__init__()

        assert grid_rows % patch_size == 0, \
            f"grid_rows ({grid_rows}) must be divisible by patch_size ({patch_size})"
        assert grid_cols % patch_size == 0, \
            f"grid_cols ({grid_cols}) must be divisible by patch_size ({patch_size})"

        self.patch_size  = patch_size
        self.h_patches   = grid_rows // patch_size
        self.w_patches   = grid_cols // patch_size
        self.num_patches = self.h_patches * self.w_patches * 1
        self.T_total     = T_total

        # Conv3d: kernel covers full spatial patch and full time depth
        # Input  to Conv3d: (B, C, H, W, T) but PyTorch Conv3d expects (B, C_in, T, H, W)
        self.proj = nn.Conv3d(
            in_channels  = in_channels,
            out_channels = hidden_size,
            kernel_size  = (T_total, patch_size, patch_size),
            stride       = (T_total, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, T)
        Returns:
            tokens: (B, N, D)   time dimension fully consumed by the tube kernel
        """
        # Conv3d expects (B, C, D, H, W) — move T_total to the D position
        x = x.permute(0, 1, 4, 2, 3)          # (B, C, T, H, W)
        x = self.proj(x)                       # (B, D, 1, h_p, w_p)
        x = x.squeeze(2)                       # (B, D, h_p, w_p)
        x = x.flatten(2).transpose(1, 2)       # (B, N, D)
        return x

class PatchUnEmbed4D(nn.Module):
    """
    Reassemble (B, N, F*C*p*p) token sequence → (B, C, H, W, F).

    With tube patching there is no time dimension in the tokens —
    FinalLayer already projected each token to F*C*p*p (all future frames
    packed per spatial token).
    """
    def __init__(self, grid_rows, grid_cols, patch_size, out_channels, future_len):
        super().__init__()
        self.patch_size   = patch_size
        self.h_patches    = grid_rows // patch_size
        self.w_patches    = grid_cols // patch_size
        self.out_channels = out_channels
        self.future_len   = future_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, F*C*p*p)
        Returns:
            (B, C, H, W, F)
        """
        B    = x.shape[0]
        p    = self.patch_size
        h, w = self.h_patches, self.w_patches
        C    = self.out_channels
        F    = self.future_len

        x = x.reshape(B, h, w, F, C, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3)    # (B, C, h, p, w, p, F)
        x = x.reshape(B, C, h*p, w*p, F)       # (B, C, H, W, F)
        return x

def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
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
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        x_mod = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out
        x_mod = modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.mlp(x_mod)
        return x


class FinalLayer(nn.Module):
    """
    Projects each token from hidden_size → F*C*p*p.
    With tube patching each token predicts ALL future frames at its spatial location.
    """
    def __init__(self, hidden_size, patch_size, out_channels, future_len):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, future_len * out_channels * patch_size * patch_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT4D(nn.Module):
    """
    DiT version 2.
    Diffusion Transformer for spatiotemporal macroscopic sequences.

    Uses Conv3D full temporal tube patching. Each of the N spatial tokens
    encodes the full temporal evolution (all past+future frames) at one grid cell.
    FinalLayer predicts F*C*p*p per token (all future frames at once)
    Mirrors UNet's forward signature: forward(future, t, past) → future_hat
    """
    def __init__(
        self,
        input_channels:    int   = 4,
        output_channels:   int   = 4,
        grid_rows:         int   = 12,
        grid_cols:         int   = 36,
        past_len:          int   = 5,
        future_len:        int   = 3,
        patch_size:        int   = 4,
        hidden_size:       int   = 256,
        depth:             int   = 6,
        num_heads:         int   = 4,
        mlp_ratio:         float = 4.0,
        dropout_rate:      float = 0.1,
        time_multiple:     int   = 4,        # same name as UNet
        total_time_steps:  int   = 1000,     # same name as UNet
        condition:         str   = "Past",
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.condition      = condition
        self.input_channels = input_channels
        self.past_len       = past_len
        self.future_len     = future_len
        T_total             = past_len + future_len

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
        self.patch_embed = PatchEmbed4D(grid_rows, grid_cols, T_total, patch_size, input_channels, hidden_size)
        N = self.patch_embed.num_patches   # spatiotemporal patches

        # Spatial-temporal encoding
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, N, hidden_size))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])

        # Output: predict F frames per spatial token
        self.final_layer = FinalLayer(hidden_size, patch_size, output_channels, future_len)
        self.unpatch     = PatchUnEmbed4D(grid_rows, grid_cols, patch_size, output_channels, future_len)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight.view(m.weight.size(0), -1))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, future: torch.Tensor, t: torch.Tensor, past: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            future : (B, C, H, W, F)  noisy future frames to be predicted
            t : (B,)  long            diffusion timestep in [0, total_time_steps)
            past: (B, C, H, W, P)     clean past frames (conditioning)
        Returns:
            (B, C, H, W, F)  predicted noise (DDPM) or velocity field (FM)
        """
        # Concatenate past + future along time (mirrors UNet)
        if self.condition == "Past" and past is not None:
            x = torch.cat([past, future], dim=4)    # (B, C, H, W, P+F)
        else:
            x = future

        # Diffusion timestep conditioning → (B, D)
        c = self.time_proj(self.time_embeddings(t))         # (B, hidden_size)

        # Patchify
        tokens = self.patch_embed(x) + self.spatial_pos_embed    # (B, N, D)

        # DiT blocks — attention over N tokens
        for block in self.blocks:
            tokens = block(tokens, c)

        # Project to F future frames per token + unpatch
        tokens = self.final_layer(tokens, c)    # (B, N, F*C*p*p)
        return self.unpatch(tokens)             # (B, C, H, W, F)