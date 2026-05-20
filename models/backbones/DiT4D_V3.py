import torch
import torch.nn as nn
from .embeddings import SinusoidalPositionEmbeddings


class PatchEmbed4D(nn.Module):
    """
    Patchify (B, C, H, W, T) using Conv3d with a partial temporal kernel.

    One token per spatial patch position, each encoding the full
    temporal evolution at that location.

    For ATC with T=8, t_patch_size=2, patch_size=4:
        h_patches = 12/4 = 3
        w_patches = 36/4 = 9
        t_patches = 8/2  = 4
        total tokens = 3 × 9 × 4 = 108

    Each token encodes what happens at one (4×4) spatial cell across 2
    consecutive frames — strong local temporal coupling without full collapse.
    """
    def __init__(self, grid_rows, grid_cols, T_total, patch_size, t_patch_size, in_channels, hidden_size):
        super().__init__()

        assert grid_rows % patch_size == 0, \
            f"grid_rows ({grid_rows}) must be divisible by patch_size ({patch_size})"
        assert grid_cols % patch_size == 0, \
            f"grid_cols ({grid_cols}) must be divisible by patch_size ({patch_size})"
        assert T_total % t_patch_size == 0, \
            f"T_total ({T_total}) must be divisible by t_patch_size ({t_patch_size})"

        self.patch_size    = patch_size
        self.t_patch_size  = t_patch_size
        self.h_patches     = grid_rows // patch_size
        self.w_patches     = grid_cols // patch_size
        self.t_patches     = T_total   // t_patch_size
        self.num_patches   = self.h_patches * self.w_patches * self.t_patches

        self.proj = nn.Conv3d(
            in_channels  = in_channels,
            out_channels = hidden_size,
            kernel_size  = (t_patch_size, patch_size, patch_size),
            stride       = (t_patch_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, T)
        Returns:
            tokens: (B, T_p*N_s, D)
                T_p = t_patches = T/pt
                N_s = h_patches * w_patches
        """
        # Conv3d expects (B, C, D, H, W) — move T_total to the D position
        x = x.permute(0, 1, 4, 2, 3)           # (B, C, T, H, W)
        x = self.proj(x)                       # (B, D, T_p, h_p, w_p)
        B, D, T_p, h_p, w_p = x.shape
        x = x.permute(0, 2, 3, 4, 1)           # (B, T_p, h_p, w_p, D)
        x = x.reshape(B, T_p * h_p * w_p, D)   # (B, T_p*N_s, D)
        return x

class PatchUnEmbed4D(nn.Module):
    """
    Reassemble token sequence → (B, C, H, W, T), then slice to future frames.

    Each token predicts t_patch consecutive frames at its spatial location.
    FinalLayer projects each token to t_patch*C*p*p.
    """
    def __init__(self, grid_rows, grid_cols, patch_size, out_channels, t_patch_size, past_len):
        super().__init__()
        self.patch_size   = patch_size
        self.t_patch_size = t_patch_size
        self.h_patches    = grid_rows // patch_size
        self.w_patches    = grid_cols // patch_size
        self.out_channels = out_channels
        self.past_len     = past_len

    def forward(self, x: torch.Tensor, T_p: int) -> torch.Tensor:
        """
        Args:
            x  : (B, T_p*N_s, pt*C*p*p)  — projected by FinalLayer
            T_p: number of temporal patch slots (T_total / t_patch)
        Returns:
            (B, C, H, W, future_len)
        """
        B    = x.shape[0]
        p    = self.patch_size
        pt   = self.t_patch_size
        h, w = self.h_patches, self.w_patches
        C    = self.out_channels

        x = x.reshape(B, T_p, h, w, pt, C, p, p)
        x = x.permute(0, 5, 1, 4, 2, 6, 3, 7)    # (B, C, T_p, pt, h, p, w, p)
        x = x.reshape(B, C, T_p * pt, h * p, w * p)  # (B, C, T, H, W)
        x = x.permute(0, 1, 3, 4, 2)              # (B, C, H, W, T)

        # Slice off past frames — mirrors UNet's h[:,:,:,:,past_len:]
        return x[:, :, :, :, self.past_len:]       # (B, C, H, W, future_len)

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
    Projects each token from hidden_size → t_patch_size*C*p*p.
    Each token predicts t_patch_size consecutive frames at its spatial location.
    """
    def __init__(self, hidden_size, patch_size, out_channels, t_patch_size):
        super().__init__()
        self.norm   = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, t_patch_size * out_channels * patch_size * patch_size)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT4D_V3(nn.Module):
    """
    DiT backbone with partial temporal tube patching (Conv3D, kernel pt < T).

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
        t_patch_size:      int   = 2,
        patch_size:        int   = 4,
        hidden_size:       int   = 256,
        depth:             int   = 6,
        num_heads:         int   = 4,
        mlp_ratio:         float = 4.0,
        dropout_rate:      float = 0.1,
        time_multiple:     int   = 4,        # same name as UNet
        total_time_steps:  int   = 1000,     # same name as UNet
        condition:         str   = "Past",
        T_max:             int   = 32,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert (past_len + future_len) % t_patch_size == 0, \
            f"T_total={past_len+future_len} must be divisible by t_patch={t_patch_size}"

        self.condition      = condition
        self.input_channels = input_channels
        self.past_len       = past_len
        self.future_len     = future_len
        self.t_patch_size   = t_patch_size
        T_total             = past_len + future_len
        self.T_p            = T_total // t_patch_size

        time_emb_dims_exp = hidden_size * time_multiple

        # Diffusion timestep embedding (same as UNet)
        # this t is the diffusion step — separate from the temporal frames in the sequence below.
        self.dif_time_embeddings = SinusoidalPositionEmbeddings(
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
        self.patch_embed = PatchEmbed4D(grid_rows, grid_cols, T_total, patch_size, t_patch_size, input_channels, hidden_size)
        N_s = self.patch_embed.h_patches * self.patch_embed.w_patches   # spatiotemporal patches

        # Spatial positional encoding
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, N_s, hidden_size))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # Temporal positional encoding
        t_max_slots = T_max // t_patch_size
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, t_max_slots, hidden_size))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])

        # Output: predict F frames per spatial token
        self.final_layer = FinalLayer(hidden_size, patch_size, output_channels, t_patch_size)
        self.unpatch     = PatchUnEmbed4D(grid_rows, grid_cols, patch_size, output_channels, t_patch_size, past_len)

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

    def _add_positional_embeddings(self, tokens: torch.Tensor, T_p: int, N_s: int) -> torch.Tensor:
        """
        Add spatial + temporal positional encodings.

        tokens: (B, T_p*N_s, D) — flat sequence
        Reshape to (B, T_p, N_s, D), add embeddings, flatten back.

        spatial:  (1, 1,   N_s, D) → same position encoding for every temporal slot
        temporal: (1, T_p, 1,   D) → same slot encoding for every spatial patch
        """
        B = tokens.shape[0]
        D = tokens.shape[2]
        tokens = tokens.reshape(B, T_p, N_s, D)

        spatial  = self.spatial_pos_embed.unsqueeze(1)            # (1, 1,   N_s, D)
        temporal = self.temporal_pos_embed[:, :T_p, :].unsqueeze(2)  # (1, T_p, 1,   D)
        tokens   = tokens + spatial + temporal                    # (B, T_p, N_s, D)

        return tokens.reshape(B, T_p * N_s, D)

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
        c = self.time_proj(self.dif_time_embeddings(t))         # (B, D=hidden_size)

        # Patchify
        tokens = self.patch_embed(x)
        N_s    = self.patch_embed.h_patches * self.patch_embed.w_patches
        tokens = self._add_positional_embeddings(tokens, self.T_p, N_s)

        # DiT blocks — attention over tokens
        for block in self.blocks:
            tokens = block(tokens, c)

        # Project each token → t_patch frames, then unpatch + slice
        tokens = self.final_layer(tokens, c)    # (B, T_p*N_s, pt*C*p*p)
        return self.unpatch(tokens)             # (B, C, H, W, future_len)