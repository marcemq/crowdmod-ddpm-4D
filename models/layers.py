import torch
import torch.nn as nn
# Attention block
# Future AR: temporal attention, spacial attention
class AttentionBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.channels  = channels
        self.group_norm= nn.GroupNorm(num_groups=8, num_channels=channels)
        self.spatial_mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)
        self.temporal_mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, H, W, L = x.shape
        h_spa = self.group_norm(x)

        # Reshape for spatial attention: [B, C, H, W, L] --> [B*L, C, H*W]
        h_spatial = h_spa.permute(0, 4, 1, 2, 3).reshape(B*L, C, H*W).swapaxes(1, 2)  # [B*L, H*W, C]
        h_spatial, _ = self.spatial_mhsa(h_spatial, h_spatial, h_spatial)  # [B*L, H*W, C]
        h_spatial = h_spatial.swapaxes(1, 2).reshape(B, L, C, H, W).permute(0, 2, 3, 4, 1)  # [B, C, H, W, L]

        # Reshape for temporal attention: [B, C, H, W, L] --> [B*H*W, C, L]
        h_temporal = h_spatial.permute(0, 2, 3, 1, 4).reshape(B*H*W, C, L).swapaxes(1, 2)  # [B*H*W, L, C]
        h_temporal, _ = self.temporal_mhsa(h_temporal, h_temporal, h_temporal)  # [B*H*W, L, C]
        h_temporal = h_temporal.swapaxes(1, 2).reshape(B, H, W, C, L).permute(0, 3, 1, 2, 4)  # [B, C, H, W, L]

        # Reshape for spatial attention: [B, C, H, W, L] --> [B*L, C, H*W]
        h_spatial = h_temporal.permute(0, 4, 1, 2, 3).reshape(B*L, C, H*W).swapaxes(1, 2)  # [B*L, H*W, C]
        h_spatial, _ = self.spatial_mhsa(h_spatial, h_spatial, h_spatial)  # [B*L, H*W, C]
        h_spatial = h_spatial.swapaxes(1, 2).reshape(B, L, C, H, W).permute(0, 2, 3, 4, 1)  # [B, C, H, W, L]

        # Add temporal attention to the original input
        x = x + h_spatial

        return x
    
# Resnet block
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels, dropout_rate=0.1, time_emb_dims=512, apply_attention=False, condition="Past"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels

        self.activation  = nn.SiLU()
        self.condition   = condition
        # Group 1
        self.normalize_1 = nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        #AR: N, C, L, H, W we might need to change last channel to be in the midle one??
        self.conv_1      = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        # Group 2 time embedding
        self.dense_1    = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)
        # TODO: review usage of condition here, is there a handle of actual frames?
        #if self.condition!="None":
        #    self.dense_2 = nn.Linear(in_features=time_emb_dims, out_features=self.out_channels)

        # Group 3
        self.normalize_2= nn.GroupNorm(num_groups=8, num_channels=self.out_channels)
        self.dropout    = nn.Dropout3d(p=dropout_rate)
        self.conv_2     = nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding="same")

        if self.in_channels != self.out_channels:
            self.match_input = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

        if apply_attention:
            self.attention = AttentionBlock(channels=self.out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x, t, y=None):
        # Group 1
        h = self.activation(self.normalize_1(x))
        h = self.conv_1(h)

        # Group 2
        # add in timestep embedding
        h += self.dense_1(self.activation(t))[:, :, None, None, None]
        # TODO: past frames here?
        #if self.condition!="None" and y is not None:
        #    cond = self.dense_2(self.activation(y))
            # Use broadcasting to add the condition
        #    h += cond[:, :, None, None, None]

        # Group 3
        h = self.activation(self.normalize_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        # Residual and attention
        h = h + self.match_input(x)
        h = self.attention(h)

        return h

# Downsampling-convolutive layer
class DownSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.downsample = nn.Conv3d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x, *args):
        return self.downsample(x)

# Upsampling-convolutive layer
class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1))
    def forward(self, x, *args):
        return self.upsample(x)