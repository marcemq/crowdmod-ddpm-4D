import torch
import torch.nn as nn
from models.embeddings import SinusoidalPositionEmbeddings
from models.layers import DownSample, UpSample, ResnetBlock


class MacropropsDenoiser(nn.Module):
    def __init__(
        self,
        input_channels = 4,
        output_channels= 4,
        num_res_blocks = 2,
        base_channels  = 128, #AR: not sure about this base size
        # Resolutions
        #base_channels_multiples=(1, 2, 4, 8, 16), #AR: not sure about last 16 #Altura de la Unet
        base_channels_multiples=[1, 2, 4, 8],
        # Attention, per resolution level
        apply_attention=[False, False, True, False, False],
        dropout_rate   = 0.1,
        time_multiple  = 4,
        condition = "Past"
    ):
        super().__init__()
        self.input_channels = input_channels
        time_emb_dims_exp    = base_channels * time_multiple
        self.condition      = condition
        self.time_embeddings = SinusoidalPositionEmbeddings(time_emb_dims=base_channels, time_emb_dims_exp=time_emb_dims_exp)

        self.first = nn.Conv3d(in_channels=input_channels, out_channels=base_channels, kernel_size=3, stride=1, padding="same")

        num_resolutions = len(base_channels_multiples)
        if self.condition == "Past":
            # TODO: We need a way to emb the past frames
            self.past_encoding  = None

        # Encoder part of the UNet. Dimension reduction.
        self.encoder_blocks = nn.ModuleList()
        curr_channels       = [base_channels]
        in_channels         = base_channels

        # For each resolution level
        for level in range(num_resolutions):
            # Number of output channels for this level
            out_channels = base_channels * base_channels_multiples[level]

            # For each residual block
            for _ in range(num_res_blocks):
                # Create block
                block = ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                    condition=self.condition
                )
                # Add it to the group of blocks
                self.encoder_blocks.append(block)

                in_channels = out_channels
                curr_channels.append(in_channels)

            # For all the levels before the last, add a DownSample
            if level != (num_resolutions - 1):
                self.encoder_blocks.append(DownSample(channels=in_channels))
                curr_channels.append(in_channels)

        # Bottleneck between the encoder and decoder
        self.bottleneck_blocks = nn.ModuleList(
            (
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=True,
                    condition=self.condition
                ),
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=False,
                    condition=self.condition
                ),
            )
        )

        # Decoder part of the UNet. Dimension restoration with skip-connections.
        self.decoder_blocks = nn.ModuleList()

        # Decoder
        for level in reversed(range(num_resolutions)):
            out_channels = base_channels * base_channels_multiples[level]

            for _ in range(num_res_blocks + 1):
                encoder_in_channels = curr_channels.pop()
                block = ResnetBlock(
                    in_channels=encoder_in_channels + in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    time_emb_dims=time_emb_dims_exp,
                    apply_attention=apply_attention[level],
                    condition=self.condition
                )

                in_channels = out_channels
                self.decoder_blocks.append(block)

            if level != 0:
                self.decoder_blocks.append(UpSample(in_channels))

        # A last convolutional block
        self.final = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels), #AR: not sure about 8
            nn.SiLU(),
            nn.Conv3d(in_channels=in_channels, out_channels=output_channels, kernel_size=3, stride=1, padding="same"),
        )

    def forward(self, future, t, past=None):
        """
        future: correspond to future frames to be predicted
        past: correspond to the past frames on which the condition states
        return a tensor of shape future TODO: review that this is true
        """
        # Time embeddings
        time_emb = self.time_embeddings(t)
        x = None
        _,_,_,_,past_len_frames = past.shape
        # Past embeddings
        if self.condition == "Past":
            #_,(past_encodings,__)= self.past_encoding(past)
            #past_encodings        = past_encodings[-1]
            x = torch.cat([past, future], dim=self.input_channels)
            past_encodings = None
        else:
            x = future
            past_encodings = None

        h    = self.first(x)
        outs = [h]

        # Encoder
        for layer in self.encoder_blocks:
            h = layer(h, time_emb, past_encodings)
            outs.append(h)

        # Bottleneck
        for layer in self.bottleneck_blocks:
            h = layer(h, time_emb, past_encodings)

        # Decoder
        for layer in self.decoder_blocks:
            if isinstance(layer, ResnetBlock):
                out = outs.pop()
                h = torch.cat([h, out], dim=1)
            h = layer(h, time_emb, past_encodings)

        h = self.final(h)

        if self.condition == "Past":
            h = h[:,:,:,:,past_len_frames:]
        return h