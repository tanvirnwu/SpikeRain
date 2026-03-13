"""SpikeRain network definition aligned with WACV 2026 paper sections.

The core blocks are implemented in :mod:`model.modules` and correspond to:
- Section 3.2: DSRB
- Section 3.3: MDSA
- Section 3.4: Temporal Fusion
- Section 3.5: ARFE
"""

from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import functional, layer
import torch
import torch.nn as nn

from .modules import (
    ARFE,
    DSRB,
    TemporalFusion,
    ThresholdDependentBatchNorm2d,
    OverlapPatchEmbed,
    DownSampling,
    UpSampling,
)

v_th = 0.15


class SpikeRain(nn.Module):
    """SpikeRain deraining model.

    Inputs:
        input_image: (B, C, H, W) or (T, B, C, H, W)
    Outputs:
        final_output: (B, C, H, W)

    The forward pass follows an encoder-decoder with spiking residual blocks,
    temporal fusion, and ARFE refinement as described in the paper.
    """

    def __init__(self, inp_channels=3, out_channels=3, dim=24,
                 en_num_blocks=[4, 4, 6, 6], de_num_blocks=[4, 4, 6, 6],
                 bias=False, T=4):
        super(SpikeRain, self).__init__()

        functional.set_backend(self, backend='torch')
        functional.set_step_mode(self, step_mode='m')

        self.T = T

        # Input Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim)

        # Encoder Level 1
        self.encoder_level1 = nn.Sequential(*[
            DSRB(dim=dim) for _ in range(en_num_blocks[0])
        ])

        # Encoder Level 2
        self.downsample_1_to_2 = DownSampling(dim)
        self.encoder_level2 = nn.Sequential(*[
            DSRB(dim=dim * 2) for _ in range(en_num_blocks[1])
        ])

        # Encoder Level 3
        self.downsample_2_to_3 = DownSampling(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            DSRB(dim=dim * 4) for _ in range(en_num_blocks[2])
        ])

        # Decoder Level 3
        self.decoder_level3 = nn.Sequential(*[
            DSRB(dim=dim * 4) for _ in range(de_num_blocks[2])
        ])

        # Decoder Level 2
        self.upsample_3_to_2 = UpSampling(dim * 4)
        self.reduce_channels_2 = nn.Sequential(
            LIFNode(v_threshold=v_th, backend='torch', step_mode='m', decay_input=False),
            layer.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias, step_mode='m'),
            ThresholdDependentBatchNorm2d(dim * 2),
        )
        self.decoder_level2 = nn.Sequential(*[
            DSRB(dim=dim * 2) for _ in range(de_num_blocks[1])
        ])

        # Decoder Level 1
        self.upsample_2_to_1 = UpSampling(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            DSRB(dim=dim * 2) for _ in range(de_num_blocks[0])
        ])

        # Temporal Fusion Module
        self.temporal_fusion = TemporalFusion(T=self.T)

        # Feature Refinement and Output
        self.refinement = ARFE(channel=dim * 2, reduction=8)
        self.output_conv = nn.Conv2d(dim * 2, out_channels,
                                     kernel_size=3, stride=1, padding=1)

    def forward(self, input_image):
        """Run SpikeRain encoder-decoder, temporal fusion, and residual output."""
        residual_image = input_image.clone()

        # Expand temporal dimension if necessary
        if input_image.dim() < 5:
            input_image = input_image.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        # ----- Encoding Path -----
        enc_lvl1_inp = self.patch_embed(input_image)
        enc_lvl1_out = self.encoder_level1(enc_lvl1_inp)

        enc_lvl2_inp = self.downsample_1_to_2(enc_lvl1_out)
        enc_lvl2_out = self.encoder_level2(enc_lvl2_inp)

        enc_lvl3_inp = self.downsample_2_to_3(enc_lvl2_out)
        enc_lvl3_out = self.encoder_level3(enc_lvl3_inp)

        # ----- Decoding Path -----
        dec_lvl3_out = self.decoder_level3(enc_lvl3_out)

        dec_lvl2_inp = self.upsample_3_to_2(dec_lvl3_out)
        dec_lvl2_inp = torch.cat([dec_lvl2_inp, enc_lvl2_out], dim=2)
        dec_lvl2_inp = self.reduce_channels_2(dec_lvl2_inp)
        dec_lvl2_out = self.decoder_level2(dec_lvl2_inp)

        dec_lvl1_inp = self.upsample_2_to_1(dec_lvl2_out)
        dec_lvl1_inp = torch.cat([dec_lvl1_inp, enc_lvl1_out], dim=2)
        dec_lvl1_out = self.decoder_level1(dec_lvl1_inp)

        # ----- Temporal Fusion -----
        fused_features = self.temporal_fusion(dec_lvl1_out)

        # ----- Feature Refinement and Reconstruction -----
        refined_features = self.refinement(fused_features)
        reconstructed_image = self.output_conv(refined_features)

        # Residual connection with the original input
        final_output = reconstructed_image + residual_image

        return final_output


class SpikeRainFactory:
    """Factory for SpikeRain model variants (S/M/L)."""

    def __init__(self, T=4, device='cuda'):
        self.T = T
        self.device = device

    def get_model(self, variant='M'):
        """Return a SpikeRain variant by size key: 'S', 'M', or 'L'."""
        variant = variant.upper()
        if variant == 'S':
            return self.spikerain_s()
        if variant == 'M':
            return self.spikerain_m()
        if variant == 'L':
            return self.spikerain_l()
        raise ValueError(f"Invalid variant '{variant}'. Choose from 'S', 'M', or 'L'.")

    def spikerain_s(self):
        """Small SpikeRain configuration."""
        return SpikeRain(
            dim=32,
            en_num_blocks=[2, 2, 4, 4],
            de_num_blocks=[1, 1, 1, 1],
            T=self.T
        ).to(self.device)

    def spikerain_m(self):
        """Medium SpikeRain configuration."""
        return SpikeRain(
            dim=48,
            en_num_blocks=[4, 4, 8, 8],
            de_num_blocks=[2, 2, 2, 2],
            T=self.T
        ).to(self.device)

    def spikerain_l(self):
        """Large SpikeRain configuration."""
        return SpikeRain(
            dim=64,
            en_num_blocks=[6, 6, 8, 10],
            de_num_blocks=[4, 4, 4, 4],
            T=self.T
        ).to(self.device)
