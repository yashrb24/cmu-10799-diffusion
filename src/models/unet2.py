"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep

    Encoder (Downsampling path)

    Middle

    Decoder (Upsampling path)

    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks

    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3,
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm

        # TODO: build your own unet architecture here
        # Pro tips: remember to take care of the time embeddings!
        self.num_levels = len(channel_mult)

        time_embed_dim = 4 * base_channels
        self.timestep_embedding = TimestepEmbedding(time_embed_dim=time_embed_dim)

        first_channel = base_channels * channel_mult[0]
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=first_channel, kernel_size=3, padding=1
        )

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        prev_ch = first_channel
        curr_resolution = 64

        for level_idx, mult in enumerate(channel_mult):
            ch = base_channels * mult
            level = nn.ModuleList()

            level.append(
                ResBlock(
                    in_channels=prev_ch,
                    out_channels=ch,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            if curr_resolution in attention_resolutions:
                level.append(AttentionBlock(channels=ch, num_heads=num_heads))
            

            for _ in range(num_res_blocks - 1):
                level.append(
                    ResBlock(
                        in_channels=ch,
                        out_channels=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                if curr_resolution in attention_resolutions:
                    level.append(AttentionBlock(channels=ch, num_heads=num_heads))
                
            self.skip_channels.append(ch)
            self.encoder_blocks.append(level)

            if level_idx != len(channel_mult) - 1:
                self.downsamples.append(Downsample(channels=ch))
                curr_resolution //= 2
            else:
                self.downsamples.append(None)

            prev_ch = ch

        self.bottleneck_layer = nn.ModuleList(
            [
                ResBlock(
                    in_channels=prev_ch,
                    out_channels=prev_ch,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                AttentionBlock(channels=prev_ch, num_heads=num_heads),
                ResBlock(
                    in_channels=prev_ch,
                    out_channels=prev_ch,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            ]
        )

        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for level_idx, (curr_mult, prev_mult) in reversed(
            list(enumerate(zip(channel_mult[1:], channel_mult[0:])))
        ):
            ch = base_channels * curr_mult
            prev_ch = base_channels * prev_mult

            self.upsamples.append(Upsample(channels=ch))
            curr_resolution *= 2

            level = nn.ModuleList()
            level.append(
                ResBlock(
                    in_channels=ch + self.skip_channels[level_idx],
                    out_channels=prev_ch,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )
            if curr_resolution in attention_resolutions:
                level.append(AttentionBlock(channels=prev_ch))

            for _ in range(num_res_blocks - 1):
                level.append(
                    ResBlock(
                        in_channels=prev_ch, 
                        out_channels=prev_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                if curr_resolution in attention_resolutions:
                    level.append(AttentionBlock(channels=prev_ch))

            self.decoder_blocks.append(level)


        self.norm_activation = nn.Sequential(
            GroupNorm32(num_groups=32, num_channels=first_channel),
            nn.SiLU()
        )

        self.final_conv = nn.Conv2d(
            in_channels=first_channel, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x, t):
        t_emb = self.timestep_embedding(t)
        h = self.initial_conv(x)

        skips = []
        for level, downsample in zip(self.encoder_blocks, self.downsamples):
            for block in level:
                h = self.__forward_helper(block, h, t_emb)

            if downsample is not None:
                skips.append(h)
                h = downsample(h)

        for block in self.bottleneck_layer:
            h = self.__forward_helper(block, h, t_emb)

        for level, upsample in zip(self.decoder_blocks, self.upsamples):
            h = upsample(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for block in level:
                h = self.__forward_helper(block, h, t_emb)

        h = self.norm_activation(h)
        h = self.final_conv(h)
        return h

    def __forward_helper(self, block: nn.Module, x: torch.Tensor, t: torch.Tensor):
        if isinstance(block, ResBlock):
            return block(x, t)
        else:
            return block(x)


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.

    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters

    Returns:
        Instantiated UNet model
    """
    model_config = config["model"]
    data_config = config["data"]

    return UNet(
        in_channels=data_config["channels"],
        out_channels=data_config["channels"],
        base_channels=model_config["base_channels"],
        channel_mult=tuple(model_config["channel_mult"]),
        num_res_blocks=model_config["num_res_blocks"],
        attention_resolutions=model_config["attention_resolutions"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        use_scale_shift_norm=model_config["use_scale_shift_norm"],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)

    with torch.no_grad():
        out = model(x, t)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
