from re import A
import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        # batch_size x Channl x Height, Width -> batch_size x 128 x Height, Width
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # batch_size x 128 x Height, Width -> batch_size x 128 x Height, Width
            VAE_ResidualBlock(128, 128),

            # batch_size x 128 x Height, Width -> batch_size x 128 x Height, Width
            VAE_ResidualBlock(128, 128),

            # batch_size x 128 x Height, Width -> batch_size x 128 x Height/2, Width/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # batch_size x 128 x Height/2, Width/2 -> batch_size x 256 x Height/4, Width/4
            VAE_ResidualBlock(128, 256),

            # batch_size x 256 x Height/4, Width/4 -> batch_size x 256 x Height/8, Width/8
            VAE_ResidualBlock(256, 256),


            # batch_size x 256 x Height/2, Width/2 -> batch_size x 256 x Height/4, Width/4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # batch_size x 256 x Height/4, Width/4 -> batch_size x 512 x Height/4, Width/4
            VAE_ResidualBlock(256, 512),

            # batch_size x 512 x Height/4, Width/4 -> batch_size x 512 x Height/4, Width/4
            VAE_ResidualBlock(512, 512),

            # batch_size x 512 x Height/4, Width/4 -> batch_size x 512 x Height/8, Width/8
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/8, Width/8
            VAE_ResidualBlock(512, 512),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/8, Width/8
            VAE_ResidualBlock(512, 512),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/8, Width/8
            VAE_ResidualBlock(512, 512),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/8, Width/8
            VAE_AttentionBlock(512),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/8, Width/8
            VAE_ResidualBlock(512, 512),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/8, Width/8
            nn.GroupNorm(32, 512),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/8, Width/8
            nn.SiLU(),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 8 x Height/8, Width/8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # batch_size x 8 x Height/8, Width/8 -> batch_size x 8 x Height/8, Width/8
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor ) -> torch.Tensor:
        # x: batch_size x channel x Height, Width
        # noise: batch_size x noise_channels x Height / 8, Width / 8


        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # (pad_left, pad_right, pad_top, pad_bottom)
                x = x.pad(0, (0, 1, 0, 1))

            x = module(x)

        # batch_size x 8 x Height/8, Width/8 -> two tensors of size batch_size x 4 x Height/8, Width/8
        mean, log_var = torch.chunk(x, 2, dim=1)

        log_variance = torch.clamp(log_var, min=-30, max=20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        # Reparameterization trick
        z = mean + stdev * noise

        # scale by constant
        z *= 0.18215

        return z
