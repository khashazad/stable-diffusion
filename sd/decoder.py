import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):

        super().__init__()

        self.group_norm = nn.GroupNorm(32, channels)

        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : batch_size x features x Height, Width

        residual = x

        n, c, h, w = x.shape

        # batch_size x features x Height x Width -> batch_size x features x Height * Width
        x = x.view(n, c, h * w)

        # batch_size x features x Height * Width -> batch_size x Height * Width x features
        x = x.transpose(-1, -2)

        # batch_size x Height * Width x features -> batch_size x Height * Width x features
        x = self.attention(x)

        # batch_size x height * width x features -> batch_size x features x height * width
        x = x.transpose(-1, -2)

        # batch_size x features x Height * Width -> batch_size x features x Height x Width
        x = x.view(n, c, h, w)

        x += residual

        return x

class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.group_norm1 = nn.GroupNorm(32, in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)


        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: batch_size x in_channels x Height, Width
        residual = x

        x = self.group_norm1(x)

        x = F.silu(x)

        x = self.conv1(x)

        x = self.group_norm2(x)

        x = F.silu(x)

        x = self.conv2(x)

        return x + self.residual_layer(residual)

class VAE_Decoder(nn.Sequential):

    def __init__(self):

        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/8, Width/8
            VAE_ResidualBlock(512, 512),

            # batch_size x 512 x Height/8, Width/8 -> batch_size x 512 x Height/4, Width/4
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # batch_size x 512 x Height/4, Width/4 -> batch_size x 512 x Height/2, Width/2
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),

            VAE_ResidualBlock(256, 256),

            VAE_ResidualBlock(256, 256),

            # batch_size x 256 x Height/2, Width/2 -> batch_size x 256 x Height, Width
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),

            VAE_ResidualBlock(128, 128),

            VAE_ResidualBlock(128, 128),

            # groups of size 32
            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # batch_size x 128 x Height, Width -> batch_size x 3 x Height, Width
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: batch_size x 4 x Height / 8, Width / 8

        x /= 0.18215

        for module in self:
            x = module(x)

        return x


