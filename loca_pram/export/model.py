"""LOCA-PRAM model (inference copy of ../training/train.ipynb) for export."""

import torch
import torch.nn as nn

# Model input size (128x128 = 256x256 native, downsampled x2).
WINDOW_SIZE = (128, 128)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        # Convs feed straight into BatchNorm, whose beta absorbs the bias.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding="same", bias=True),
        )

    def forward(self, x):
        return self.up(x)


class multi_head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.multi_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                      padding="same", bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                      padding="same", bias=True),
        )

    def forward(self, x):
        return self.multi_head(x)


class GaussianMixtureModel(nn.Module):
    """LOCA-PRAM U-Net (inference copy of the training-notebook model).

    Heads: p (1), mu_xy (2), mu_a (1), sigma (3), bg (1).
    """

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

        # Encoder: 128 -> 64 -> 32 -> 16 (dilated bottleneck)
        self.Conv1 = conv_block(num_channels, 36)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = conv_block(36, 72)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = conv_block(72, 144)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv4 = nn.Sequential(
            conv_block(144, 288, dilation=2),
            conv_block(288, 288, dilation=4),
        )

        # Decoder
        self.Up3 = up_conv(288, 144)
        self.Up_conv3 = conv_block(288, 144)
        self.Up2 = up_conv(144, 72)
        self.Up_conv2 = conv_block(144, 72)
        self.Up1 = up_conv(72, 36)
        self.Up_conv1 = conv_block(72, 36)

        # Scale on the tanh mu_xy offset (fixed; not stored in the state_dict).
        self.tanh_scale = 6.0

        self.head_p     = multi_head(36, 1)
        self.head_mu_xy = multi_head(36, 2)
        self.head_mu_a  = multi_head(36, 1)
        self.head_sigma = multi_head(36, 3)
        self.head_bg    = multi_head(36, 1)

    def forward_heads(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool1(x1))
        x3 = self.Conv3(self.Maxpool2(x2))
        x4 = self.Conv4(self.Maxpool3(x3))
        d3 = self.Up3(x4); d3 = torch.cat((x3, d3), dim=1); d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3); d2 = torch.cat((x2, d2), dim=1); d2 = self.Up_conv2(d2)
        d1 = self.Up1(d2); d1 = torch.cat((x1, d1), dim=1); d1 = self.Up_conv1(d1)
        p     = torch.sigmoid(self.head_p(d1))
        mu_xy = torch.tanh(self.head_mu_xy(d1))
        mu_a  = torch.sigmoid(self.head_mu_a(d1))
        sigma = torch.sigmoid(self.head_sigma(d1))
        bg    = self.head_bg(d1)
        return p, mu_xy, mu_a, sigma, bg

    def forward(self, x, training=False):
        p, mu_xy, mu_a, sigma, bg = self.forward_heads(x)
        pxyn_mean = torch.cat([mu_xy * self.tanh_scale, mu_a * 100.0], dim=1)
        pxy_std = sigma * 2.4 + 0.1
        return p, pxyn_mean, pxy_std, bg

    def forward_export(self, x):
        """Export path: only p and mu_xy (tanh_scale applied host-side)."""
        p, mu_xy, _, _, _ = self.forward_heads(x)
        return p, mu_xy
