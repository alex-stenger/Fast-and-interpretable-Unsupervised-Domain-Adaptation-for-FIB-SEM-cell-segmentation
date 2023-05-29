import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(
        self, in_channels, out_channels, norm_layer, mid_channels=None, return_features=None
    ):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels
        self.return_features = return_features

        Norm = nn.BatchNorm2d if norm_layer == "bn" else nn.InstanceNorm2d
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = Norm(mid_channels) if norm_layer == "bn" else Norm(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = Norm(out_channels) if norm_layer == "bn" else Norm(out_channels)

    def forward(self, x):
        f1 = self.conv1(x)
        fn1 = self.norm1(f1)
        act1 = fn1.relu()
        f2 = self.conv2(act1)
        fn2 = self.norm2(f2)
        act2 = fn2.relu()
        if self.return_features is not None:
            if self.return_features == "conv":
                return (f1, f2), act2
            if self.return_features == "norm":
                return (fn1, fn2), act2
            elif self.return_features == "act":
                return (act1, act2), act2
        return act2


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer, return_features=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_layer, return_features=return_features)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, norm_layer,
        bilinear=True, skip_connection=True, return_features=None
    ):
        super().__init__()
        self.return_features = return_features
        self.skip_connection = skip_connection

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2,
                norm_layer, return_features=return_features
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels, out_channels,
                norm_layer, return_features=return_features
            )

    def forward(self, x1, x2):

        if self.skip_connection :
            x1 = self.up(x1)
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

        else :
            x1 = self.up(x1)
            return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self, n_channels, n_classes, norm_layer="bn", bilinear=False, return_features=None
    ):
        super(UNet, self).__init__()
        self.return_features = return_features
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.norm_layer = norm_layer

        self.inc = DoubleConv(n_channels, 64, norm_layer, return_features=return_features)
        self.down1 = Down(64, 128, norm_layer, return_features=return_features)
        self.down2 = Down(128, 256, norm_layer, return_features=return_features)
        self.down3 = Down(256, 512, norm_layer, return_features=return_features)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, norm_layer, return_features=return_features)
        self.up1 = Up(1024, 512 // factor, norm_layer, bilinear, return_features=return_features)
        self.up2 = Up(512, 256 // factor, norm_layer, bilinear, return_features=return_features)
        self.up3 = Up(256, 128 // factor, norm_layer, bilinear, return_features=return_features)
        self.up4 = Up(128, 64, norm_layer, bilinear, return_features=return_features)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        if self.return_features is not None:
            f1 = x1
        d1 = self.down1(x1)
        if self.return_features is not None:
            f2 = d1
        d2 = self.down2(d1)
        if self.return_features is not None:
            f3 = d2
        d3 = self.down3(d2)
        if self.return_features is not None:
            f4 = d3
        d4 = self.down4(d3)
        if self.return_features is not None:
            f5 = d4
        u1 = self.up1(d4, d3)
        if self.return_features is not None:
            f6 = u1
        u2 = self.up2(u1, d2)
        if self.return_features is not None:
            f7 = u2
        u3 = self.up3(u2, d1)
        if self.return_features is not None:
            f8 = u3
        u4 = self.up4(u3, x1)
        if self.return_features is not None:
            f9 = u4
        logits = self.outc(u4)
        if self.return_features is not None:
            return (f1, f2, f3, f4, f5, f6, f7, f8, f9), logits
        return logits
