import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convrelu(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)
        self.deconv = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, kernel_size=4, stride=2, padding=1, output_padding=0
        )
        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x


class ResNeXtUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters = [256, 512, 1024, 2048]

        # Encoder layers
        self.encoder0 = nn.Sequential(*self.base_layers[:3])  # Conv1 + BN + ReLU
        self.encoder1 = nn.Sequential(*self.base_layers[4])   # Layer1
        self.encoder2 = nn.Sequential(*self.base_layers[5])   # Layer2
        self.encoder3 = nn.Sequential(*self.base_layers[6])   # Layer3
        self.encoder4 = nn.Sequential(*self.base_layers[7])   # Layer4

        # Decoder layers
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], 64)

        # Final classifier
        self.final_conv = nn.Sequential(
            ConvRelu(64, 32, 3, 1),
            nn.Conv2d(32, n_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x0 = self.encoder0(x)
        e1 = self.encoder1(x0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        return self.final_conv(d1)