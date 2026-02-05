import torch
import torch.nn as nn
import torch.nn.functional as F

#Conv3D -> ReLU -> Conv3D -> ReLU
class DoubleConv(nn.Module):
        
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_channels=16):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)

        # Decoder
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Output layer
        self.out_conv = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)         # (B, base, D, H, W)
        x2 = self.pool1(x1)

        x3 = self.enc2(x2)        # (B, base*2, ...)
        x4 = self.pool2(x3)

        # Bottleneck
        x5 = self.bottleneck(x4)  # (B, base*4, ...)

        # Decoder
        x = self.up2(x5)

        # shapes can mismatch by 1 because of odd dimensions
        x = self._pad_if_needed(x, x3)
        x = torch.cat([x3, x], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = self._pad_if_needed(x, x1)
        x = torch.cat([x1, x], dim=1)
        x = self.dec1(x)

        return self.out_conv(x)

    
    #It just adds a tiny amount of padding (0s) to the upsampled tensor so:
    #both tensors become the same size    
    def _pad_if_needed(self, x, skip):
        
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)

        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x = F.pad(
                x,
                [
                    diff_w // 2, diff_w - diff_w // 2,
                    diff_h // 2, diff_h - diff_h // 2,
                    diff_d // 2, diff_d - diff_d // 2,
                ],
            )
        return x




# Simple test

# model = UNet3D(in_channels=1, num_classes=3)

# x = torch.randn(2, 1, 43, 59, 47)  # batch of 2
# y = model(x)

# print("Input shape:", x.shape)
# print("Output shape:", y.shape)
