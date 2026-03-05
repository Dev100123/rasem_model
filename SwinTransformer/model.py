#Model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SwinUNet, self).__init__()
        # backbone
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            features_only=True
        )
        # feature maps
        self.feature_info = self.backbone.feature_info
        chs = [f['num_chs'] for f in self.feature_info]  # Ej: [C0, C1, C2, C3]

        # skip connections
        self.conv_decoder3 = nn.Sequential(
            nn.Conv2d(chs[3], 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_decoder2 = nn.Sequential(
            nn.Conv2d(chs[2] + 512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_decoder1 = nn.Sequential(
            nn.Conv2d(chs[1] + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_decoder0 = nn.Sequential(
            nn.Conv2d(chs[0] + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        # feature map de channels-last a channels-first
        features = [f.permute(0, 3, 1, 2) for f in features]
        f0, f1, f2, f3 = features

        d3 = self.conv_decoder3(f3)
        d3_up = F.interpolate(d3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d3_up, f2], dim=1)
        d2 = self.conv_decoder2(d2)

        d2_up = F.interpolate(d2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d2_up, f1], dim=1)
        d1 = self.conv_decoder1(d1)

        d1_up = F.interpolate(d1, size=f0.shape[2:], mode='bilinear', align_corners=False)
        d0 = torch.cat([d1_up, f0], dim=1)
        d0 = self.conv_decoder0(d0)

        out = self.final_conv(d0)
        # Upsample
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        return out
