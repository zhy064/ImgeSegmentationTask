import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from attention_module import SEBlock

class ASPP_with_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP_with_Attention, self).__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    SEBlock(out_channels)
                )
            )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SEBlock(out_channels)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(rates)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        x1x1 = self.conv1x1(x)
        atrous_outs = [conv(x) for conv in self.atrous_convs]
        img_pool = F.interpolate(self.image_pool(x), size=size, mode='bilinear', align_corners=True)
        out = torch.cat([x1x1, *atrous_outs, img_pool], dim=1)
        return self.project(out)


class DeepLabV3Plus_with_Attention(nn.Module):
    def __init__(self, num_classes, backbone='resnet101', pretrained=True, rates=[6, 12, 18]):
        super(DeepLabV3Plus_with_Attention, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)
        low_level_channels = 256
        high_level_channels = 2048
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        self.low_level_features = nn.Sequential(*list(self.backbone.children())[:5])
        self.high_level_features = nn.Sequential(*list(self.backbone.children())[5:])

        self.aspp = ASPP_with_Attention(high_level_channels, 256, rates)

        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        low_level_feat = self.low_level_features(x)
        high_level_feat = self.high_level_features(low_level_feat)
        
        aspp_out = self.aspp(high_level_feat)

        aspp_out_upsampled = F.interpolate(aspp_out, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        low_level_processed = self.decoder_conv1(low_level_feat)
        
        concat_feat = torch.cat([aspp_out_upsampled, low_level_processed], dim=1)
        
        decoder_out = self.decoder_conv2(concat_feat)
        logits = self.classifier(decoder_out)
        
        output = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=True)
        return output