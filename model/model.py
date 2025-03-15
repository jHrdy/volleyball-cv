import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleSegHead(nn.Module):
    def __init__(self, in_channels=576, mid_channels=64, num_classes=4):
        super(SimpleSegHead, self).__init__()
        # Reduce channels from 576 to 64
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(mid_channels)  
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x.size: [batch_size, 576, 7, 7]
        x = self.conv1(x)  
        x = self.batch_norm(x)
        x = self.relu(x)
        # Upsample to original size (e.g., 224x224)
        x = F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=True)  
        x = self.conv2(x)  # [batch_size, 4, 224, 224]
        return x

class SegmentationModel(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super(SegmentationModel, self).__init__()
        if encoder is None:
            self.encoder = models.mobilenet_v3_small(weights='DEFAULT').features
        else:
            self.encoder = encoder
        self.segmentation_head = SimpleSegHead(in_channels=576, num_classes=num_classes)

    def forward(self, x, labels=None):
        x = self.encoder(x)  # [batch_size, 576, 7, 7]
        x = self.segmentation_head(x)      # [batch_size, 4, 224, 224]

        # if labels is not None:
        #     loss = F.cross_entropy(x, labels)
        #     return loss, x
        return x