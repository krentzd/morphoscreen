from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch

import torch.nn as nn

model_param_dict = {
    'b0': (32, 1280),
    'b4': (48, 1792)
}

class ContrastiveClassifierAvgPoolCNN(nn.Module):
    def __init__(self, model='b0', num_channels=3, num_features=128, num_classes=2, pretrained=False, dropout=0, n_crops=9):
        super().__init__()

        if model=='b0':
            self.backbone = models.efficientnet_b0()
        elif model=='b1':
            self.backbone = models.efficientnet_b1()
        elif model=='b2':
            self.backbone = models.efficientnet_b2()
        elif model=='b3':
            self.backbone = models.efficientnet_b3()
        elif model=='b4':
            self.backbone = models.efficientnet_b4()
        elif model=='b5':
            self.backbone = models.efficientnet_b5()
        elif model=='b6':
            self.backbone = models.efficientnet_b6()
        elif model=='b7':
            self.backbone = models.efficientnet_b7()

        if pretrained:
            print('Using pretrained model')
            weights = torch.load(f'../../../EfficientNetWeights/efficientnet_{model}.pth')
            self.backbone.load_state_dict(weights)

        self.backbone.features[0][0] = nn.Conv2d(num_channels, model_param_dict[model][0], 3, stride=2, padding=1)
        # GAP
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.avg_pool_2 = nn.AvgPool1d(kernel_size=n_crops) #n_views

        self.fc_head = nn.Linear(model_param_dict[model][1], num_features)

        self.clsf_head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # Take input x composed of n tiles
        bs, ncrops, c, h, w = x.size()
        # Reshape data to predict on all crops in one pass
        x = x.view(-1, c, h, w)
        # Pass n tiles through network
        out = self.backbone.features(x)
        out = self.avg_pool(out).view(bs, ncrops, -1)

        out = self.avg_pool_2(out.permute((0, 2, 1))).view(bs, -1)

        # tanh activation function to constrain feature values between -1 and 1
        feat_vec = F.tanh(self.fc_head(out))

        out = self.clsf_head(feat_vec)

        return out, feat_vec
