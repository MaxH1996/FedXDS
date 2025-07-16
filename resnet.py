import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np
import torchvision

from functools import partial


class Model(nn.Module):
    def __init__(self, feature_dim=128, group_norm=False):
        super(Model, self).__init__()

        self.f = []
        for name, module in ResNet(
            BasicBlock, [1, 1, 1, 1], num_classes=10
        ).named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class resnet8(nn.Module):
    def __init__(self, num_classes=10, pretrained_path=None, group_norm=False):
        super(resnet8, self).__init__()

        # encoder
        self.f = Model(group_norm=group_norm).f
        # classifier
        self.classification_layer = nn.Linear(512, num_classes, bias=True)

        if pretrained_path:
            self.load_state_dict(
                torch.load(pretrained_path, map_location="cpu"), strict=False
            )

    def extract_features(self, x):
        return torch.flatten(self.f(x), start_dim=1)

    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out
