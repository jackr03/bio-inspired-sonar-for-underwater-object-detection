import torch
from torch import nn


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second convolution block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third convolution block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Wrap convolution blocks together
        self.conv_layers = nn.Sequential(*conv_layers)

        # Global pooling
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Linear classifier
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pooling(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)