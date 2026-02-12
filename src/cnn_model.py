from torch import nn, Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        nn.init.kaiming_normal_(self.block[0].weight, nonlinearity='relu')

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class CNNAudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2),
            ConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        return self.classifier(x)