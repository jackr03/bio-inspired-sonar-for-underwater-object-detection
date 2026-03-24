from torch import nn, Tensor

from src.types.model_type import ModelType


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        nn.init.kaiming_normal_(self.block[0].weight, nonlinearity='relu')

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class CNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=8, kernel_size=5),
            ConvBlock(in_channels=8, out_channels=16, kernel_size=3),
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        return self.classifier(x)

    @property
    def name(self) -> ModelType:
        return ModelType.CNN
