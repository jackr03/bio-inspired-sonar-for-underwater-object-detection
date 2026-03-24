from torch import nn, Tensor

from src.types.model_type import ModelType


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return self.pool(x)


class CNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, channels: list[int]):
        super().__init__()

        vgg_blocks = []
        for out_channels in channels:
            vgg_blocks.append(VGGBlock(in_channels, out_channels))
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*vgg_blocks)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.LazyLinear(out_features=num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        return self.classifier(x)

    @property
    def name(self) -> ModelType:
        return ModelType.CNN
