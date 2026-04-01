import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn, Tensor

from src.types.model_type import ModelType


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, beta_init: float, spike_grad):
        super().__init__()

        beta = torch.ones(1, out_channels, 1) * beta_init
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(spike_grad=spike_grad, beta=beta, learn_beta=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(spike_grad=spike_grad, beta=beta, learn_beta=True)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: Tensor, mem1: Tensor, mem2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        spk, mem1 = self.lif1(x, mem1)
        x = self.conv2(spk)
        spk, mem2 = self.lif2(x, mem2)
        return self.pool(spk), mem1, mem2

    def init_leaky(self) -> tuple[Tensor, Tensor]:
        return self.lif1.init_leaky(), self.lif2.init_leaky()


class SNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: list[int],
        dropout: float = 0.5,
        beta_init: float = 0.5,
        slope: int = 25,
    ):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope)

        self.blocks = nn.ModuleList()
        for out_channels in channels:
            self.blocks.append(VGGBlock(in_channels, out_channels, beta_init, spike_grad))
            in_channels = out_channels

        self.classifier = nn.Sequential(
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.LazyLinear(out_features=num_classes),
        )

        beta_out = torch.ones(num_classes) * beta_init
        self.lif_out = snn.Leaky(spike_grad=spike_grad, beta=beta_out, learn_beta=True, output=True)

    def forward(self, x: Tensor) -> Tensor:
        block_mems = [block.init_leaky() for block in self.blocks]
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        for t in range(x.shape[2]):
            x_t = x[:, :, t, :]

            for i, block in enumerate(self.blocks):
                mem1, mem2 = block_mems[i]
                x_t, mem1, mem2 = block(x_t, mem1, mem2)
                block_mems[i] = (mem1, mem2)

            logits = self.classifier(x_t)
            spk_out, mem_out = self.lif_out(logits, mem_out)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec, dim=0)

    @property
    def name(self) -> ModelType:
        return ModelType.SNN
