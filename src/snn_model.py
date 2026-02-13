import snntorch as snn
import torch
from snntorch import surrogate, spikegen
from torch import nn, Tensor

class SNNConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, spike_grad):
        super().__init__()

        self.beta = torch.ones(1, out_channels, 1, 1) * 0.9

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = snn.Leaky(spike_grad=spike_grad, beta=self.beta, learn_beta=True)

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')

    def forward(self, x: Tensor, mem: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.bn(x)
        spk, mem = self.lif(x, mem)

        return spk, mem

    def init_leaky(self) -> Tensor:
        return self.lif.init_leaky()

# TODO: Implement other encoding types
class SNNAudioClassifier(nn.Module):
    def __init__(self, slope: int):
        super().__init__()

        # Encode here in the model itself for best performance
        self.encoder = lambda x: spikegen.rate(x, num_steps=25)

        spike_grad = surrogate.fast_sigmoid(slope)
        self.block1 = SNNConvBlock(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2, spike_grad=spike_grad)
        self.block2 = SNNConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, spike_grad=spike_grad)
        self.block3 = SNNConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, spike_grad=spike_grad)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )

        self.beta_out = torch.ones(10) * 0.9
        self.lif_out = snn.Leaky(spike_grad=spike_grad, beta=self.beta_out, learn_beta=True, output=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)

        mem1 = self.block1.init_leaky()
        mem2 = self.block2.init_leaky()
        mem3 = self.block3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []

        for t in range(x.shape[0]):
            spk1, mem1 = self.block1(x[t], mem1)
            spk2, mem2 = self.block2(spk1, mem2)
            spk3, mem3 = self.block3(spk2, mem3)

            logits = self.classifier(spk3)
            spk_out, mem_out = self.lif_out(logits, mem_out)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec, dim=0)
