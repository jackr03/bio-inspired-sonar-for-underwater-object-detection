import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn, Tensor

class SNN1dConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, beta_init: float, spike_grad):
        super().__init__()

        beta = torch.ones(1, out_channels, 1) * beta_init

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lif = snn.Leaky(spike_grad=spike_grad, beta=beta, learn_beta=True)

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')

    def forward(self, x: Tensor, mem: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv(x)
        x = self.pool(x)
        spk, mem = self.lif(x, mem)

        return spk, mem

    def init_leaky(self) -> Tensor:
        return self.lif.init_leaky()

class SNN1DEncodedClassifier(nn.Module):

    NAME = 'snn_1d_encoded'

    def __init__(self, beta_init: float, slope: int):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope)
        self.block1 = SNN1dConvBlock(in_channels=2, out_channels=8, kernel_size=5, beta_init=beta_init, spike_grad=spike_grad)
        self.block2 = SNN1dConvBlock(in_channels=8, out_channels=16, kernel_size=3, beta_init=beta_init, spike_grad=spike_grad)
        self.block3 = SNN1dConvBlock(in_channels=16, out_channels=32, kernel_size=3, beta_init=beta_init, spike_grad=spike_grad)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 10)
        )

        beta_out = torch.ones(10) * beta_init
        self.lif_out = snn.Leaky(spike_grad=spike_grad, beta=beta_out, learn_beta=True, output=True)

    def forward(self, x: Tensor) -> Tensor:
        # Expected shape of tensor is [Batch, Channel, Time, Mel Bins]
        mem1 = self.block1.init_leaky()
        mem2 = self.block2.init_leaky()
        mem3 = self.block3.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_rec = []
        for t in range(x.shape[2]):
            x_t = x[:, :, t, :]

            spk1, mem1 = self.block1(x_t, mem1)
            spk2, mem2 = self.block2(spk1, mem2)
            spk3, mem3 = self.block3(spk2, mem3)

            logits = self.classifier(spk3)
            spk_out, mem_out = self.lif_out(logits, mem_out)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec, dim=0)
