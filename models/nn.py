import torch
from torch import nn
import numpy as np

# ─────────────────────────────  helper  ────────────────────────────── #
def _calc_conv_out(conv_net: nn.Module, in_shape, device):
    with torch.no_grad():
        dummy = torch.zeros(1, *in_shape, device = device)
        return int(torch.prod(torch.tensor(conv_net(dummy).shape)).item())

# ─────────────────────────────── CNN  ──────────────────────────────── #
class CNN(nn.Module):
    def __init__(self, input_shape, n_actions,  freeze=False, device=torch.device("cpu")):

        super().__init__()

        self.device = device
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        ).to(self.device)

        conv_out = _calc_conv_out(self.conv_layers, input_shape, self.device)

        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        ).to(self.device)
        

        if freeze:
            for p in self.network.parameters():
                p.requires_grad = False
            self.eval()

    def forward(self, x):
        # uint8 → float32 in [0,1]
        if x.dtype != torch.float32:
            x = x.float().div_(255)
        return self.network(x)
