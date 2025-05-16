import torch
from torch import nn
import numpy as np

# ─────────────────────────────  helper  ────────────────────────────── #
def _calc_conv_out(conv_net: nn.Module, in_shape, device):
    with torch.no_grad():
        dummy = torch.zeros(1, *in_shape, device = device)
        return int(np.prod(conv_net(dummy).shape))

# ─────────────────────────────── CNN  ──────────────────────────────── #
class CNN(nn.Module):
    def __init__(self, input_shape, n_actions,  freeze=False):
        super().__init__()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
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
        return self.network(x.to(self.device, non_blocking=True))

# ───────────────────────────── SmallCNN ────────────────────────────── #
class SmallCNN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        ).to(self.device)

        conv_out = _calc_conv_out(self.conv_layers, input_shape, self.device)

        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            ).to(self.device)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float().div_(255)
        return self.network(x.to(self.device, non_blocking=True))