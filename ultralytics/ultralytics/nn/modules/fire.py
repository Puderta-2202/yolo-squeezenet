import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class Fire(nn.Module):
    def __init__(self, c1, c2=None, *args):
        super().__init__()

        if c2 is None:
            c2 = c1

        squeeze = max(c2 // 8, 8)

        self.squeeze = Conv(c1, squeeze, 1, 1)
        self.expand1 = Conv(squeeze, c2 // 2, 1, 1)
        self.expand3 = Conv(squeeze, c2 // 2, 3, 1)

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat((self.expand1(x), self.expand3(x)), dim=1)
