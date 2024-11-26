import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, (3, 3), (1, 1), 1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_sample(x)