import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Conv3DConfig


class Conv3D(nn.Module):
    def __init__(self, config: Conv3DConfig):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv3d(n_in_channel, n_out_channel, kernel_size=config.kernel_size)
            for n_in_channel, n_out_channel
            in zip(config.n_channels[:-1], config.n_channels[1:])
        )
        self.bns = nn.ModuleList(
            nn.BatchNorm3d(n_out_channel)
            for n_out_channel 
            in config.n_channels[1:]
        )
        self.mp = nn.MaxPool3d(config.max_pooling_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, n_in_channel, x_in_size, y_in_size, z_in_size)
        for conv, bn in zip(self.convs, self.bns):
            x = self.mp(F.relu(bn(conv(x))))
        # x.shape = (batch_size, n_out_channel, x_out_size, y_out_size, z_out_size)
        x = x.reshape(x.shape[0], -1)
        # x.shape = (batch_size, n_out_channel * x_out_size * y_out_size * z_out_size)
        return x
