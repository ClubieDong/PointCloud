import torch
import torch.nn as nn
from models.mlp import MLP, MLPConv1D


class TNet(nn.Module):
    def __init__(self, n_channel: int):
        super().__init__()
        self.n_channel = n_channel
        self.mlp_conv = MLPConv1D(layers=[n_channel, 64, 128, 1024])
        self.mlp = MLP(layers=[1024, 512, 256, n_channel*n_channel])
        self.I = torch.eye(n_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_point, n_channel)
        x = self.mlp_conv(x)
        # x.shape = (n_batch, n_point, 1024)
        x, _ = torch.max(x, dim=1)
        # x.shape = (n_batch, 1024)
        x = self.mlp(x)
        # x.shape = (n_batch, n_channel*n_channel)
        x = x.view(-1, self.n_channel, self.n_channel)
        # x.shape = (n_batch, n_channel, n_channel)
        x += self.I.expand_as(x)
        # x.shape = (n_batch, n_channel, n_channel)
        return x


class PointNetBlock(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        self.t_net = TNet(n_channel=layers[0])
        self.mlp_conv = MLPConv1D(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_point, n_in_channel)
        trans = self.t_net(x)
        # trans.shape = (n_batch, n_in_channel, n_in_channel)
        x = x.transpose(1, 2)
        # x.shape = (n_batch, n_in_channel, n_point)
        x = torch.bmm(trans, x)
        # x.shape = (n_batch, n_in_channel, n_point)
        x = x.transpose(1, 2)
        # x.shape = (n_batch, n_point, n_in_channel)
        x = self.mlp_conv(x)
        # x.shape = (n_batch, n_point, n_out_channel)
        return x


class PointNet(nn.Module):
    def __init__(self, n_in_channel: int):
        super().__init__()
        self.block1 = PointNetBlock(layers=[n_in_channel, 64])
        self.block2 = PointNetBlock(layers=[64, 128, 1024])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_point, n_in_channel)
        x = self.block1(x)
        # x.shape = (n_batch, n_point, 64)
        x = self.block2(x)
        # x.shape = (n_batch, n_point, 1024)
        x, _ = torch.max(x, dim=1)
        # x.shape = (n_batch, 1024)
        return x
