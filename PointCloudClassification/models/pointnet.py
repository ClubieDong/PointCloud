import torch
import torch.nn as nn
from models.mlp import MLP, MLPConv1D
from config import PointNetConfig, device, torch_float_t


class TNet(nn.Module):
    def __init__(self, mlp_conv_layers: list[int], mlp_layers: list[int]):
        assert mlp_conv_layers[0] ** 2 == mlp_layers[-1]
        assert mlp_conv_layers[-1] == mlp_layers[0]
        super().__init__()
        self.n_channel = mlp_conv_layers[0]
        self.mlp_conv = MLPConv1D(mlp_conv_layers)
        self.mlp = MLP(mlp_layers)
        self.I = torch.eye(self.n_channel, dtype=torch_float_t).to(device)

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
    def __init__(self, mlp_conv_layers: list[int], t_net_mlp_conv_layers: list[int], t_net_mlp_layers: list[int]):
        assert mlp_conv_layers[0] == t_net_mlp_conv_layers[0]
        super().__init__()
        self.t_net = TNet(t_net_mlp_conv_layers, t_net_mlp_layers)
        self.mlp_conv = MLPConv1D(mlp_conv_layers)

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
    def __init__(self, config: PointNetConfig):
        super().__init__()
        self.blocks = nn.ModuleList(
            PointNetBlock(cfg.mlp_conv_layers, cfg.t_net_mlp_conv_layers, cfg.t_net_mlp_layers)
            for cfg in config.blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_point, n_in_channel)
        for block in self.blocks:
            x = block(x)
        # x.shape = (n_batch, n_point, n_out_channel)
        x, _ = torch.max(x, dim=1)
        # x.shape = (n_batch, n_out_channel)
        return x
