import torch
import torch.nn as nn
import pytorch3d.ops
from config import PointNetPPMSGConfig
from models.mlp import MLPConv1D


class SetAbstractionMSG(nn.Module):
    def __init__(self, n_out_point: int, ball_query_n_sample: list[int], ball_query_radius: list[float], mlp_layers: list[list[int]]):
        super().__init__()
        self.n_out_point = n_out_point
        self.ball_query_n_sample = ball_query_n_sample
        self.ball_query_radius = ball_query_radius
        self.mlp_convs = nn.ModuleList(MLPConv1D(layer) for layer in mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        # x.shape = (n_batch, n_in_point, n_in_channel)
        centroid, _ = pytorch3d.ops.sample_farthest_points(x[:, :, :3], K=self.n_out_point, random_start_point=True)
        # centroid.shape = (n_batch, n_out_point, 3)
        result: list[torch.Tensor] = []
        for n_sample, radius, mlp_conv in zip(self.ball_query_n_sample, self.ball_query_radius, self.mlp_convs):
            x = original_x
            x = self.group(x, centroid, n_sample, radius)
            # x.shape = (n_batch, n_out_point, n_sample, n_in_channel)
            x = x.reshape(x.shape[0], -1, x.shape[3])
            # x.shape = (n_batch, n_out_point * n_sample, n_in_channel)
            x = mlp_conv(x)
            # x.shape = (n_batch, n_out_point * n_sample, n_out_channel)
            x = x.reshape(x.shape[0], self.n_out_point, n_sample, x.shape[2])
            # x.shape = (n_batch, n_out_point, n_sample, n_out_channel)
            x, _ = torch.max(x, dim=2)
            # x.shape = (n_batch, n_out_point, n_out_channel)
            result.append(x)
        x = torch.cat([centroid, *result], dim=2)
        # x.shape = (n_batch, n_out_point, 3 + sum(n_out_channel))
        return x

    def group(self, x: torch.Tensor, centroid: torch.Tensor, n_sample: int, radius: float) -> torch.Tensor:
        # x.shape = (n_batch, n_in_point, n_channel)
        # centroid.shape = (n_batch, n_out_point, 3)
        _, group_idx, _ = pytorch3d.ops.ball_query(centroid, x[:, :, :3], K=n_sample, radius=radius, return_nn=False)
        # group_idx.shape = (n_batch, n_out_point, n_sample)
        group_idx = group_idx[:, :, :, None].expand(-1, -1, -1, x.shape[2])
        # group_idx.shape = (n_batch, n_out_point, n_sample, n_channel)
        x = x[:, :, None, :].expand(-1, -1, n_sample, -1)
        # x.shape = (n_batch, n_out_point, n_sample, n_channel)
        x = torch.gather(x, 1, group_idx)
        # x.shape = (n_batch, n_out_point, n_sample, n_channel)
        x[:, :, :, :3] -= centroid[:, :, None, :]
        # x.shape = (n_batch, n_out_point, n_sample, n_channel)
        return x


class PointNetPPMSG(nn.Module):
    def __init__(self, config: PointNetPPMSGConfig):
        super().__init__()
        self.sas = nn.ModuleList(
            SetAbstractionMSG(cfg.n_out_point, cfg.ball_query_n_sample, cfg.ball_query_radius, cfg.mlp_layers)
            for cfg in config.set_abstractions
        )
        self.final_mlp_conv = MLPConv1D(config.final_mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_in_point, n_in_channel)
        for sa in self.sas:
            x = sa(x)
        # x.shape = (n_batch, n_out_point, n_channel)
        x = self.final_mlp_conv(x)
        # x.shape = (n_batch, n_out_point, n_out_channel)
        x, _ = torch.max(x, dim=1)
        # x.shape = (n_batch, n_out_channel)
        return x
