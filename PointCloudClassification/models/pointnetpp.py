import torch
import torch.nn as nn
import pytorch3d.ops
from config import PointNetPPConfig
from models.mlp import MLPConv1D


class SetAbstraction(nn.Module):
    def __init__(self, n_out_point: int, ball_query_n_sample: int, ball_query_radius: float, mlp_layers: list[int]):
        super().__init__()
        self.n_out_point = n_out_point
        self.ball_query_n_sample = ball_query_n_sample
        self.ball_query_radius = ball_query_radius
        self.mlp_conv = MLPConv1D(mlp_layers[:-1] + [mlp_layers[-1] - 3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_in_point, n_in_channel)
        centroid, x = self.sample_and_group(x)
        # centroid.shape = (n_batch, n_out_point, 3)
        # x.shape = (n_batch, n_out_point, ball_query_n_sample, n_in_channel)
        x = x.reshape(x.shape[0], -1, x.shape[3])
        # x.shape = (n_batch, n_out_point * ball_query_n_sample, n_in_channel)
        x = self.mlp_conv(x)
        # x.shape = (n_batch, n_out_point * ball_query_n_sample, n_out_channel)
        x = x.reshape(x.shape[0], self.n_out_point, self.ball_query_n_sample, x.shape[2])
        # x.shape = (n_batch, n_out_point, ball_query_n_sample, n_out_channel)
        x, _ = torch.max(x, dim=2)
        # x.shape = (n_batch, n_out_point, n_out_channel)
        x = torch.cat([centroid, x], dim=2)
        # x.shape = (n_batch, n_out_point, 3 + n_out_channel)
        return x

    def sample_and_group(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x.shape = (n_batch, n_in_point, n_channel)
        pos = x[:, :, :3]
        # pos.shape = (n_batch, n_in_point, 3)
        centroid, _ = pytorch3d.ops.sample_farthest_points(pos, K=self.n_out_point, random_start_point=True)
        # centroid.shape = (n_batch, n_out_point, 3)
        _, group_idx, _ = pytorch3d.ops.ball_query(centroid, pos, K=self.ball_query_n_sample, radius=self.ball_query_radius, return_nn=False)
        # group_idx.shape = (n_batch, n_out_point, ball_query_n_sample)
        group_idx = group_idx[:, :, :, None].expand(-1, -1, -1, x.shape[2])
        # group_idx.shape = (n_batch, n_out_point, ball_query_n_sample, n_channel)
        x = x[:, :, None, :].expand(-1, -1, self.ball_query_n_sample, -1)
        # x.shape = (n_batch, n_out_point, ball_query_n_sample, n_channel)
        x = torch.gather(x, 1, group_idx)
        # x.shape = (n_batch, n_out_point, ball_query_n_sample, n_channel)
        x[:, :, :, :3] -= centroid[:, :, None, :]
        # x.shape = (n_batch, n_out_point, ball_query_n_sample, n_channel)
        return centroid, x


class PointNetPP(nn.Module):
    def __init__(self, config: PointNetPPConfig):
        super().__init__()
        self.sas = nn.ModuleList(
            SetAbstraction(cfg.n_out_point, cfg.ball_query_n_sample, cfg.ball_query_radius, cfg.mlp_layers)
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
