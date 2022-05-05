import torch
import torch.nn as nn
from config import device, torch_float_t, ClassifierConfig
from models.mlp import MLP
import pytorch3d.ops
from pytorch3d.structures import Pointclouds, Volumes


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, config: ClassifierConfig):
        super().__init__()
        self.config = config
        self.backbone = backbone
        if config.rnn_config.name == "rnn":
            rnn_module = nn.RNN
        elif config.rnn_config.name == "gru":
            rnn_module = nn.GRU
        elif config.rnn_config.name == "lstm":
            rnn_module = nn.LSTM
        else:
            raise ValueError(f"Unknown RNN name: {config.rnn_name}")
        self.rnn = rnn_module(
            input_size=config.rnn_config.input_size,
            hidden_size=config.rnn_config.hidden_size,
            num_layers=config.rnn_config.num_layers,
            dropout=config.rnn_config.dropout,
            bidirectional=config.rnn_config.bidirectional,
        )
        self.head = MLP(config.head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.type == "point_cloud":
            return self.point_cloud_forward(x)
        if self.config.type == "conv_3d":
            return self.conv_3d_forward(x)
        raise ValueError(f"Unknown type: {self.config.type}")

    def point_cloud_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_chunk, n_point, n_in_channel)
        x = x.transpose(0, 1)
        # x.shape = (n_chunk, n_batch, n_point, n_in_channel)
        x = torch.stack([self.backbone(chunk) for chunk in x])
        # x.shape = (n_chunk, n_batch, 1024)
        x, _ = self.rnn(x)
        # x.shape = (n_chunk, n_batch, 256)
        x = x.transpose(0, 1)
        # x.shape = (n_batch, n_chunk, 256)
        x = x[:, -1, :]
        # x.shape = (n_batch, 256)
        x = self.head(x)
        # x.shape = (n_batch, n_class)
        return x
    
    def conv_3d_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_chunk, n_point, n_in_channel)
        x = x.transpose(0, 1)
        # x.shape = (n_chunk, n_batch, n_point, n_in_channel)
        x = torch.stack([self.backbone(self.voxelization(chunk)) for chunk in x])
        # x.shape = (n_chunk, n_batch, 1024)
        x, _ = self.rnn(x)
        # x.shape = (n_chunk, n_batch, 256)
        x = x.transpose(0, 1)
        # x.shape = (n_batch, n_chunk, 256)
        x = x[:, -1, :]
        # x.shape = (n_batch, 256)
        x = self.head(x)
        # x.shape = (n_batch, n_class)
        return x

    def voxelization(self, points: torch.Tensor) -> torch.Tensor:
        # points.shape = (n_batch, n_point, 3 + n_feature)
        xyz = points[:, :, :3]
        features = points[:, :, 3:]
        point_cloud = Pointclouds(xyz, features=features)
        volume = Volumes(
            features = torch.zeros(points.shape[0], features.shape[2], *self.config.dim_size, dtype=torch_float_t, device=device),
            densities = torch.zeros(points.shape[0], 1, *self.config.dim_size, dtype=torch_float_t, device=device),
            volume_translation = self.config.center,
            voxel_size = 1.0,  # TODO
        )
        volume = pytorch3d.ops.add_pointclouds_to_volumes(point_cloud, volume)
        result = torch.hstack((volume.densities(), volume.features()))
        # result.shape = (n_batch, 1 + n_feature, dim_size[0], dim_size[1], dim_size[2])
        return result
