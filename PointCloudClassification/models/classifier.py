import torch
import torch.nn as nn
from config import ClassifierConfig, device, np_float_t
from models.mlp import MLP
import utils
from sklearn.decomposition import PCA


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
        self.pca: PCA = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_chunk, n_point, n_in_channel)
        x = x.transpose(0, 1)
        # x.shape = (n_chunk, n_batch, n_point, n_in_channel)
        if self.config.type == "point_cloud":
            x = torch.stack([self.backbone(chunk) for chunk in x])
        elif self.config.type == "conv_3d":
            x = torch.stack([self.backbone(utils.voxelization(
                points=chunk,
                dim_size=self.config.dim_size,
                center=self.config.center,
                voxel_size=self.config.voxel_size
            )) for chunk in x])
        elif self.config.type == "pca":
            # TODO
            chunk_list = []
            for chunk in x:
                # chunk.shape = (n_batch, n_point, n_in_channel)
                chunk = utils.voxelization(
                    points=chunk,
                    dim_size=self.config.dim_size,
                    center=self.config.center,
                    voxel_size=self.config.voxel_size
                )
                # chunk.shape = (n_batch, 1 + n_feature, dim_size[0], dim_size[1], dim_size[2])
                chunk = chunk.reshape(chunk.shape[0], -1)
                # chunk.shape = (n_batch, (1 + n_feature) * dim_size[0] * dim_size[1] * dim_size[2])
                chunk = torch.from_numpy(self.pca.transform(chunk.cpu()).astype(np_float_t)).to(device)
                # chunk.shape = (n_batch, n_components)
                chunk = self.backbone(chunk)
                # chunk.shape = (n_batch, 1024)
                chunk_list.append(chunk)
            x = torch.stack(chunk_list)
        else:
            raise ValueError(f"Unknown type: {self.config.type}")
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
