import torch
import torch.nn as nn
from models.pointnet import PointNet
from models.pointnetpp import PointNetPP
from models.mlp import MLP

class Classifier(nn.Module):
    def __init__(self, n_in_channel: int, n_chunk: int, n_class: int):
        super().__init__()
        self.backbone = PointNet(n_in_channel)
        # self.backbone = PointNetPP(
        #     n_out_point_list=[50, 20],
        #     ball_query_n_sample_list=[8, 16],
        #     ball_query_radius_list=[1000, 1000],
        #     mlp_layers_list=[[8, 64, 128], [128, 128, 256]],
        #     final_mlp_layers=[256, 512, 1024],
        # )
        self.rnn = nn.LSTM(input_size=1024, hidden_size=256, num_layers=2)
        self.head = MLP(layers=[n_chunk * 256, 64, n_class])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_chunk, n_point, n_in_channel)
        x = x.transpose(0, 1)
        # x.shape = (n_chunk, n_batch, n_point, n_in_channel)
        x = torch.stack([self.backbone(chunk) for chunk in x])
        # x.shape = (n_chunk, n_batch, 1024)
        x, _ = self.rnn(x)
        # x.shape = (n_chunk, n_batch, 256)
        x = x.transpose(0, 1)
        # x.shape = (n_batch, n_chunk, 256)
        x = x.reshape(x.shape[0], -1)
        # x.shape = (n_batch, n_chunk * 256)
        x = self.head(x)
        # x.shape = (n_batch, n_class)
        return x
