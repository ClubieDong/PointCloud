import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layers: list[int], drop_rate: float = None):
        super().__init__()
        self.drop_rate = drop_rate
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        if drop_rate is not None:
            self.dos = nn.ModuleList()
        for n_in_feature, n_out_feature in zip(layers[:-2], layers[1:-1]):
            self.fcs.append(nn.Linear(n_in_feature, n_out_feature))
            self.bns.append(nn.BatchNorm1d(n_out_feature))
            if drop_rate is not None:
                self.dos.append(nn.Dropout(drop_rate))
        self.final_fc = nn.Linear(layers[-2], layers[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_in_channel)
        if self.drop_rate is None:
            for fc, bn in zip(self.fcs, self.bns):
                x = F.relu(bn(fc(x)))
        else:
            for fc, bn, do in zip(self.fcs, self.bns, self.dos):
                x = F.relu(bn(do(fc(x))))
        x = self.final_fc(x)
        # x.shape = (n_batch, n_out_channel)
        return x


class MLPConv1D(nn.Module):
    def __init__(self, layers: list[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for n_in_channel, n_out_channel in zip(layers[:-1], layers[1:]):
            self.convs.append(nn.Conv1d(n_in_channel, n_out_channel, 1))
            self.bns.append(nn.BatchNorm1d(n_out_channel))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (n_batch, n_point, n_in_channel)
        x = x.transpose(1, 2)
        # x.shape = (n_batch, n_in_channel, n_point)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
        # x.shape = (n_batch, n_out_channel, n_point)
        x = x.transpose(1, 2)
        # x.shape = (n_batch, n_point, n_out_channel)
        return x
