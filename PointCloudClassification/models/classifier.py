import torch
import torch.nn as nn
from config import ClassifierConfig
from models.mlp import MLP


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, config: ClassifierConfig):
        super().__init__()
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
