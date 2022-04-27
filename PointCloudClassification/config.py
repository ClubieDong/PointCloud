from typing import Union
import numpy as np
import torch
import copy
from dataclasses import dataclass, field

def f(x):
    return field(default_factory=lambda: copy.deepcopy(x))

np_float_t = np.float32
torch_float_t = torch.float32
device = "cuda"

@dataclass
class RadHARDatasetConfig:
    name: str = "RadHAR"
    path: str = "data/RadHAR"
    # DataLoader
    batch_size: int = 128
    num_workers: int = 0
    # Input channels
    include_range: bool = True
    include_velocity: bool = True
    include_doppler_bin: bool = True
    include_bearing: bool = True
    include_intensity: bool = True
    # Batching
    n_frame_per_chunk: int = 10    # 0.33 seconds
    n_chunk_per_data: int = 6      # 2 seconds
    n_sample_per_chunk: int = 100  # 100 out of 240 samples per chunk on average
    # Tranforms
    translate_total_dist_std: float = 0.1   # X += N(0, 0.1)
    translate_point_dist_std: float = 0.01  # Xi += N(0, 0.01)  (i = 0, 1, 2...)
    scale_factor_std: float = 0.1           # X *= N(1, 0.1)


@dataclass
class PantomimeDatasetConfig:
    name: str = "Pantomime"
    path: str = "data/Pantomime"
    # DataLoader
    batch_size: int = 512
    num_workers: int = 0
    # Labels
    envs: list[str] = f(["office", "open", "industrial", "multi_people", "occluded", "restaurant"])
    angles: list[int] = f([-45, -30, -15, 0, 15, 30, 45])
    speeds: list[str] = f(["slow", "normal", "fast"])
    actions: list[int] = f([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    # Batching
    n_chunk_per_data: int = 4     # 2 seconds
    n_sample_per_chunk: int = 40  # 40 out of 55 samples per chunk on average
    # Tranforms
    translate_total_dist_std: float = 0.1   # X += N(0, 0.1)
    translate_point_dist_std: float = 0.01  # Xi += N(0, 0.01)  (i = 0, 1, 2...)
    scale_factor_std: float = 0.1           # X *= N(1, 0.1)


@dataclass
class PointNetConfig:
    @dataclass
    class PointNetBlockConfig:
        mlp_conv_layers: list[int]
        t_net_mlp_conv_layers: list[int]
        t_net_mlp_layers: list[int]
    
    name: str = "PointNet"
    blocks: list[PointNetBlockConfig] = f([
        PointNetBlockConfig(                           # n_in_channel = 8
            mlp_conv_layers=[8, 64],                   # [n_in_channel, 64]
            t_net_mlp_conv_layers=[8, 64, 128, 1024],  # [n_in_channel, 64, 128, 1024]
            t_net_mlp_layers=[1024, 512, 256, 8*8],    # [1024, 512, 256, n_in_channel*n_in_channel]
        ),
        PointNetBlockConfig(                            # n_in_channel = 64
            mlp_conv_layers=[64, 128, 1024],            # [n_in_channel, 128, 1024]
            t_net_mlp_conv_layers=[64, 64, 128, 1024],  # [n_in_channel, 64, 128, 1024]
            t_net_mlp_layers=[1024, 512, 256, 64*64],   # [1024, 512, 256, n_in_channel*n_in_channel]
        ),
    ])


@dataclass
class PointNetPPConfig:
    @dataclass
    class SetAbstractionConfig:
        n_out_point: int
        ball_query_n_sample: int
        ball_query_radius: int
        mlp_layers: list[int]
    
    name: str = "PointNetPP"
    set_abstractions: list[SetAbstractionConfig] = f([
        SetAbstractionConfig(
            n_out_point=50,
            ball_query_n_sample=8,
            ball_query_radius=1000,  # TODO
            mlp_layers=[8, 64, 128],
        ),
        SetAbstractionConfig(
            n_out_point=20,
            ball_query_n_sample=16,
            ball_query_radius=1000,  # TODO
            mlp_layers=[128, 128, 256],
        ),
    ])
    final_mlp_layers: list[int] = f([256, 512, 1024])


@dataclass
class ClassifierConfig:
    @dataclass
    class LLVMConfig:
        input_size: int
        hidden_size: int
        num_layers: int
        dropout: float
        bidirectional: bool
    
    rnn_name: str = "lstm"  # lstm, gru, rnn
    rnn_config: LLVMConfig = f(LLVMConfig(
        input_size=1024,  # n_out_channel
        hidden_size=256,
        num_layers=2,
        dropout=0.0,
        bidirectional=False,
    ))
    head_layers: list[int] = f([6 * 256, 64, 5])  # [n_chunk * hidden_size, 64, n_class]


@dataclass
class TrainConfig:
    n_epoch: int = 1
    test_interval: int = 1
    dataset_config: Union[RadHARDatasetConfig, PantomimeDatasetConfig] = f(RadHARDatasetConfig())
    backbone_config: Union[PointNetConfig, PointNetPPConfig] = f(PointNetPPConfig())
    classifier_config: ClassifierConfig = f(ClassifierConfig())
