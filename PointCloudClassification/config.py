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


# min(x)= 0.00, max(x)=8.62
# min(y)=-5.20, max(y)=3.54
# min(z)=-8.18, max(z)=7.87
@dataclass
class RadHARDatasetConfig:
    name: str = "RadHAR"
    path: str = "data/RadHAR"
    # DataLoader
    batch_size: int = 256
    num_workers: int = 20
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


# min(x)= 0.00, max(x)=1.64
# min(y)=-1.59, max(y)=1.59
# min(z)=-1.63, max(z)=1.63
@dataclass
class PantomimeDatasetConfig:
    name: str = "Pantomime"
    path: str = "data/Pantomime"
    # DataLoader
    batch_size: int = 256
    num_workers: int = 20
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
class PointNetPPSSGConfig:
    @dataclass
    class SetAbstractionConfig:
        n_out_point: int
        ball_query_n_sample: int
        ball_query_radius: int
        mlp_layers: list[int]
    
    name: str = "PointNetPPSSG"
    set_abstractions: list[SetAbstractionConfig] = f([
        SetAbstractionConfig(
            n_out_point=50,
            ball_query_n_sample=8,
            ball_query_radius=0.2,
            mlp_layers=[8, 64, 128],
        ),
        SetAbstractionConfig(
            n_out_point=20,
            ball_query_n_sample=16,
            ball_query_radius=0.4,
            mlp_layers=[128, 128, 256],
        ),
    ])
    final_mlp_layers: list[int] = f([256, 512, 1024])


@dataclass
class PointNetPPMSGConfig:
    @dataclass
    class SetAbstractionConfig:
        n_out_point: int
        ball_query_n_sample: list[int]
        ball_query_radius: list[int]
        mlp_layers: list[list[int]]
    
    name: str = "PointNetPPMSG"
    set_abstractions: list[SetAbstractionConfig] = f([
        SetAbstractionConfig(
            n_out_point=50,
            ball_query_n_sample=[8, 16, 32],
            ball_query_radius=[0.2, 0.4, 0.6],
            mlp_layers=[[8, 64, 128], [8, 64, 128], [8, 64, 128]],
        ),
        SetAbstractionConfig(
            n_out_point=20,
            ball_query_n_sample=[16, 24, 32],
            ball_query_radius=[0.4, 0.8, 1.2],
            mlp_layers=[[3+128*3, 256, 512], [3+128*3, 256, 512], [3+128*3, 256, 512]],
        ),
    ])
    final_mlp_layers: list[int] = f([3+512*3, 512, 1024])


@dataclass
class Conv3DConfig:
    name: str = "Conv3D"
    n_channels: list[int] = f([6, 8])
    kernel_size: int = 3
    max_pooling_size: int = 2


@dataclass
class PCABackboneConfig:
    name: str = "PCABackbone"
    layers: list[int] = f([100, 128])


@dataclass
class ClassifierConfig:
    @dataclass
    class RNNConfig:
        name: str = "rnn"  # lstm, gru, rnn
        input_size: int = 1024
        hidden_size: int = 256
        num_layers: int = 1
        dropout: float = 0.0
        bidirectional: bool = False

    type: str = "point_cloud"  # point_cloud, conv_3d, pca
    # Voxelization
    dim_size: tuple[int, int, int] = f((5, 10, 10))
    center: tuple[float, float, float] = f((4.0, 0.0, 0.0))
    voxel_size: float = 1.0
    # PCA
    n_components: int = 100
    # Models
    rnn_config: RNNConfig = f(RNNConfig(
        name="rnn",
        input_size=1024,  # n_out_channel
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ))
    head_layers: list[int] = f([256, 64, 5])  # [hidden_size, 64, n_class]


@dataclass
class Config:
    n_epoch: int = 1
    test_interval: int = 1
    n_test_resample_time: int = 1
    dataset_config: Union[RadHARDatasetConfig, PantomimeDatasetConfig] = f(RadHARDatasetConfig())
    backbone_config: Union[PointNetConfig, PointNetPPSSGConfig] = f(PointNetPPSSGConfig())
    classifier_config: ClassifierConfig = f(ClassifierConfig())
