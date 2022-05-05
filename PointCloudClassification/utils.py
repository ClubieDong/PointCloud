import numpy as np
import torch
import pytorch3d.ops
from pytorch3d.structures import Pointclouds, Volumes
from config import device, torch_float_t, np_float_t
from tqdm import tqdm
from sklearn.decomposition import PCA


def voxelization(points: torch.Tensor, dim_size: tuple[int, int, int], center: tuple[float, float, float], voxel_size: float) -> torch.Tensor:
    # points.shape = (n_batch, n_point, 3 + n_feature)
    point_cloud = Pointclouds(
        points=points[:, :, :3],
        features=points[:, :, 3:],
    )
    volume = Volumes(
        features = torch.zeros(points.shape[0], points.shape[2]-3, *dim_size, dtype=torch_float_t, device=device),
        densities = torch.zeros(points.shape[0], 1, *dim_size, dtype=torch_float_t, device=device),
        volume_translation = center,
        voxel_size = voxel_size,
    )
    volume = pytorch3d.ops.add_pointclouds_to_volumes(point_cloud, volume)
    result = torch.hstack((volume.densities(), volume.features()))
    # result.shape = (n_batch, 1 + n_feature, dim_size[0], dim_size[1], dim_size[2])
    return result


def calc_pca(raw_data: list[list[np.ndarray]], n_components: int, dim_size: tuple[int, int, int], center: tuple[float, float, float], voxel_size: float) -> PCA:
    print("Calculating PCA...")
    batch_size = 4096
    xyz = [torch.from_numpy(y[:, :3]).to(device) for x in raw_data for y in x]
    features = [torch.from_numpy(y[:, 3:]).to(device) for x in raw_data for y in x]
    results: list[torch.Tensor] = []
    for idx in tqdm(range(0, len(xyz), batch_size)):
        point_cloud = Pointclouds(
            points=xyz[idx:idx+batch_size],
            features=features[idx:idx+batch_size],
        )
        volume = Volumes(
            features = torch.zeros(len(xyz[idx:idx+batch_size]), features[0].shape[1], *dim_size, dtype=torch_float_t, device=device),
            densities = torch.zeros(len(xyz[idx:idx+batch_size]), 1, *dim_size, dtype=torch_float_t, device=device),
            volume_translation = center,
            voxel_size = voxel_size,
        )
        volume = pytorch3d.ops.add_pointclouds_to_volumes(point_cloud, volume)
        results.append(torch.hstack((volume.densities(), volume.features())))
    results = torch.vstack(results)
    results = results.cpu().numpy()
    results = results.reshape(results.shape[0], -1)
    pca = PCA(n_components)
    pca.fit(results)
    print(f"PCA explaned ratio: {sum(pca.explained_variance_ratio_)}\n")
    return pca
