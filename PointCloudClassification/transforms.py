import numpy as np
from config import float_t
import torch


class Resampler:
    def __init__(self, n_points: int):
        self.n_points = n_points
        self.rng = np.random.default_rng()

    def resample(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] >= self.n_points:
            return self.rng.choice(points, self.n_points, replace=False)
        # print(f"[WARN] Points are not enough to resample, {self.n_points = }, {len(points) = }")
        # TODO: try using padding instead of duplicating
        return np.concatenate([points, self.rng.choice(points, self.n_points - points.shape[0], replace=True)])


class Translater:
    def __init__(self, total_dist_std: float, point_dist_std: float):
        self.total_dist_std = total_dist_std
        self.point_dist_std = point_dist_std
        self.rng = np.random.default_rng()
    
    def translate(self, points: np.ndarray) -> np.ndarray:
        total_dist = self.rng.normal(0, self.total_dist_std, size=3)
        point_dist = self.rng.normal(0, self.point_dist_std, size=(points.shape[0], 3))
        points[:, :3] += total_dist
        points[:, :3] += point_dist
        return points


class Scaler:
    def __init__(self, factor_std: float):
        self.factor_std = factor_std
        self.rng = np.random.default_rng()

    def scale(self, points: np.ndarray) -> np.ndarray:
        factor = self.rng.normal(1, self.factor_std)
        points[:, :3] *= factor
        return points


class Transform:
    def __init__(self):
        self.resampler: Resampler = None
        self.translater: Translater = None
        self.scaler: Scaler = None

    def transform(self, points: list[np.ndarray]) -> torch.Tensor:
        points = [x.copy() for x in points]
        if self.resampler is not None:
            for idx in range(len(points)):
                points[idx] = self.resampler.resample(points[idx])
        if self.translater is not None:
            for idx in range(len(points)):
                points[idx] = self.translater.translate(points[idx])
        if self.scaler is not None:
            for idx in range(len(points)):
                points[idx] = self.scaler.scale(points[idx])
        points = np.array(points, dtype=float_t)
        points = torch.tensor(points)
        return points
