from abc import ABC, abstractmethod
import numpy as np


class Resampler(ABC):
    @abstractmethod
    def resample(self, points: np.ndarray) -> np.ndarray:
        pass


class RandomResampler(Resampler):
    def __init__(self, n_points: int):
        self.n_points = n_points
        self.rng = np.random.default_rng()

    def resample(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] >= self.n_points:
            return self.rng.choice(points, self.n_points, replace=False)
        print("[WARN] Points are not enough to resample.")
        return np.concatenate([points, self.rng.choice(points, self.n_points - points.shape[0], replace=True)])
