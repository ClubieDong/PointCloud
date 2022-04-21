from transforms.resampler import Resampler
import numpy as np
from config import float_t
import torch

class Transform:
    def __init__(self):
        self.resampler: Resampler = None

    def transform(self, points: list[np.ndarray]) -> torch.Tensor:
        if self.resampler is not None:
            for idx in range(len(points)):
                points[idx] = self.resampler.resample(points[idx])
        points = np.array(points, dtype=float_t)
        points = torch.tensor(points)
        return points
