import numpy as np
from config import float_t

class FrameDivider:
    def __init__(self, n_chunk: int, n_frame_per_chunk: int):
        self.n_chunk = n_chunk
        self.n_frame_per_chunk = n_frame_per_chunk
        self.n_frame = n_chunk * n_frame_per_chunk

    def divide(self, points: list[np.ndarray]) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        for idx in range(0, self.n_frame, self.n_frame_per_chunk):
            if idx >= len(points):
                print("[WARN] Frames are not enough to divide into chunks.")
                result.append(np.zeros((1, points[0].shape[1]), dtype=float_t))
            else:
                chunk = points[idx:idx+self.n_frame_per_chunk]
                result.append(np.concatenate(chunk))
        return result
