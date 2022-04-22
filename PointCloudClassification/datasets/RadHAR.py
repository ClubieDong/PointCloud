import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transforms import Transform, Resampler, Translater, Scaler
from config import float_t, config_radhar_dataset


class RadHARDataset(Dataset):
    def __init__(self, root: str):
        self.transform = Transform()
        self.transform.resampler = Resampler(config_radhar_dataset["n_sample_per_chunk"])
        self.transform.translater = Translater(config_radhar_dataset["translate_total_dist_std"], config_radhar_dataset["translate_point_dist_std"])
        self.transform.scaler = Scaler(config_radhar_dataset["scale_factor_std"])
        raw_data = self.read_all_data(root)
        self.idx2label = raw_data.keys()
        self.label2idx = { key: idx for idx, key in enumerate(self.idx2label)}
        self.label: list[int] = []
        self.data: list[list[np.ndarray]] = []
        for dir, data_list in raw_data.items():
            for data in data_list:
                divided_data = self.frame_divider(data)
                self.label += [self.label2idx[dir]] * len(divided_data)
                self.data += divided_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.transform.transform(self.data[idx]), self.label[idx]

    def read_file(self, path: str) -> list[np.ndarray]:
        points: list[tuple[int, list[float]]] = []
        with open(path, "r") as f:
            read = lambda: f.readline().split(": ")[1]
            while True:
                if f.readline() == "":       # header:
                    break
                f.readline()                 # seq: 3026037
                f.readline()                 # stamp:
                f.readline()                 # secs: 1561686032
                f.readline()                 # nsecs: 152442104
                f.readline()                 # frame_id: "ti_mmwave"
                point_id = int(read())       # point_id: 0
                x = float(read())            # x: 0.01171875
                y = float(read())            # y: -0.0390625
                z = float(read())            # z: -0.0078125
                range = float(read())        # range: 0.0435769706964
                velocity = float(read())     # velocity: 0.0
                doppler_bin = float(read())  # doppler_bin: 8
                bearing = float(read())      # bearing: -73.3007583618
                intensity = float(read())    # intensity: 30.7445068359
                f.readline()                 # ---
                points.append((point_id, [x, y, z, range, velocity, doppler_bin, bearing, intensity]))
        data: list[np.ndarray] = []
        frame: list[list[float]] = []
        for idx, (_, point) in enumerate(points):
            frame.append(point)
            if idx == len(points) - 1 or points[idx + 1][0] == 0:
                data.append(np.array(frame, dtype=float_t))
                frame = []
        return data

    def read_all_data(self, root: str) -> dict[str, list[list[np.ndarray]]]:
        list_dir = lambda path: (d for d in os.listdir(path) if not d.startswith("."))
        if os.path.exists(os.path.join(root, "data.pickle")):
            with open(os.path.join(root, "data.pickle"), "rb") as f:
                return pickle.load(f)
        result: dict[str, list[list[np.ndarray]]] = {}
        for dir in list_dir(root):
            data: list[list[np.ndarray]] = []
            for file in list_dir(os.path.join(root, dir)):
                data.append(self.read_file(os.path.join(root, dir, file)))
            result[dir] = data
        with open(os.path.join(root, "data.pickle"), "wb") as f:
            pickle.dump(result, f)
        return result

    def frame_divider(self, data: list[np.ndarray]) -> list[list[np.ndarray]]:
        n_frame_per_chunk = config_radhar_dataset["n_frame_per_chunk"]
        n_chunk_per_data = config_radhar_dataset["n_chunk_per_data"]
        chunks: list[np.ndarray] = []
        for idx in range(0, len(data), n_frame_per_chunk):
            if idx + n_frame_per_chunk <= len(data):
                chunks.append(np.vstack(data[idx:idx+n_frame_per_chunk]))
        divided_data: list[list[np.ndarray]] = []
        for idx in range(0, len(chunks)):
            if idx + n_chunk_per_data <= len(chunks):
                divided_data.append(chunks[idx:idx+n_chunk_per_data])
        return divided_data
