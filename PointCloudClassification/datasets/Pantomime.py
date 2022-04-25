import os
import math
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transforms import Transform, Resampler, Translater, Scaler
from config import PantomimeDatasetConfig


class PantomimeDataset(Dataset):
    def __init__(self, config: PantomimeDatasetConfig):
        self.config = config
        # Transform
        self.transform = Transform()
        self.transform.resampler = Resampler(config.n_sample_per_chunk)
        self.transform.translater = Translater(config.translate_total_dist_std, config.translate_point_dist_std)
        self.transform.scaler = Scaler(config.scale_factor_std)
        # Load data
        raw_data = self.read_data(config.path)
        self.idx2label = config.actions
        self.label2idx = { key: idx for idx, key in enumerate(self.idx2label)}
        # Read data
        self.label: list[int] = []
        self.data: list[list[np.ndarray]] = []
        for env, angle, speed, action, data in raw_data:
            if env not in config.envs:
                continue
            if angle not in config.angles:
                continue
            if speed not in config.speeds:
                continue
            if action not in config.actions:
                continue
            self.label.append(self.label2idx[action])
            self.data.append(self.frame_divider(data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        return self.transform.transform(self.data[idx]), self.label[idx]

    # Return list of [environment, angle, speed, action, data]
    def read_data(self, root: str) -> list[tuple[str, int, str, int, list[np.ndarray]]]:
        list_dir = lambda path: (d for d in os.listdir(path) if not d.startswith("."))
        if os.path.exists(os.path.join(root, "data.pickle")):
            with open(os.path.join(root, "data.pickle"), "rb") as f:
                return pickle.load(f)
        result: list[tuple[str, int, str, int, list[np.ndarray]]] = []
        for exp in list_dir(root):
            for env in list_dir(os.path.join(root, exp)):
                for angle in list_dir(os.path.join(root, exp, env, "1")):
                    for speed in list_dir(os.path.join(root, exp, env, "1", angle)):
                        for id in list_dir(os.path.join(root, exp, env, "1", angle, speed)):
                            for action in list_dir(os.path.join(root, exp, env, "1", angle, speed, id)):
                                for file in list_dir(os.path.join(root, exp, env, "1", angle, speed, id, action)):
                                    with open(os.path.join(root, exp, env, "1", angle, speed, id, action, file), "rb") as f:
                                        data = pickle.load(f, encoding="latin1")
                                    result.append((env, int(angle), speed, int(action), data))
        with open(os.path.join(root, "data.pickle"), "wb") as f:
            pickle.dump(result, f)
        return result

    def show_data_statistics(self, all_data: list[tuple[str, int, str, int, list[np.ndarray]]]):
        env_dict, angle_dict, speed_dict, action_dict, frame_count_dict = {}, {}, {}, {}, {}
        total_point_count, total_frame_count = 0, 0
        for env, angle, speed, action, data in all_data:
            env_dict[env] = env_dict[env] + 1 if env in env_dict else 1
            angle_dict[angle] = angle_dict[angle] + 1 if angle in angle_dict else 1
            speed_dict[speed] = speed_dict[speed] + 1 if speed in speed_dict else 1
            action_dict[action] = action_dict[action] + 1 if action in action_dict else 1
            frame_count_dict[len(data)] = frame_count_dict[len(data)] + 1 if len(data) in frame_count_dict else 1
            total_point_count += sum(len(x) for x in data)
            total_frame_count += len(data)
        average_point_count = total_point_count / len(all_data)
        print(f"{env_dict = }")
        print(f"{angle_dict = }")
        print(f"{speed_dict = }")
        print(f"{action_dict = }")
        print(f"{frame_count_dict = }")
        print(f"{len(all_data) = }")
        print(f"{average_point_count = }")

    def frame_divider(self, data: list[np.ndarray]) -> list[np.ndarray]:
        n_frame_per_chunk = math.ceil(len(data) / self.config.n_chunk_per_data)
        chunks: list[np.ndarray] = []
        for idx in range(0, len(data), n_frame_per_chunk):
            chunks.append(np.vstack(data[idx:idx+n_frame_per_chunk])[:, :3])
        return chunks
