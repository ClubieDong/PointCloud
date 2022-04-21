import numpy as np

float_t = np.float32
device = "cuda"

config_radhar_dataset = {
    "n_chunk": 10,
    "n_frame_per_chunk": 60,
    "n_sample_per_chunk": 100,
}
