import numpy as np

float_t = np.float32
device = "cuda"

config_radhar_dataset = {
    "n_frame_per_chunk": 10,   # 0.33 seconds
    "n_chunk_per_data": 6,     # 2 seconds
    "n_sample_per_chunk": 100, # 100 out of 240 samples per chunk on average
}
