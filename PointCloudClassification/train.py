from datasets.RadHAR import RadHARDataset
from transforms.transforms import Transform
from transforms.resampler import RandomResampler
from torch.utils.data import DataLoader
from models.pointnet import PointNet
from config import config_radhar_dataset

transfrom = Transform()
transfrom.resampler = RandomResampler(config_radhar_dataset["n_sample_per_chunk"])
dataset = DataLoader(RadHARDataset("RadHAR/Data/Test", transfrom), batch_size=4, shuffle=True)

model = PointNet(n_in_channel=8)
for x, y in dataset:
    input = x[:, 0, :, :]
    output = model(input)
    print(input.shape)
    print(output.shape)
    break
