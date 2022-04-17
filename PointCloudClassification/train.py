from datasets.RadHAR import RadHARDataset
from transforms.transforms import Transform
from transforms.resampler import RandomResampler
from torch.utils.data import DataLoader
from models.pointnetpp import PointNetPP
from config import config_radhar_dataset

transfrom = Transform()
transfrom.resampler = RandomResampler(config_radhar_dataset["n_sample_per_chunk"])
dataset = DataLoader(RadHARDataset("RadHAR/Data/Test", transfrom), batch_size=4, shuffle=True)

model = PointNetPP(5)
for x, y in dataset:
    x = x[:, 0, :, :]
    yy = model(x)
