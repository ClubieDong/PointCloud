from datasets.RadHAR import RadHARDataset
from transforms.transforms import Transform
from transforms.resampler import RandomResampler
from torch.utils.data import DataLoader
from config import config_radhar_dataset, device
from models.classifier import Classifier
import torch
import torch.nn as nn
from tqdm import tqdm

transfrom = Transform()
transfrom.resampler = RandomResampler(config_radhar_dataset["n_sample_per_chunk"])
train_dataset = DataLoader(RadHARDataset("RadHAR/Data/Train", transfrom), batch_size=4, shuffle=True)
test_dataset = DataLoader(RadHARDataset("RadHAR/Data/Test", transfrom), batch_size=4, shuffle=True)
model = Classifier(n_in_channel=8, n_chunk=config_radhar_dataset["n_chunk"], n_class=5).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

print("Model size: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
for epoch_idx in range(100):
    # Train
    model.train()
    train_loss, n_train_correct = 0, 0
    for batch_idx, (x, y) in enumerate(tqdm(train_dataset, desc=f"Epoch #{epoch_idx+1}")):
        x, y = x.to(device), y.to(device)
        # x.shape = (n_batch, n_chunk, n_point, n_channel)
        # y.shape = (n_batch)  y.dtype = torch.int64
        pred = model(x)
        # pred.shape = (n_batch, n_class)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss_fn(pred, y).item()
        n_train_correct += (pred.argmax(dim=1) == y).sum().item()
    train_loss /= len(train_dataset)
    n_train_correct /= len(train_dataset.dataset)
    print(f"Train Accuracy: {(100*n_train_correct):>0.1f}%, Avg loss: {train_loss:>8f}")
    # Test
    if (epoch_idx + 1) % 5 == 0:
        model.eval()
        test_loss, n_test_correct = 0, 0
        with torch.no_grad():
            for x, y in test_dataset:
                x, y = x.to(device), y.to(device)
                # x.shape = (n_batch, n_chunk, n_point, n_channel)
                # y.shape = (n_batch)  y.dtype = torch.int64
                pred = model(x)
                # pred.shape = (n_batch, n_class)
                test_loss += loss_fn(pred, y).item()
                n_test_correct += (pred.argmax(dim=1) == y).sum().item()
        test_loss /= len(test_dataset)
        n_test_correct /= len(test_dataset.dataset)
        print(f"Test Accuracy: {(100*n_test_correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
