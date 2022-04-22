import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from config import config_radhar_dataset, config_pantomime_dataset, device
from models.classifier import Classifier
from datasets.RadHAR import RadHARDataset
from datasets.Pantomime import PantomimeDataset

# ========== RadHAR ==========
# train_dataset = DataLoader(RadHARDataset("RadHAR/Data/Train"), shuffle=True, batch_size=128, num_workers=0)
# test_dataset = DataLoader(RadHARDataset("RadHAR/Data/Test"), shuffle=True, batch_size=128, num_workers=0)

# ========== Pantomime ==========
dataset = PantomimeDataset("data/Pantomime", envs=["open", "office"], angles=[0], speeds=["normal"])
train_dataset, test_dataset = random_split(dataset, (int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)))
train_dataset = DataLoader(train_dataset, shuffle=True, batch_size=512, num_workers=0)
test_dataset = DataLoader(test_dataset, shuffle=True, batch_size=512, num_workers=0)

# model = Classifier(n_in_channel=config_radhar_dataset["n_in_channel"],
#                    n_chunk=config_radhar_dataset["n_chunk_per_data"],
#                    n_class=config_radhar_dataset["n_class"]).to(device)
model = Classifier(n_in_channel=config_pantomime_dataset["n_in_channel"],
                   n_chunk=config_pantomime_dataset["n_chunk_per_data"],
                   n_class=config_pantomime_dataset["n_class"]).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print("Model size:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Train dataset size:", len(train_dataset.dataset))
print("Test dataset size:", len(test_dataset.dataset))

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
