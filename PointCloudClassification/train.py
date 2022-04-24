import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from config import config_radhar_dataset, config_pantomime_dataset, device
from models.classifier import Classifier
from datasets.RadHAR import RadHARDataset
from datasets.Pantomime import PantomimeDataset


def get_dataset(dataset_name: str) -> tuple[DataLoader, DataLoader]:
    if dataset_name == "RadHAR":
        train_dataset = DataLoader(RadHARDataset("data/RadHAR/Train"), shuffle=True, batch_size=128, num_workers=0)
        test_dataset = DataLoader(RadHARDataset("data/RadHAR/Test"), shuffle=True, batch_size=128, num_workers=0)
    elif dataset_name == "Pantomime":
        dataset = PantomimeDataset("data/Pantomime", envs=["open", "office"], angles=[0], speeds=["normal"])
        train_dataset, test_dataset = random_split(dataset, (int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)))
        train_dataset = DataLoader(train_dataset, shuffle=True, batch_size=512, num_workers=0)
        test_dataset = DataLoader(test_dataset, shuffle=True, batch_size=512, num_workers=0)
    return train_dataset, test_dataset


def get_model(dataset_name: str) -> nn.Module:
    if dataset_name == "RadHAR":
        model = Classifier(n_in_channel=config_radhar_dataset["n_in_channel"],
                           n_chunk=config_radhar_dataset["n_chunk_per_data"],
                           n_class=config_radhar_dataset["n_class"]).to(device)
    elif dataset_name == "Pantomime":
        model = Classifier(n_in_channel=config_pantomime_dataset["n_in_channel"],
                           n_chunk=config_pantomime_dataset["n_chunk_per_data"],
                           n_class=config_pantomime_dataset["n_class"]).to(device)
    return model


def train(model: nn.Module, train_dataset: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, epoch_idx: int):
    model.train()
    train_loss, n_train_correct = 0.0, 0
    for x, y in tqdm(train_dataset, desc=f"Epoch #{epoch_idx+1}"):
        x, y = x.to(device), y.to(device)
        # x.shape = (n_batch, n_chunk, n_point, n_channel)
        # y.shape = (n_batch)  y.dtype = torch.int64
        pred = model(x)
        # pred.shape = (n_batch, n_class)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_train_correct += (pred.argmax(dim=1) == y).sum().item()
    train_loss /= len(train_dataset)
    n_train_correct /= len(train_dataset.dataset)
    print(f"Train accuracy: {(100*n_train_correct):>0.1f}%, Average loss: {train_loss:>8f}")


def test(model: nn.Module, test_dataset: DataLoader, criterion: nn.Module):
    model.eval()
    test_loss, n_test_correct = 0.0, 0
    with torch.no_grad():
        for x, y in tqdm(test_dataset, desc="Test"):
            x, y = x.to(device), y.to(device)
            # x.shape = (n_batch, n_chunk, n_point, n_channel)
            # y.shape = (n_batch)  y.dtype = torch.int64
            pred = model(x)
            # pred.shape = (n_batch, n_class)
            test_loss += criterion(pred, y).item()
            n_test_correct += (pred.argmax(dim=1) == y).sum().item()
    test_loss /= len(test_dataset)
    n_test_correct /= len(test_dataset.dataset)
    print(f"Test accuracy: {(100*n_test_correct):>0.1f}%, Average loss: {test_loss:>8f}")
    print()


if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset("RadHAR")
    model = get_model("RadHAR")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print("Model size:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Train dataset size:", len(train_dataset.dataset))
    print("Test dataset size:", len(test_dataset.dataset))
    print()

    for epoch_idx in range(100):
        train(model, train_dataset, criterion, optimizer, epoch_idx)
        if (epoch_idx + 1) % 1 == 0:
            test(model, test_dataset, criterion)
