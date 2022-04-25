import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from config import *
from models.pointnet import PointNet
from models.pointnetpp import PointNetPP
from models.classifier import Classifier
from datasets.RadHAR import RadHARDataset
from datasets.Pantomime import PantomimeDataset


def get_dataset(config) -> tuple[DataLoader, DataLoader]:
    if isinstance(config, RadHARDatasetConfig):
        train_dataset = DataLoader(RadHARDataset(config, "train"), shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
        test_dataset = DataLoader(RadHARDataset(config, "test"), shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)
    elif isinstance(config, PantomimeDatasetConfig):
        dataset = PantomimeDataset(config)
        train_dataset, test_dataset = random_split(dataset, (int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)))
        train_dataset = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
        test_dataset = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
    else:
        raise ValueError("Invalid dataset config")
    return train_dataset, test_dataset


def get_model(backbone_config, classifier_config: ClassifierConfig) -> nn.Module:
    if isinstance(backbone_config, PointNetConfig):
        backbone = PointNet(backbone_config)
    elif isinstance(backbone_config, PointNetPPConfig):
        backbone = PointNetPP(backbone_config)
    else:
        raise ValueError("Invalid backbone config")
    classifier = Classifier(backbone, classifier_config)
    return classifier.to(device)


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
    dataset_config = RadHARDatasetConfig()
    backbone_config = PointNetPPConfig()
    classifier_config = ClassifierConfig()

    train_dataset, test_dataset = get_dataset(dataset_config)
    model = get_model(backbone_config, classifier_config)
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
