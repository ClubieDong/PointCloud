import os
import json
import time
import dataclasses
from datetime import datetime
from models.mlp import MLP
import utils
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from config import *
from models.pointnet import PointNet
from models.pointnetpp_ssg import PointNetPPSSG
from models.pointnetpp_msg import PointNetPPMSG
from models.conv3d import Conv3D
from models.classifier import Classifier
from datasets.RadHAR import RadHARDataset
from datasets.Pantomime import PantomimeDataset


def get_dataset(config) -> tuple[DataLoader, DataLoader, list[list[np.ndarray]]]:
    if isinstance(config, RadHARDatasetConfig):
        train_dataset = DataLoader(RadHARDataset(config, "train"), shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
        test_dataset = DataLoader(RadHARDataset(config, "test"), shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)
        raw_data = train_dataset.dataset.data
    elif isinstance(config, PantomimeDatasetConfig):
        dataset = PantomimeDataset(config)
        train_dataset, test_dataset = random_split(dataset, (int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)), generator=torch.Generator().manual_seed(42))
        train_dataset = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
        test_dataset = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)
        raw_data = dataset.data
    else:
        raise ValueError("Invalid dataset config")
    return train_dataset, test_dataset, raw_data


def get_model(backbone_config, classifier_config: ClassifierConfig) -> Classifier:
    if isinstance(backbone_config, PointNetConfig):
        backbone = PointNet(backbone_config)
    elif isinstance(backbone_config, PointNetPPSSGConfig):
        backbone = PointNetPPSSG(backbone_config)
    elif isinstance(backbone_config, PointNetPPMSGConfig):
        backbone = PointNetPPMSG(backbone_config)
    elif isinstance(backbone_config, Conv3DConfig):
        backbone = Conv3D(backbone_config)
    elif isinstance(backbone_config, PCABackboneConfig):
        backbone = MLP(backbone_config.layers)
    else:
        raise ValueError("Invalid backbone config")
    classifier = Classifier(backbone, classifier_config)
    return classifier.to(device)


def train_epoch(model: nn.Module, train_dataset: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, epoch_idx: int) -> tuple[float, float, float]:
    start_time = time.time()
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
    finish_time = time.time()
    elapsed_time = finish_time - start_time
    print(f"Train accuracy: {(100*n_train_correct):>0.2f}%, Average loss: {train_loss:>0.6f}, Elapsed time: {(elapsed_time):>0.2f}s")
    return n_train_correct, train_loss, elapsed_time


def test_epoch(model: nn.Module, test_dataset: DataLoader, criterion: nn.Module, n_resample_time: int, n_class: int) -> tuple[float, float, float]:
    start_time = time.time()
    zipped_test_dataset = zip(*([test_dataset] * n_resample_time))
    model.eval()
    test_loss, n_test_correct = 0.0, 0
    with torch.no_grad():
        for data in tqdm(zipped_test_dataset, desc="Test", total=len(test_dataset)):
            # Resample multiple times to get more robust results
            total_pred = torch.zeros(len(data[0][0]), n_class, device=device)
            # total_pred.shape = (n_batch, n_class)
            for x, y in data:
                x, y = x.to(device), y.to(device)
                # x.shape = (n_batch, n_chunk, n_point, n_channel)
                # y.shape = (n_batch)  y.dtype = torch.int64
                pred = model(x)
                # pred.shape = (n_batch, n_class)
                test_loss += criterion(pred, y).item()
                total_pred += pred
            n_test_correct += (total_pred.argmax(dim=1) == data[0][1].to(device)).sum().item()
    test_loss /= len(test_dataset) * n_resample_time
    n_test_correct /= len(test_dataset.dataset)
    finish_time = time.time()
    elapsed_time = finish_time - start_time
    print(f"Test accuracy: {(100*n_test_correct):>0.2f}%, Average loss: {test_loss:>0.6f}, Elapsed time: {(elapsed_time):>0.2f}s")
    print()
    return n_test_correct, test_loss, elapsed_time


def train(config: Config):
    # Save config
    log_dir = os.path.join("logs", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    config_dict = dataclasses.asdict(config)
    config_dict["python_expr"] = config.__repr__()
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)
    # Load dataset and model
    train_dataset, test_dataset, raw_data = get_dataset(config.dataset_config)
    model = get_model(config.backbone_config, config.classifier_config)
    if config.classifier_config.type == "pca":
        model.pca = utils.calc_pca(
            raw_data=raw_data, 
            n_components=config.classifier_config.n_components,
            dim_size=config.classifier_config.dim_size,
            center=config.classifier_config.center,
            voxel_size=config.classifier_config.voxel_size
        )
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # Show statistics
    print("Log directory:", log_dir)
    print("Model size:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Train dataset size:", len(train_dataset.dataset))
    print("Test dataset size:", len(test_dataset.dataset))
    print()
    # Train
    data: list[dict[str, float]] = []
    best_test_accuracy = 0.0
    try:
        for epoch_idx in range(config.n_epoch):
            epoch_data = {"epoch_idx": epoch_idx}
            train_accuracy, train_loss, train_time = train_epoch(model, train_dataset, criterion, optimizer, epoch_idx)
            epoch_data["train_accuracy"] = train_accuracy
            epoch_data["train_loss"] = train_loss
            epoch_data["train_time"] = train_time
            if (epoch_idx + 1) % config.test_interval == 0:
                test_accuracy, test_loss, test_time = test_epoch(model, test_dataset, criterion, config.n_test_resample_time, config.classifier_config.head_layers[-1])
                epoch_data["test_accuracy"] = test_accuracy
                epoch_data["test_loss"] = test_loss
                epoch_data["test_time"] = test_time
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            data.append(epoch_data)
    except KeyboardInterrupt:
        pass
    # Save model and data
    print("Saving model and data...")
    torch.save(model.state_dict(), os.path.join(log_dir, "model.pth"))
    with open(os.path.join(log_dir, "training_log.json"), "w") as f:
        json.dump(data, f, indent=4)
    statistics = {
        "train_dataset_size": len(train_dataset.dataset),
        "test_dataset_size": len(test_dataset.dataset),
        "model_size": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_test_accuracy": best_test_accuracy,
        "total_time": sum(d["train_time"] for d in data) + sum(d["test_time"] for d in data),
    }
    with open(os.path.join(log_dir, "statistics.json"), "w") as f:
        json.dump(statistics, f, indent=4)
    print("Done!")


def evaluate(model_path: str, config: Config):
    # Load dataset and model
    _, test_dataset, raw_data = get_dataset(config.dataset_config)
    model = get_model(config.backbone_config, config.classifier_config)
    model.load_state_dict(torch.load(model_path))
    if config.classifier_config.type == "pca":
        model.pca = utils.calc_pca(
            raw_data=raw_data, 
            n_components=config.classifier_config.n_components,
            dim_size=config.classifier_config.dim_size,
            center=config.classifier_config.center,
            voxel_size=config.classifier_config.voxel_size
        )
    criterion = nn.CrossEntropyLoss()
    # Evaluate
    test_epoch(model, test_dataset, criterion, config.n_test_resample_time, config.classifier_config.head_layers[-1])


if __name__ == "__main__":
    point_net_cfg = Config(
        n_epoch=40,
        dataset_config=PantomimeDatasetConfig(
            envs=["office", "open", "industrial", "restaurant"],
            angles=[0],
            speeds=["normal"],
            actions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            translate_total_dist_std=0.1,
            translate_point_dist_std=0.01,
            scale_factor_std=0.1,
        ),
        backbone_config=PointNetConfig(
            blocks=[
                PointNetConfig.PointNetBlockConfig(            # n_in_channel = 8
                    mlp_conv_layers=[3, 64],                   # [n_in_channel, 64]
                    t_net_mlp_conv_layers=[3, 64, 128, 1024],  # [n_in_channel, 64, 128, 1024]
                    t_net_mlp_layers=[1024, 512, 256, 3*3],    # [1024, 512, 256, n_in_channel*n_in_channel]
                ),
                PointNetConfig.PointNetBlockConfig(             # n_in_channel = 64
                    mlp_conv_layers=[64, 128, 1024],            # [n_in_channel, 128, 1024]
                    t_net_mlp_conv_layers=[64, 64, 128, 1024],  # [n_in_channel, 64, 128, 1024]
                    t_net_mlp_layers=[1024, 512, 256, 64*64],   # [1024, 512, 256, n_in_channel*n_in_channel]
                ),
            ]
        ),
        classifier_config=ClassifierConfig(
            rnn_config=ClassifierConfig.RNNConfig(
                name="lstm",
                input_size=1024,
                hidden_size=256,
            ),
            head_layers=[256, 64, 21],
        ),
    )
    point_net_pp_ssg_cfg = Config(
        n_epoch=40,
        dataset_config=PantomimeDatasetConfig(
            envs=["office", "open", "industrial", "restaurant"],
            angles=[0],
            speeds=["normal"],
            actions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            translate_total_dist_std=0.1,
            translate_point_dist_std=0.01,
            scale_factor_std=0.1,
        ),
        backbone_config=PointNetPPSSGConfig(
            set_abstractions=[
                PointNetPPSSGConfig.SetAbstractionConfig(
                    n_out_point=50,
                    ball_query_n_sample=8,
                    ball_query_radius=0.2,
                    mlp_layers=[3, 64, 128],
                ),
                PointNetPPSSGConfig.SetAbstractionConfig(
                    n_out_point=20,
                    ball_query_n_sample=16,
                    ball_query_radius=0.4,
                    mlp_layers=[128, 128, 256],
                ),
            ],
            final_mlp_layers=[256, 512, 1024]
        ),
        classifier_config=ClassifierConfig(
            rnn_config=ClassifierConfig.RNNConfig(
                name="rnn",
                input_size=1024,
                hidden_size=256,
            ),
            head_layers=[256, 64, 21],
        ),
    )
    point_net_pp_msg_cfg = Config(
        n_epoch=40,
        dataset_config=PantomimeDatasetConfig(
            envs=["office", "open", "industrial", "restaurant"],
            angles=[0],
            speeds=["normal"],
            actions=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            translate_total_dist_std=0.1,
            translate_point_dist_std=0.01,
            scale_factor_std=0.1,
        ),
        backbone_config=PointNetPPMSGConfig(
            set_abstractions=[
                PointNetPPMSGConfig.SetAbstractionConfig(
                    n_out_point=50,
                    ball_query_n_sample=[8, 16, 32],
                    ball_query_radius=[0.2, 0.4, 0.6],
                    mlp_layers=[[3, 16, 32], [3, 16, 32], [3, 16, 32]],
                ),
                PointNetPPMSGConfig.SetAbstractionConfig(
                    n_out_point=20,
                    ball_query_n_sample=[16, 24, 32],
                    ball_query_radius=[0.4, 0.8, 1.2],
                    mlp_layers=[[3+32*3, 128, 256], [3+32*3, 128, 256], [3+32*3, 128, 256]],
                ),
            ],
            final_mlp_layers=[3+256*3, 1024]
        ),
        classifier_config=ClassifierConfig(
            rnn_config=ClassifierConfig.RNNConfig(
                name="lstm",
                input_size=1024,
                hidden_size=256,
            ),
            head_layers=[256, 64, 21],
        ),
    )
    
    for cfg in [point_net_cfg, point_net_pp_ssg_cfg, point_net_pp_msg_cfg]:
        for rnn_name in ["rnn", "gru", "lstm"]:
            cfg.classifier_config.rnn_config.name = rnn_name
            train(cfg)
