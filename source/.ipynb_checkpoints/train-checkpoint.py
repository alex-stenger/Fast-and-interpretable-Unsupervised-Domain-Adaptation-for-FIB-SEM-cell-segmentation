from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from .datasets import Cityscapes, GTAV
from .model import UNet
from .dice import DiceLoss


@dataclass
class Model:

    device:    torch.device
    network:   torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    criterion: torch.nn.modules.loss._Loss


def make_dataloader(
    rootdir: str, crop_size: tuple[int], split: str, batch_size: int, num_workers: int
) -> DataLoader:
    loader_kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    Dataset = Cityscapes if "cityscapes" in rootdir else GTAV
    dataset = Dataset(rootdir, crop_size, split)
    return DataLoader(dataset, **loader_kwargs, pin_memory=True)


def make_model(
    n_channels: int, n_classes: int, lr: float, dataloader: DataLoader, epochs: int
) -> Model:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = UNet(n_channels=n_channels, n_classes=n_classes).to(device)
    steps_per_epoch = len(dataloader)
    scheduler_params = dict(max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch)
    optimizer = torch.optim.AdamW(network.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    criterion = DiceLoss()
    return Model(device, network, optimizer, scheduler, criterion)


def prepare_labels(labels: Tensor, num_classes: int, device: torch.device) -> Tensor:
    """ ignore unlabelled and encode to one hot. """
    labels = labels.long()
    labels[labels == 255] = num_classes
    labels = labels[:, 0, :, :].unsqueeze(1)
    batch_size, _, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes + 1, height, width).to(device)
    one_hot.scatter_(1, labels, 1)
    one_hot = one_hot[:, :num_classes].float()
    return one_hot


def train(
    model: Model, dataloader: DataLoader, epochs: int,
) -> tuple[Model, list[float], list[float]]:
    losses, learning_rates = list(), list()
    best_epoch = 0
    pbar = tqdm(total=len(dataloader))
    for i in range(epochs):
        pbar.set_description_str(f"Epoch [{i+1}/{epochs}] (best epoch: {best_epoch})")
        epoch_loss = 0
        for batch in dataloader:
            batch = map(lambda x: x.to(model.device, non_blocking=True), batch)
            images, labels = batch
            outputs = model.network(images)
            labels = prepare_labels(labels, model.network.n_classes, model.device)
            loss = model.criterion(outputs.softmax(dim=1), labels)
            model.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.optimizer.step()
            epoch_loss += loss.detach()
            pbar.update()
            pbar.set_postfix(loss=loss.detach().item(), lr=model.optimizer.param_groups[0]["lr"])
            if isinstance(model.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                model.scheduler.step()
        epoch_loss = epoch_loss.mean().item() / len(dataloader)
        losses.append(epoch_loss)
        learning_rates.append(model.optimizer.param_groups[0]["lr"])
        if epoch_loss <= min(losses):
            best_epoch = i + 1
        if isinstance(model.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            model.scheduler.step(epoch_loss)
        pbar.reset()
    pbar.close()
    return model, losses, learning_rates


def save(
    model: Model, losses: list[float], learning_rates: list[float],
    rootdir: str, output_dir: str, run_name: str
) -> None:
    dataset = "cityscapes" if "cityscapes" in rootdir else "gta"
    dir = Path(output_dir)
    dir.mkdir(exist_ok=True)
    #path = dir / f"{run_name}_{dataset}.pt"
    path = dir / f"{run_name}"

    state = dict(
        network=model.network.state_dict(),
        optimizer=model.optimizer.state_dict(),
        scheduler=model.scheduler.state_dict(),
        losses=losses,
        learning_rates=learning_rates,
    )
    torch.save(state, path)


def main(
    rootdir: str, output_dir: str, run_name: str,
    crop_size: tuple[int], batch_size: int, num_workers: int,
    n_channels: int, n_classes: int, lr: float, epochs: int,
) -> None:
    dataloader = make_dataloader(rootdir, crop_size, "train", batch_size, num_workers)
    model = make_model(n_channels, n_classes, lr, dataloader, epochs)
    model, losses, learning_rates = train(model, dataloader, epochs)
    save(model, losses, learning_rates, rootdir, output_dir, run_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Training pipeline")
    parser.add_argument("--rootdir",     type=str)
    parser.add_argument("--output_dir",  type=str,   default="./")
    parser.add_argument("--run_name",    type=str,   default="run")
    parser.add_argument("--crop_size",   type=int,   default=(321, 321), nargs=2)
    parser.add_argument("--batch_size",  type=int,   default=10)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--n_channels",  type=int,   default=3)
    parser.add_argument("--n_classes",   type=int,   default=19)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=100)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
