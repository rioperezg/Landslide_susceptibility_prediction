from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pipeline_training.dataset import LandslideTensorDataset, TrainingConfig, load_split_ids
from pipeline_training.losses import build_criterion
from pipeline_training.models import build_model

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def run_train(
    cfg: TrainingConfig,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_split_ids(cfg.split_file, cfg.split_extra_file)

    train_dataset = LandslideTensorDataset(
        pt_dirs=cfg.data_dir,
        patch_ids=splits["train"],
        mean=mean,
        std=std,
        selected_variables=cfg.selected_variables,
        target_mode=cfg.target_mode,
    )
    val_dataset = LandslideTensorDataset(
        pt_dirs=cfg.data_dir,
        patch_ids=splits["val"],
        mean=mean,
        std=std,
        selected_variables=cfg.selected_variables,
        target_mode=cfg.target_mode,
    )

    batch_size = int(cfg.training.get("batch_size", 4))
    num_workers = int(cfg.training.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(
        cfg.model,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        params=cfg.model_params,
    ).to(device)

    criterion = build_criterion(cfg.training, device)
    lr = float(cfg.training.get("lr", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = int(cfg.training.get("num_epochs", 50))
    best_val = float("inf")
    history: list[dict[str, float]] = []

    logger.info("Training %s on %s (%d train / %d val patches)", cfg.model, device, len(train_dataset), len(val_dataset))

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f",
            epoch + 1,
            num_epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": cfg.model,
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "selected_variables": cfg.selected_variables,
                    "target_mode": cfg.target_mode,
                    "num_classes": cfg.num_classes,
                    "model_params": cfg.model_params,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                },
                cfg.checkpoint_path,
            )
            logger.info("Saved best checkpoint to %s", cfg.checkpoint_path)

    torch.save({"history": history}, cfg.output_dir / "training_history.pt")
    return cfg.checkpoint_path
