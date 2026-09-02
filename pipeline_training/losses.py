from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 7.0,
        bce_weight: float = 0.6,
        dice_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight.to(logits.device),
        )
        dice = binary_dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def build_criterion(cfg_training: dict, device: torch.device) -> nn.Module:
    loss_name = str(cfg_training.get("loss", "cross_entropy")).lower()
    target_mode = str(cfg_training.get("target_mode", "ce")).lower()

    if loss_name == "bce_dice" or target_mode == "bce":
        return BCEDiceLoss(
            pos_weight=float(cfg_training.get("pos_weight", 7.0)),
            bce_weight=float(cfg_training.get("bce_weight", 0.6)),
            dice_weight=float(cfg_training.get("dice_weight", 0.4)),
        ).to(device)

    class_weights = cfg_training.get("class_weights", [1.0, 5.0])
    weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight)
