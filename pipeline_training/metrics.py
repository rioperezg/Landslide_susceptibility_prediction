from __future__ import annotations

import torch
import torch.nn.functional as F


def logits_to_probs(logits: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "bce":
        return torch.sigmoid(logits)
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=1)[:, 1:2]


def binary_predictions(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    if probs.shape[1] > 1:
        probs = probs[:, 1:2]
    return (probs >= threshold).long()


def update_confusion(
    pred: torch.Tensor,
    target: torch.Tensor,
    tallies: dict[str, int],
) -> None:
    pred = pred.view(-1)
    target = target.view(-1)
    tallies["tp"] += int(((pred == 1) & (target == 1)).sum().item())
    tallies["fp"] += int(((pred == 1) & (target == 0)).sum().item())
    tallies["fn"] += int(((pred == 0) & (target == 1)).sum().item())
    tallies["tn"] += int(((pred == 0) & (target == 0)).sum().item())


def metrics_from_confusion(tallies: dict[str, int], eps: float = 1e-8) -> dict[str, float]:
    tp = tallies["tp"]
    fp = tallies["fp"]
    fn = tallies["fn"]
    tn = tallies["tn"]
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


def scan_thresholds(
    probs_list: list[torch.Tensor],
    targets_list: list[torch.Tensor],
    thresholds: torch.Tensor | None = None,
) -> dict[str, float]:
    if thresholds is None:
        thresholds = torch.linspace(0.05, 0.95, 19)

    best: dict[str, float] = {"f1": -1.0}
    for threshold in thresholds:
        tallies = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for probs, target in zip(probs_list, targets_list):
            if target.dim() == 3:
                target = target.unsqueeze(1)
            pred = binary_predictions(probs, float(threshold.item()))
            update_confusion(pred, target.long(), tallies)
        result = metrics_from_confusion(tallies)
        result["threshold"] = float(threshold.item())
        if result["f1"] > best.get("f1", -1.0):
            best = result
    return best
