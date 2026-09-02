from __future__ import annotations

from typing import Any

import torch.nn as nn

from .wrappers import SimpleConvGRU, SimpleFCNCRNN, SimpleUNet3D, SimpleUTAE

MODEL_NAMES = ("unet3d", "utae", "convgru", "fcn_crnn")


def build_model(
    name: str,
    in_channels: int,
    num_classes: int,
    params: dict[str, Any] | None = None,
) -> nn.Module:
    params = params or {}
    img_res = int(params.get("img_res", 128))

    if name == "unet3d":
        return SimpleUNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            img_res=img_res,
            dropout=float(params.get("dropout", 0.0)),
        )
    if name == "utae":
        return SimpleUTAE(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_widths=list(params.get("encoder_widths", [64, 64, 64, 128])),
            decoder_widths=list(params.get("decoder_widths", [32, 32, 64, 128])),
            pad_value=float(params.get("pad_value", 0.0)),
        )
    if name == "convgru":
        kernel_size = params.get("kernel_size", [3, 3])
        return SimpleConvGRU(
            in_channels=in_channels,
            num_classes=num_classes,
            img_res=img_res,
            hidden_dim=int(params.get("hidden_dim", 128)),
            kernel_size=(int(kernel_size[0]), int(kernel_size[1])),
            pad_value=float(params.get("pad_value", 0.0)),
        )
    if name == "fcn_crnn":
        return SimpleFCNCRNN(
            in_channels=in_channels,
            num_classes=num_classes,
            img_res=img_res,
            crnn_model_name=str(params.get("crnn_model_name", "clstm")),
            train_stage=int(params.get("train_stage", 2)),
            pretrained=bool(params.get("pretrained", False)),
        )

    raise ValueError(f"Unknown model '{name}'. Choose from: {MODEL_NAMES}")
