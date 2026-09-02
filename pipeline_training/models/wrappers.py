from __future__ import annotations

import torch
import torch.nn as nn

from .convgru import ConvGRU_Seg
from .fcn_crnn import FCN_CRNN
from .unet3d import UNet3D
from .utae import UTAE


class SimpleUNet3D(nn.Module):
    """Thin wrapper around Sen12Landslides UNet3D."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_res: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = UNet3D(
            in_channels=in_channels,
            num_classes=num_classes,
            img_res=img_res,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["segmentation"]


class SimpleUTAE(nn.Module):
    """Thin wrapper around Sen12Landslides UTAE."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        encoder_widths: list[int] | None = None,
        decoder_widths: list[int] | None = None,
        pad_value: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_widths = encoder_widths or [64, 64, 64, 128]
        decoder_widths = decoder_widths or [32, 32, 64, 128]
        self.model = UTAE(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_widths=encoder_widths,
            decoder_widths=decoder_widths,
            pad_value=pad_value,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["segmentation"]


class SimpleConvGRU(nn.Module):
    """Thin wrapper around Sen12Landslides ConvGRU_Seg."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_res: int = 128,
        hidden_dim: int = 128,
        kernel_size: tuple[int, int] = (3, 3),
        pad_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = ConvGRU_Seg(
            num_classes=num_classes,
            img_res=img_res,
            in_channels=in_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            pad_value=pad_value,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["segmentation"]


class SimpleFCNCRNN(nn.Module):
    """Thin wrapper around Sen12Landslides FCN_CRNN (ConvLSTM path)."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        img_res: int = 128,
        crnn_model_name: str = "clstm",
        train_stage: int = 2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.model = FCN_CRNN(
            num_classes=num_classes,
            in_channels=in_channels,
            img_res=img_res,
            crnn_model_name=crnn_model_name,
            train_stage=train_stage,
            pretrained=pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["segmentation"]
