from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    root_dir: Path
    input_dir: Path
    output_dir: Path
    work_dir: Path
    variants: dict[str, dict[str, str]]
    temporal: dict[str, Any]
    dem: dict[str, Any]
    variables: dict[str, Any]
    topography: dict[str, Any]

    @classmethod
    def load(cls, config_path: Path, root_dir: Path | None = None) -> PipelineConfig:
        root = (root_dir or config_path.parent.parent).resolve()
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def resolve(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else (root / path).resolve()

        return cls(
            root_dir=root,
            input_dir=resolve(raw["input_dir"]),
            output_dir=resolve(raw["output_dir"]),
            work_dir=resolve(raw["work_dir"]),
            variants=raw["variants"],
            temporal=raw["temporal"],
            dem=raw["dem"],
            variables=raw["variables"],
            topography=raw["topography"],
        )

    def variant_input(self, variant: str) -> Path:
        return self.input_dir / self.variants[variant]["subdir"]

    def variant_work(self, stage: str, variant: str) -> Path:
        return self.work_dir / stage / self.variants[variant]["subdir"]

    def variant_output(self, variant: str) -> Path:
        return self.output_dir / self.variants[variant]["subdir"]

    def sectors_path(self) -> Path:
        sectors = self.dem["sectors_config"]
        path = Path(sectors)
        if not path.is_absolute():
            path = self.root_dir / path
        return path.resolve()

    def load_sectors_lonlat(self) -> dict[str, dict[str, tuple[float, float]]]:
        with open(self.sectors_path(), encoding="utf-8") as f:
            raw = json.load(f)
        sectors: dict[str, dict[str, tuple[float, float]]] = {}
        for name, points in raw.items():
            sectors[name] = {
                key: (float(value[0]), float(value[1]))
                for key, value in points.items()
            }
        return sectors

    def is_sen2(self, variant: str) -> bool:
        return variant.lower() == "sen2"

    def variable_profile(self, variant: str) -> dict[str, list[str]]:
        key = "sen2" if self.is_sen2(variant) else "sar"
        profile = self.variables[key]
        return {
            "dynamic": list(profile["dynamic"]),
            "static": list(profile["static"]),
            "final_order": list(profile["final_order"]),
        }
