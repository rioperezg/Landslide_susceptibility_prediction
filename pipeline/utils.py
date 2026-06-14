from __future__ import annotations

import json
import re
from pathlib import Path

import xarray as xr


PATCH_NUM_RE = re.compile(r"_(\d+)(?:_enriched)?\.(?:nc|pt)$", re.IGNORECASE)


def patch_number(filename: str) -> str:
    match = PATCH_NUM_RE.search(filename)
    if not match:
        raise ValueError(f"Cannot extract patch number from: {filename}")
    return match.group(1)


def list_nc_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.nc"))


def list_pt_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.pt"))


def common_patch_numbers(variant_dirs: dict[str, Path]) -> set[str]:
    sets = []
    for directory in variant_dirs.values():
        numbers = {patch_number(path.name) for path in list_nc_files(directory)}
        sets.append(numbers)
    common = set.intersection(*sets) if sets else set()
    return common


def filter_files_by_numbers(files: list[Path], numbers: set[str]) -> list[Path]:
    return sorted(
        path for path in files if patch_number(path.name) in numbers
    )


def read_event_date(ds: xr.Dataset, attr_names: list[str]):
    for attr in attr_names:
        value = ds.attrs.get(attr)
        if value is not None:
            return value
    return None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
