from __future__ import annotations

from pathlib import Path
from typing import Dict

from .ml_package import MLPackage


class MLRegistry:
    """Registry for storing ML packages identified by keys."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.packages: Dict[str, MLPackage] = {}

    def save(self, key: str, obj: MLPackage) -> None:
        dir_ = self.root_dir / key
        obj.save(dir_)
        self.packages[key] = obj

    def load(self, key: str) -> MLPackage:
        dir_ = self.root_dir / key
        pkg = MLPackage.load(dir_)
        self.packages[key] = pkg
        return pkg

