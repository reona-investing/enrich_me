from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

from .base_mlobject import BaseMLObject
from .data_bundle import DataBundle


@dataclass
class MLPackage:
    """Bundle of ML object, data and meta information."""

    ml_object: BaseMLObject
    data: DataBundle
    meta: dict = field(default_factory=dict)

    def train(self) -> None:
        """Train internal ML object."""
        self.data.prepare()
        self.ml_object.fit(self.data)

    def predict(self, features: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predict using the ML object."""
        feats = features if features is not None else self.data.features
        return self.ml_object.predict(feats)

    def save(self, dir_: Path) -> None:
        """Save package components to disk."""
        dir_.mkdir(parents=True, exist_ok=True)
        self.ml_object.save(dir_ / "ml_object.pkl")
        self.data.save(dir_ / "data")
        with open(dir_ / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

    @classmethod
    def load(cls, dir_: Path) -> "MLPackage":
        """Load package from disk."""
        ml_object = UnifiedMLObject.load(dir_ / "ml_object.pkl")
        data = DataBundle.load(dir_ / "data")
        meta_path = dir_ / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        return cls(ml_object=ml_object, data=data, meta=meta)


from .unified_mlobject import UnifiedMLObject  # after class definition
import pandas as pd

