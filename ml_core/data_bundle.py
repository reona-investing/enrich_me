from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import json
import pandas as pd


@dataclass
class DataBundle:
    """Container for feature and target data with optional preprocessing."""

    features: pd.DataFrame
    targets: pd.DataFrame
    order_price: Optional[pd.DataFrame] = None
    meta: Dict | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = self.features.copy()
        self.targets = self.targets.copy()
        if self.order_price is not None:
            self.order_price = self.order_price.copy()
        if self.meta is None:
            self.meta = {}

    def prepare(self) -> None:
        """Run simple preprocessing like outlier removal."""
        threshold = self.meta.get("outlier_threshold")
        if threshold is not None:
            mask = (self.features.abs() > threshold)
            self.features[mask] = pd.NA
            self.features.dropna(inplace=True)
            self.targets = self.targets.loc[self.features.index]
            if self.order_price is not None:
                self.order_price = self.order_price.loc[self.features.index]

    def slice_date(self, date: pd.Timestamp) -> "DataBundle":
        """Return a new DataBundle filtered by date."""
        features = self.features.xs(date, level="Date", drop_level=False)
        targets = self.targets.xs(date, level="Date", drop_level=False)
        op = None
        if self.order_price is not None:
            op = self.order_price.xs(date, level="Date", drop_level=False)
        return DataBundle(features, targets, op, self.meta)

    def save(self, dir_: Path) -> None:
        """Save bundle contents to a directory."""
        dir_.mkdir(parents=True, exist_ok=True)
        self.features.to_parquet(dir_ / "features.parquet")
        self.targets.to_parquet(dir_ / "targets.parquet")
        if self.order_price is not None:
            self.order_price.to_parquet(dir_ / "order_price.parquet")
        with open(dir_ / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f)

    @classmethod
    def load(cls, dir_: Path) -> "DataBundle":
        """Load bundle contents from a directory."""
        features = pd.read_parquet(dir_ / "features.parquet")
        targets = pd.read_parquet(dir_ / "targets.parquet")
        op_path = dir_ / "order_price.parquet"
        op = pd.read_parquet(op_path) if op_path.exists() else None
        meta_path = dir_ / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        return cls(features=features, targets=targets, order_price=op, meta=meta)

