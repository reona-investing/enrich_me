from __future__ import annotations

from pathlib import Path
import pickle
from typing import Optional

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

from .base_mlobject import BaseMLObject
from .data_bundle import DataBundle


class UnifiedMLObject(BaseMLObject):
    """Single model approach with optional scaler."""

    def __init__(self, model: Optional[LGBMRegressor] = None, scaler: Optional[StandardScaler] = None) -> None:
        self.model = model or LGBMRegressor()
        self.scaler = scaler

    def fit(self, data: DataBundle) -> None:
        X = data.features
        y = data.targets.squeeze()
        if self.scaler is not None:
            X = pd.DataFrame(self.scaler.fit_transform(X), index=X.index, columns=X.columns)
        self.model.fit(X, y)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features
        if self.scaler is not None:
            X = pd.DataFrame(self.scaler.transform(X), index=X.index, columns=X.columns)
        preds = self.model.predict(X)
        return pd.DataFrame(preds, index=features.index, columns=["Pred"])

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    @classmethod
    def load(cls, path: Path) -> "UnifiedMLObject":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(model=obj.get("model"), scaler=obj.get("scaler"))

