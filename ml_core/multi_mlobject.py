from __future__ import annotations

from pathlib import Path
import pickle
from typing import Dict, Optional

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from .base_mlobject import BaseMLObject
from .data_bundle import DataBundle


class MultiMLObject(BaseMLObject):
    """Sector-wise models with corresponding scalers."""

    def __init__(self) -> None:
        self.models: Dict[str, Lasso] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    def fit(self, data: DataBundle) -> None:
        for sector in data.features.index.get_level_values("Sector").unique():
            X = data.features.xs(sector, level="Sector")
            y = data.targets.xs(sector, level="Sector").squeeze()
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
            model = Lasso()
            model.fit(X_scaled, y)
            self.models[sector] = model
            self.scalers[sector] = scaler

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sector in features.index.get_level_values("Sector").unique():
            X = features.xs(sector, level="Sector")
            scaler = self.scalers.get(sector)
            model = self.models.get(sector)
            if scaler is None or model is None:
                continue
            X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
            preds = model.predict(X_scaled)
            df = pd.DataFrame(preds, index=X.index, columns=["Pred"])
            results.append(df)
        return pd.concat(results).sort_index()

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "scalers": self.scalers}, f)

    @classmethod
    def load(cls, path: Path) -> "MultiMLObject":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        instance = cls()
        instance.models = obj.get("models", {})
        instance.scalers = obj.get("scalers", {})
        return instance

