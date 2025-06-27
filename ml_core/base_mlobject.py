from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class BaseMLObject(Protocol):
    """Protocol that all machine learning objects must follow."""

    def fit(self, data: 'DataBundle') -> None:
        """Fit the model using the provided data."""

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return predictions for the given features."""

    def save(self, path: Path) -> None:
        """Serialize the ML object to the given path."""

    @classmethod
    def load(cls, path: Path) -> 'BaseMLObject':
        """Load the ML object from a file."""

