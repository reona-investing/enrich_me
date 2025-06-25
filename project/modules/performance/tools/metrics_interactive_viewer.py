from __future__ import annotations

from typing import Dict
import pandas as pd
from IPython.display import display


class MetricsInteractiveViewer:
    """Interactively display calculated metrics."""

    def __init__(self, metrics: Dict[str, float]) -> None:
        self.metrics = metrics

    def display(self) -> None:
        df = pd.DataFrame.from_dict(self.metrics, orient="index", columns=["Value"]) 
        display(df)

