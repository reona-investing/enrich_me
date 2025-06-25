from __future__ import annotations

from typing import Dict
import pandas as pd
import ipywidgets as widgets
from IPython.display import display


class MetricsInteractiveViewer:
    """Interactively display calculated metrics."""

    def __init__(self, metrics: Dict[str, float]) -> None:
        self.metrics = metrics

    def display(self) -> None:
        """Show each metric interactively via dropdown."""
        options = list(self.metrics.keys())
        dropdown = widgets.Dropdown(options=options, description="Metric:")
        button = widgets.Button(description="Show")
        output = widgets.Output()

        def on_click(_):
            metric = dropdown.value
            with output:
                output.clear_output()
                print(f"{metric}: {self.metrics[metric]}")

        button.on_click(on_click)
        display(widgets.HBox([dropdown, button]), output)

    def display_table(self) -> None:
        """Display all metrics at once as a DataFrame."""
        df = pd.DataFrame.from_dict(self.metrics, orient="index", columns=["Value"])
        display(df)
