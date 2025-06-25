from __future__ import annotations

from typing import Dict, Union
import pandas as pd
import ipywidgets as widgets
from IPython.display import display


class MetricsInteractiveViewer:
    """Interactively display calculated metrics."""

    def __init__(self, metrics: Dict[str, Dict[str, Union[pd.DataFrame, float]]]) -> None:
        self.metrics = metrics

    def display(self) -> None:
        """Show metrics interactively via two dropdowns."""
        group_options = list(self.metrics.keys())
        group_dropdown = widgets.Dropdown(options=group_options, description="パターン:")
        metric_dropdown = widgets.Dropdown(description="Metric:")
        button = widgets.Button(description="Show")
        output = widgets.Output()

        def on_group_change(change):
            group = change["new"]
            metric_dropdown.options = list(self.metrics[group].keys())

        group_dropdown.observe(on_group_change, names="value")

        if group_options:
            metric_dropdown.options = list(self.metrics[group_options[0]].keys())

        def on_click(_):
            group = group_dropdown.value
            metric = metric_dropdown.value
            with output:
                output.clear_output()
                data = self.metrics[group][metric]
                with pd.option_context("display.max_rows", None):
                    if isinstance(data, pd.DataFrame):
                        print(f"{group}：{metric}")
                        display(data)
                    else:
                        print(f"{group}：{metric}: {data}")

        button.on_click(on_click)
        display(widgets.VBox([group_dropdown, widgets.HBox([metric_dropdown, button]), output]))

    def display_table(self) -> None:
        """Display all metrics for all patterns."""
        with pd.option_context("display.max_rows", None):
            for group, metrics in self.metrics.items():
                print(group)
                for name, data in metrics.items():
                    if isinstance(data, pd.DataFrame):
                        print(f"--- {name} ---")
                        display(data)
                    else:
                        print(f"{name}: {data}")
