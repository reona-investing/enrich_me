"""汎用的なリターン分析ユーティリティ。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import ipywidgets as widgets
from IPython.display import display


@dataclass
class ReturnDataHandler:
    """リターンデータを管理するヘルパー。

    パラメータ
    ----------
    returns : pd.DataFrame
        リターンを含むデータフレーム。 ``date_col`` と ``return_col`` で指定
        した列が必要。
    date_col : str, default "日付"
        日付を示す列名。
    return_col : str, default "リターン"
        リターンを示す列名。
    start_date : datetime, optional
        抽出する開始日。
    end_date : datetime, optional
        抽出する終了日。
    """

    returns: pd.DataFrame
    date_col: str = "日付"
    return_col: str = "リターン"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        df = self.returns.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.set_index(self.date_col, drop=True).sort_index()
        if self.start_date is not None:
            df = df[df.index >= self.start_date]
        if self.end_date is not None:
            df = df[df.index <= self.end_date]
        self.return_df = df[[self.return_col]].rename(columns={self.return_col: "リターン"})


class ReturnMetricsCalculator:
    """リターン系列から各種指標を計算するクラス。"""

    def __init__(self, handler: ReturnDataHandler, *, tax_rate: float = 0.20315, leverage: float = 3.1) -> None:
        self.df = handler.return_df
        self.tax_rate = tax_rate
        self.leverage = leverage
        self.metrics: dict[str, pd.DataFrame] = {}
        self._calculate_all()

    def _calculate_all(self) -> None:
        self.calculate_daily_returns()
        self.calculate_cumulative_returns()
        self.calculate_monthly_returns()

    def calculate_daily_returns(self) -> pd.DataFrame:
        df = self.df.copy()
        df["税引後リターン"] = df["リターン"].apply(lambda x: x * (1 - self.tax_rate) if x > 0 else x)
        df["レバレッジ込リターン"] = df["税引後リターン"] * self.leverage

        summary = pd.DataFrame(
            index=[
                "日次平均リターン",
                "年率換算リターン",
                "日次リターン標準偏差",
                "年率換算標準偏差",
                "シャープレシオ",
                "最大ドローダウン（実績）",
                "最大ドローダウン（理論）",
            ]
        )

        for col in ["リターン", "税引後リターン", "レバレッジ込リターン"]:
            mean = df[col].mean()
            std = df[col].std(ddof=0)
            annual_return = mean * 252
            annual_std = std * (252 ** 0.5)
            sharpe = annual_return / annual_std if annual_std != 0 else float("nan")

            cum = (1 + df[col]).cumprod() - 1
            dd = 1 - (1 + cum) / (1 + cum).cummax()
            max_dd = dd.max()
            theo_dd = (std ** 2) / mean * 9 / 4 if mean != 0 else float('nan')

            summary[col] = [
                mean,
                annual_return,
                std,
                annual_std,
                sharpe,
                max_dd,
                theo_dd,
            ]

        self.metrics["日次成績"] = df
        self.metrics["日次成績（集計）"] = summary
        return df

    def calculate_cumulative_returns(self) -> pd.DataFrame:
        cols = ["リターン", "税引後リターン", "レバレッジ込リターン"]
        cum_df = pd.DataFrame(index=self.metrics["日次成績"].index)
        for col in cols:
            daily_df = self.metrics["日次成績"][col]
            cum = (1 + daily_df).cumprod() - 1
            dd = 1 - (1 + cum) / (1 + cum).cummax()
            cum_df[f"累積{col}"] = cum
            cum_df[f"ドローダウン{col}"] = dd
            cum_df[f"最大ドローダウン{col}"] = dd.cummax()
            mean = daily_df.mean()
            std = daily_df.std(ddof=0)
            theo_dd = (std ** 2) / mean * 9 / 4 if mean != 0 else float('nan')
            cum_df[f"理論最大ドローダウン{col}"] = theo_dd

        self.metrics["累積成績"] = cum_df
        return cum_df

    def calculate_monthly_returns(self) -> pd.DataFrame:
        monthly_df = pd.DataFrame(index=pd.to_datetime(self.metrics["日次成績"].index).to_period("M").to_timestamp("M"))
        for col in ["リターン", "税引後リターン", "レバレッジ込リターン"]:
            monthly = (1 + self.metrics["日次成績"][col]).resample("ME").prod() - 1
            monthly_df[f"月次{col}"] = monthly
            monthly_df[f"累積{col}"] = (1 + monthly_df[f"月次{col}"]).cumprod() - 1

        self.metrics["月次成績"] = monthly_df
        return monthly_df


class ReturnVisualizer:
    """Display calculated metrics using widgets."""

    def __init__(self, calculator: ReturnMetricsCalculator) -> None:
        self.metrics = calculator.metrics

    def display_result(self) -> None:
        dropdown = widgets.Dropdown(options=self.metrics.keys(), description="選択：")
        button = widgets.Button(description="表示")
        output = widgets.Output()

        def on_button_click(_):
            df = self.metrics[dropdown.value]
            with output:
                output.clear_output()
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                display(df)
                pd.reset_option("display.max_rows")
                pd.reset_option("display.max_columns")

        button.on_click(on_button_click)
        display(widgets.HBox([dropdown, button]), output)

