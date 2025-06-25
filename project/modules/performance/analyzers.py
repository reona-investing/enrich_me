"""リターン評価アナライザー"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .transformations import TaxRate, Leverage
from .annualizer import Annualizer
from .metrics import EvaluationMetricsManager, SpearmanCorrelation


@dataclass
class ReturnSeriesTransformer:
    """リターン系列の変換を扱うクラス"""

    tax_rate_obj: TaxRate
    leverage_obj: Leverage

    def get_pre_tax_pre_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        return raw_returns

    def get_pre_tax_post_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        return self.leverage_obj.apply_leverage(raw_returns)

    def get_post_tax_post_leverage_returns(self, raw_returns: pd.Series) -> pd.Series:
        leveraged = self.leverage_obj.apply_leverage(raw_returns)
        return self.tax_rate_obj.apply_tax(leveraged)

    def get_pre_tax_pre_leverage_from_post_tax_post_leverage(self, post_tax_post_leverage_returns: pd.Series) -> pd.Series:
        untaxed = self.tax_rate_obj.remove_tax(post_tax_post_leverage_returns)
        return self.leverage_obj.remove_leverage(untaxed)


class PredictionReturnAnalyzer:
    """予測リターンと実際のリターンを評価するクラス"""

    def __init__(self, raw_predicted_returns: pd.Series, actual_returns: pd.Series,
                 transformer: ReturnSeriesTransformer, evaluation_manager: EvaluationMetricsManager) -> None:
        if not raw_predicted_returns.index.equals(actual_returns.index):
            raise ValueError("インデックスが一致していません")
        self.raw_predicted = raw_predicted_returns
        self.actual = actual_returns
        self.transformer = transformer
        self.manager = evaluation_manager

    def run_analysis(self) -> pd.DataFrame:
        patterns = {
            "PreTax_PreLeverage": self.transformer.get_pre_tax_pre_leverage_returns,
            "PostTax_PostLeverage": self.transformer.get_post_tax_post_leverage_returns,
        }
        dfs = {}
        for name, func in patterns.items():
            transformed = func(self.raw_predicted)
            result = self.manager.evaluate_all(transformed)
            # 相関は別途計算し平均値を格納
            corr_metric = SpearmanCorrelation()
            corr_metric.calculate(transformed, series2=self.actual)
            corr_df = corr_metric.value
            corr_mean = float("nan")
            if "SpearmanCorr" in corr_df.index:
                corr_mean = corr_df.loc["SpearmanCorr", "mean"]
            result[corr_metric.get_name()] = corr_mean
            dfs[name] = pd.Series(result)
        return pd.DataFrame(dfs)


class TradeResultAnalyzer:
    """実際の取引結果を評価するクラス"""

    def __init__(self, raw_trade_df: pd.DataFrame, transformer: ReturnSeriesTransformer,
                 evaluation_manager: EvaluationMetricsManager) -> None:
        if not {"Date", "Symbol", "DailyReturn"}.issubset(raw_trade_df.columns):
            raise ValueError("必要なカラムが存在しません")
        self.raw_trade_df = raw_trade_df.copy()
        self.raw_trade_df["Date"] = pd.to_datetime(self.raw_trade_df["Date"])
        self.transformer = transformer
        self.manager = evaluation_manager

    def run_analysis(self) -> Dict[str, pd.DataFrame]:
        grouped = self.raw_trade_df.groupby("Symbol")
        pre_results = {}
        post_results = {}
        for symbol, df in grouped:
            series = df.sort_values("Date")["DailyReturn"]
            post_tax_post_lev = series
            pre_tax_pre_lev = self.transformer.get_pre_tax_pre_leverage_from_post_tax_post_leverage(series)
            pre_results[symbol] = pd.Series(self.manager.evaluate_all(pre_tax_pre_lev))
            post_results[symbol] = pd.Series(self.manager.evaluate_all(self.transformer.get_post_tax_post_leverage_returns(pre_tax_pre_lev)))
        pre_df = pd.DataFrame(pre_results).T
        post_df = pd.DataFrame(post_results).T
        return {
            "PreTax_PreLeverage": pre_df,
            "PostTax_PostLeverage": post_df,
        }
