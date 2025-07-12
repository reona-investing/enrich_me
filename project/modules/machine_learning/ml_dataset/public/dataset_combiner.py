from dataclasses import dataclass, field
from typing import Iterable, Tuple, List, Dict
import pandas as pd

from .ml_dataset import MLDataset
from utils.timeseries import Duration


@dataclass
class DatasetPeriodInfo:
    """データセットと適用期間、およびモデル名を保持するメタデータ"""

    dataset: MLDataset
    period: Duration

    @property
    def model_names(self) -> List[str]:
        assets = self.dataset.ml_assets
        if isinstance(assets, list):
            return [asset.name for asset in assets]
        return [assets.name]

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": self.period.start,
            "end": self.period.end,
            "models": self.model_names,
        }


@dataclass
class DatasetPeriodCombiner:
    """期間ごとに ``MLDataset`` を結合しメタデータも保持するクラス"""

    dataset_periods: Iterable[Tuple[MLDataset, Duration]] = field(default_factory=list)

    def combine(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """登録済みデータセットを結合して ``DataFrame`` を返す

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                0番目に ``pred_result_df``、1番目に ``raw_returns_df`` を
                それぞれ時系列順に結合した ``DataFrame`` のタプル
        """
        pred_list = []
        raw_list = []

        for ds, period in self.dataset_periods:
            pred_df = period.extract_from_df(ds.pred_result_df, ds.date_column)
            pred_list.append(pred_df)

            raw_df = period.extract_from_df(ds.raw_returns_df, ds.date_column)
            raw_list.append(raw_df)

        combined_pred = pd.concat(pred_list).sort_index()
        combined_raw = pd.concat(raw_list).sort_index()

        return combined_pred, combined_raw

    @property
    def metadata(self) -> List[Dict[str, object]]:
        """結合に利用した期間とモデル名のメタデータを返す"""
        return [DatasetPeriodInfo(ds, period).to_dict() for ds, period in self.dataset_periods]


@dataclass
class WeightedDatasetInfo:
    """重み付き平均に利用するデータセットと重みを保持するメタデータ"""

    dataset: MLDataset
    weight: float

    @property
    def model_names(self) -> List[str]:
        assets = self.dataset.ml_assets
        if isinstance(assets, list):
            return [asset.name for asset in assets]
        return [assets.name]

    def to_dict(self) -> Dict[str, object]:
        return {
            "weight": self.weight,
            "models": self.model_names,
        }


@dataclass
class WeightedDatasetCombiner:
    """複数の ``MLDataset`` を割合で平均して結合するクラス"""

    dataset_weights: Iterable[WeightedDatasetInfo] = field(default_factory=list)

    def combine(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """登録済みデータセットを重み付き平均で結合して ``DataFrame`` を返す

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                0番目に ``pred_result_df``、1番目に ``raw_returns_df`` を
                それぞれ重み付き平均した ``DataFrame`` のタプル
        """

        weighted_pred: pd.DataFrame | None = None
        weighted_raw: pd.DataFrame | None = None
        total = 0.0

        for info in self.dataset_weights:
            w = info.weight
            pred_df = info.dataset.pred_result_df * w
            raw_df = info.dataset.raw_returns_df * w

            if weighted_pred is None:
                weighted_pred = pred_df
                weighted_raw = raw_df
            else:
                weighted_pred = weighted_pred.add(pred_df, fill_value=0)
                weighted_raw = weighted_raw.add(raw_df, fill_value=0)

            total += w

        if weighted_pred is None:
            return pd.DataFrame(), pd.DataFrame()

        combined_pred = (weighted_pred / total).sort_index()
        combined_raw = (weighted_raw / total).sort_index()

        return combined_pred, combined_raw

    @property
    def metadata(self) -> List[Dict[str, object]]:
        """重みとモデル名のメタデータを返す"""
        return [info.to_dict() for info in self.dataset_weights]


@dataclass
class DatasetCombinePipeline:
    """複数の結合処理を順番に実行するパイプライン"""

    steps: Iterable[object] = field(default_factory=list)
    _metadata: List[List[Dict[str, object]]] = field(init=False, default_factory=list)

    def combine(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """登録された各ステップを順に実行して ``DataFrame`` を返す"""

        pred_df: pd.DataFrame | None = None
        raw_df: pd.DataFrame | None = None
        self._metadata = []

        for step in self.steps:
            pred_df, raw_df = step.combine()
            self._metadata.append(step.metadata)

        if pred_df is None or raw_df is None:
            return pd.DataFrame(), pd.DataFrame()

        return pred_df, raw_df

    @property
    def metadata(self) -> List[List[Dict[str, object]]]:
        """各ステップで生成されたメタデータの一覧を返す"""
        return self._metadata
