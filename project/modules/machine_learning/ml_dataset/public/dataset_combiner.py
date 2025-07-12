from dataclasses import dataclass, field
from typing import Iterable, List, Dict
import pandas as pd

from utils.timeseries import Duration


@dataclass
class DatasetPeriodInfo:
    """予測結果DataFrameと期間、モデル名を保持するメタデータ"""

    pred_df: pd.DataFrame
    period: Duration
    model_names: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": self.period.start,
            "end": self.period.end,
            "models": self.model_names,
        }


@dataclass
class DatasetPeriodCombiner:
    """期間ごとに ``DataFrame`` を結合しメタデータも保持するクラス"""

    dataset_periods: Iterable[DatasetPeriodInfo] = field(default_factory=list)

    def combine(self) -> pd.DataFrame:
        """登録済みデータセットを結合して ``DataFrame`` を返す"""

        pred_list = []

        for info in self.dataset_periods:
            df = info.pred_df.copy()
            df.index = pd.to_datetime(df.index)
            filtered = df.loc[(df.index >= info.period.start) & (df.index <= info.period.end)]
            pred_list.append(filtered)

        if not pred_list:
            return pd.DataFrame()

        combined_pred = pd.concat(pred_list).sort_index()

        return combined_pred

    @property
    def metadata(self) -> List[Dict[str, object]]:
        """結合に利用した期間とモデル名のメタデータを返す"""
        return [info.to_dict() for info in self.dataset_periods]


@dataclass
class WeightedDatasetInfo:
    """重み付き平均に利用するデータセットと重みを保持するメタデータ"""

    pred_df: pd.DataFrame
    weight: float
    model_names: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "weight": self.weight,
            "models": self.model_names,
        }


@dataclass
class WeightedDatasetCombiner:
    """複数の ``DataFrame`` を割合で平均して結合するクラス"""

    dataset_weights: Iterable[WeightedDatasetInfo] = field(default_factory=list)

    def combine(self) -> pd.DataFrame:
        """登録済みデータセットを重み付き平均で結合して ``DataFrame`` を返す"""

        weighted_pred: pd.DataFrame | None = None
        total = 0.0

        for info in self.dataset_weights:
            w = info.weight
            pred_df = info.pred_df * w

            if weighted_pred is None:
                weighted_pred = pred_df
            else:
                weighted_pred = weighted_pred.add(pred_df, fill_value=0)

            total += w

        if weighted_pred is None:
            return pd.DataFrame()

        combined_pred = (weighted_pred / total).sort_index()

        return combined_pred

    @property
    def metadata(self) -> List[Dict[str, object]]:
        """重みとモデル名のメタデータを返す"""
        return [info.to_dict() for info in self.dataset_weights]


@dataclass
class DatasetCombinePipeline:
    """複数の結合処理を順番に実行するパイプライン"""

    steps: Iterable[object] = field(default_factory=list)
    _metadata: List[List[Dict[str, object]]] = field(init=False, default_factory=list)

    def combine(self) -> pd.DataFrame:
        """登録された各ステップを順に実行して ``DataFrame`` を返す"""

        pred_df: pd.DataFrame | None = None
        self._metadata = []

        for step in self.steps:
            pred_df = step.combine()
            self._metadata.append(step.metadata)

        if pred_df is None:
            return pd.DataFrame()

        return pred_df

    @property
    def metadata(self) -> List[List[Dict[str, object]]]:
        """各ステップで生成されたメタデータの一覧を返す"""
        return self._metadata
