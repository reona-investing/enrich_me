from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Dict
import json
import pandas as pd

from machine_learning.ml_dataset.components import TrainTestData
from utils.timeseries import Duration


@dataclass
class MLResourceMetadata:
    """MLResourceの前処理・分割に関する設定"""

    train_duration: Duration = field(repr=False)
    test_duration: Duration = field(repr=False)
    date_column: str = field(repr=False)
    sector_column: str = field(repr=False)
    outlier_threshold: Union[int, float] = field(repr=False)
    no_shift_features: List[str] = field(default_factory=list)


@dataclass
class MLResource:
    """学習用データ（target_df + features_df）の保持・管理および前処理を担当するクラス"""

    target_df: pd.DataFrame = field(repr=False)
    features_df: pd.DataFrame = field(repr=False)
    metadata: MLResourceMetadata = field(repr=False)

    def update_data(
        self,
        new_targets: pd.DataFrame,
        new_features: pd.DataFrame,
    ) -> "MLResource":
        """新しい日次データで更新されたインスタンスを返す。"""
        return MLResource(
            target_df=new_targets.sort_index(),
            features_df=new_features.sort_index(),
            metadata=self.metadata,
        )

    def split_train_test(self) -> TrainTestData:
        """訓練期間・テスト期間にデータを分割し、外れ値処理等の前処理を行います。"""
        return TrainTestData().archive(
            target_df=self.target_df,
            features_df=self.features_df,
            train_duration=self.metadata.train_duration,
            test_duration=self.metadata.test_duration,
            datetime_column=self.metadata.date_column,
            outlier_threshold=self.metadata.outlier_threshold,
            no_shift_features=self.metadata.no_shift_features,
            reuse_features_df=False,
        )


class MLResourceStorage:
    """MLResourceインスタンスのセーブ・ロードを司るクラス"""

    _TARGET_FILE = "target_df.parquet"
    _FEATURE_FILE = "features_df.parquet"
    _METADATA_FILE = "resource_metadata.json"

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Dict[str, Path]:
        return {
            "target_df": self.base_path / self._TARGET_FILE,
            "features_df": self.base_path / self._FEATURE_FILE,
            "metadata": self.base_path / self._METADATA_FILE,
        }

    def load(self) -> MLResource:
        """外部ファイルをロードしてMLResourceを作成します。旧形式のmetadataにも対応します。"""
        metadata_path = self.path["metadata"]
        with metadata_path.open(encoding="utf-8") as f:
            metadata_dict = json.load(f)

        if "train_duration" not in metadata_dict:
            metadata_dict["train_duration"] = Duration(
                start=pd.to_datetime(metadata_dict.pop("train_start"), unit="ms"),
                end=pd.to_datetime(metadata_dict.pop("train_end"), unit="ms"),
            )
            metadata_dict["test_duration"] = Duration(
                start=pd.to_datetime(metadata_dict.pop("test_start"), unit="ms"),
                end=pd.to_datetime(metadata_dict.pop("test_end"), unit="ms"),
            )

        metadata = MLResourceMetadata(**metadata_dict)

        return MLResource(
            target_df=pd.read_parquet(self.path["target_df"]),
            features_df=pd.read_parquet(self.path["features_df"]),
            metadata=metadata,
        )

    def save(self, resource_data: MLResource) -> None:
        """MLResourceのプロパティを外部ファイルに出力します。"""
        self._atomic_write_parquet(resource_data.target_df, self.path["target_df"])
        self._atomic_write_parquet(resource_data.features_df, self.path["features_df"])

        metadata_dict = {
            "train_start": resource_data.metadata.train_duration.start,
            "train_end": resource_data.metadata.train_duration.end,
            "test_start": resource_data.metadata.test_duration.start,
            "test_end": resource_data.metadata.test_duration.end,
            "date_column": resource_data.metadata.date_column,
            "sector_column": resource_data.metadata.sector_column,
            "outlier_threshold": resource_data.metadata.outlier_threshold,
            "no_shift_features": resource_data.metadata.no_shift_features,
        }
        pd.Series(metadata_dict).to_json(self.path["metadata"], indent=2, force_ascii=False)

    @staticmethod
    def _atomic_write_parquet(obj: pd.DataFrame, dest: Path, compression: str = "zstd") -> None:
        if obj is not None:
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            obj.to_parquet(tmp_path, compression=compression)
            tmp_path.replace(dest)