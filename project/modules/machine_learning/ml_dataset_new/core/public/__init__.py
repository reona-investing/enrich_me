from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List
import pandas as pd

from machine_learning.ml_dataset.components import MLModel, TrainTestData
from machine_learning.models import BaseTrainer

from ..resources import MLResource, MLResourceMetadata, MLResourceStorage
from ..outputs import MLOutputCollection, MLOutputCollectionStorage
from ..models import (
    MLAssetsContainer,
    MLAssetsMetadata,
    MLAssetsContainerStorage,
)
from utils.timeseries import Duration


@dataclass
class MLDataset:
    """機械学習用データセットを統合的に管理するファサードクラス"""

    dataset_path: Path | str
    resource_data: MLResource = field(repr=False)
    output_collection: MLOutputCollection = field(repr=False)
    ml_assets_container: MLAssetsContainer = field(repr=False)

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)

    # ---------------------------------------------------------------------
    # 互換性のためのプロパティ
    # ---------------------------------------------------------------------
    @property
    def target_df(self) -> pd.DataFrame:
        return self.resource_data.target_df

    @property
    def features_df(self) -> pd.DataFrame:
        return self.resource_data.features_df

    @property
    def raw_returns_df(self) -> pd.DataFrame:
        return self.output_collection.raw_returns_df

    @property
    def pred_result_df(self) -> pd.DataFrame:
        return self.output_collection.pred_result_df

    @property
    def order_price_df(self) -> pd.DataFrame:
        return self.output_collection.order_price_df

    @property
    def train_duration(self) -> Duration:
        return self.resource_data.metadata.train_duration

    @property
    def test_duration(self) -> Duration:
        return self.resource_data.metadata.test_duration

    @property
    def date_column(self) -> str:
        return self.resource_data.metadata.date_column

    @property
    def sector_column(self) -> str:
        return self.resource_data.metadata.sector_column

    @property
    def is_model_divided(self) -> bool:
        return self.ml_assets_container.metadata.is_model_divided

    @property
    def outlier_threshold(self) -> Union[int, float]:
        return self.resource_data.metadata.outlier_threshold

    @property
    def no_shift_features(self) -> List[str]:
        return self.resource_data.metadata.no_shift_features

    @property
    def ml_assets(self) -> Union[MLModel, List[MLModel]]:
        return self.ml_assets_container.assets

    # ---------------------------------------------------------------------
    # ファクトリメソッド
    # ---------------------------------------------------------------------
    @classmethod
    def from_files(cls, dataset_path: str | Path) -> "MLDataset":
        """ファクトリ：指定パスからファイルを読み込み、インスタンスを生成する。"""
        return MLDatasetStorage(dataset_path).load()

    @classmethod
    def from_raw(
        cls,
        dataset_path: str | Path,
        target_df: pd.DataFrame,
        features_df: pd.DataFrame,
        raw_returns_df: pd.DataFrame,
        pred_return_df: pd.DataFrame,
        order_price_df: pd.DataFrame,
        train_duration: Duration,
        test_duration: Duration,
        date_column: str,
        sector_column: str,
        is_model_divided: bool,
        ml_assets: Union[MLModel, List[MLModel]],
        outlier_threshold: int | float,
        no_shift_features: List[str],
        save: bool = True,
    ) -> "MLDataset":
        """ファクトリ：既存オブジェクトからインスタンスを構築する。"""
        resource_metadata = MLResourceMetadata(
            train_duration=train_duration,
            test_duration=test_duration,
            date_column=date_column,
            sector_column=sector_column,
            outlier_threshold=outlier_threshold,
            no_shift_features=no_shift_features,
        )

        resource_data = MLResource(
            target_df=target_df,
            features_df=features_df,
            metadata=resource_metadata,
        )

        output_collection = MLOutputCollection(
            raw_returns_df=raw_returns_df,
            pred_result_df=pred_return_df,
            order_price_df=order_price_df,
        )

        assets_metadata = MLAssetsMetadata(
            is_model_divided=is_model_divided,
        )

        ml_assets_container = MLAssetsContainer(
            assets=ml_assets,
            metadata=assets_metadata,
        )

        ds = cls(
            dataset_path=dataset_path,
            resource_data=resource_data,
            output_collection=output_collection,
            ml_assets_container=ml_assets_container,
        )

        if save:
            ds.save()
        return ds

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def update_data(
        self,
        new_targets: pd.DataFrame,
        new_features: pd.DataFrame,
        new_raw_returns: Optional[pd.DataFrame] = None,
        new_order_price: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> None:
        """新しい日次データを追加する。"""
        self.resource_data = self.resource_data.update_data(new_targets, new_features)

        self.output_collection = self.output_collection.update_outputs(
            new_raw_returns=new_raw_returns,
            new_order_price=new_order_price,
        )

        if save:
            self.save()

    def train(self, trainer: BaseTrainer, save: bool = True, **kwargs):
        """学習期間のデータを用いてモデルを学習する。"""
        print("学習を開始します...")

        index_cols = [self.resource_data.metadata.date_column]
        if self.resource_data.metadata.sector_column:
            index_cols.append(self.resource_data.metadata.sector_column)

        ttd = self.resource_data.split_train_test()

        target_train = ttd.target_train_df.reset_index(drop=False).set_index(index_cols, drop=True)
        features_train = ttd.features_train_df.reset_index(drop=False).set_index(index_cols, drop=True)

        self.ml_assets_container.train_models(
            trainer=trainer,
            target_train=target_train,
            features_train=features_train,
            sector_column=self.resource_data.metadata.sector_column,
            **kwargs,
        )

        if save:
            self.save()
        print("学習が完了しました。")

    def predict(self, save: bool = True):
        """テスト期間のデータを用いて予測を行う。"""
        print("予測を開始します...")

        if not self.ml_assets_container.assets:
            raise ValueError("モデルが学習されていません。またはロードパスが指定されていません。")

        ttd = self.resource_data.split_train_test()

        index_cols = ttd.target_test_df.index.names
        target_test = ttd.target_test_df.reset_index(drop=False).set_index(index_cols, drop=True)
        features_test = ttd.features_test_df.reset_index(drop=False).set_index(index_cols, drop=True)

        pred_result_df = self.ml_assets_container.predict(
            target_test=target_test,
            features_test=features_test,
            sector_column=self.resource_data.metadata.sector_column,
        )

        self.output_collection = self.output_collection.update_outputs(
            new_pred_result=pred_result_df,
        )

        if save:
            self.save()
        print("予測が完了しました。")

    # ------------------------------------------------------------------
    # ヘルパーメソッド
    # ------------------------------------------------------------------
    def save(self) -> None:
        """現在のインスタンスを ``dataset_path`` 配下に保存する。"""
        MLDatasetStorage(self.dataset_path).save(self)

    def _split_train_test(self) -> TrainTestData:
        """互換性のために残されたメソッド。"""
        return self.resource_data.split_train_test()


class MLDatasetStorage:
    """MLDatasetインスタンスのセーブ・ロードを司るクラス"""

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.resource_storage = MLResourceStorage(self.base_path)
        self.output_storage = MLOutputCollectionStorage(self.base_path)
        self.assets_storage = MLAssetsContainerStorage(self.base_path)

    def load(self) -> MLDataset:
        """外部ファイルをロードしてMLDatasetを作成します。"""
        resource_data = self.resource_storage.load()
        output_collection = self.output_storage.load()
        ml_assets_container = self.assets_storage.load()

        return MLDataset(
            dataset_path=self.base_path,
            resource_data=resource_data,
            output_collection=output_collection,
            ml_assets_container=ml_assets_container,
        )

    def save(self, ds: MLDataset) -> None:
        """MLDatasetのプロパティを外部ファイルに出力します。"""
        self.resource_storage.save(ds.resource_data)
        self.output_storage.save(ds.output_collection)
        self.assets_storage.save(ds.ml_assets_container)

