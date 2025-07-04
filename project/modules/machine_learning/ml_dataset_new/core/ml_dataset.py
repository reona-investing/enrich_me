from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from typing import Optional, Union, List, Dict
import json
import pickle

from machine_learning.ml_dataset.components import MLModel, TrainTestData
from machine_learning.models import BaseTrainer
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
    """
    学習用データ（target_df + features_df）の保持・管理および前処理を担当するクラス
    """

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
            metadata=self.metadata
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
            reuse_features_df=False
        )


@dataclass
class MLOutputCollection:
    """機械学習の出力データ群を管理するクラス"""

    raw_returns_df: pd.DataFrame = field(repr=False)
    pred_result_df: pd.DataFrame = field(repr=False)
    order_price_df: pd.DataFrame = field(repr=False)

    def update_outputs(
        self,
        new_raw_returns: Optional[pd.DataFrame] = None,
        new_pred_result: Optional[pd.DataFrame] = None,
        new_order_price: Optional[pd.DataFrame] = None,
    ) -> "MLOutputCollection":
        """出力データを更新したインスタンスを返す。"""
        return MLOutputCollection(
            raw_returns_df=new_raw_returns.sort_index() if new_raw_returns is not None else self.raw_returns_df,
            pred_result_df=new_pred_result.sort_index() if new_pred_result is not None else self.pred_result_df,
            order_price_df=new_order_price.sort_index() if new_order_price is not None else self.order_price_df,
        )


@dataclass
class MLAssetsMetadata:
    """MLModelの管理方針に関する設定"""
    is_model_divided: bool


@dataclass
class MLAssetsContainer:
    """MLModelの管理を担当するクラス"""

    assets: Union[MLModel, List[MLModel]] = field(default_factory=list)
    metadata: MLAssetsMetadata = field(repr=False)

    def train_models(
        self,
        trainer: BaseTrainer,
        target_train: pd.DataFrame,
        features_train: pd.DataFrame,
        sector_column: str,
        **kwargs
    ) -> None:
        """モデルを学習する。"""
        if self.metadata.is_model_divided:
            self.assets = []
            sectors = target_train.index.get_level_values(sector_column).unique()

            for sector in sectors:
                print(f"セクター '{sector}' のモデルを学習中...")
                sector_mask = target_train.index.get_level_values(sector_column) == sector
                sector_target = target_train[sector_mask].copy()
                sector_features = features_train[sector_mask].copy()
                ml_asset = trainer.train(model_name=sector, target_df=sector_target, features_df=sector_features, **kwargs)
                self.assets.append(ml_asset)
        else:
            print("単一モデルを学習中...")
            ml_asset = trainer.train(model_name='Global', target_df=target_train, features_df=features_train, **kwargs)
            self.assets = ml_asset

    def predict(
        self,
        target_test: pd.DataFrame,
        features_test: pd.DataFrame,
        sector_column: str
    ) -> pd.DataFrame:
        """予測を実行する。"""
        if isinstance(self.assets, list):
            print("複数モデルで予測中...")
            all_predictions_df = []
            for ml_asset_item in self.assets:
                print(f"セクター '{ml_asset_item.name}' で予測中...")
                sector_mask = target_test.index.get_level_values(sector_column) == ml_asset_item.name
                target_sector = target_test[sector_mask].copy()
                features_sector = features_test[sector_mask].copy()
                predictions_sector = ml_asset_item.predict(features_sector)
                predictions_sector = pd.concat([target_sector, predictions_sector], axis=1)
                all_predictions_df.append(predictions_sector)
            return pd.concat(all_predictions_df, axis=0).sort_index()
        else:
            print("単一モデルで予測中...")
            predictions = self.assets.predict(features_test)
            return pd.concat([target_test, predictions], axis=1).sort_index()


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
            new_order_price=new_order_price
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
            **kwargs
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
            sector_column=self.resource_data.metadata.sector_column
        )

        self.output_collection = self.output_collection.update_outputs(
            new_pred_result=pred_result_df
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


#=====================================
# セーブ・ロード用のクラス群
#=====================================

class MLResourceStorage:
    """MLResourceインスタンスのセーブ・ロードを司るクラス"""

    _TARGET_FILE = "target_df.parquet"
    _FEATURE_FILE = "features_df.parquet"
    _METADATA_FILE = "resource_metadata.json"
    _OLD_METADATA_FILE = "metadata.json"

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Dict[str, Path]:
        return {
            "target_df": self.base_path / self._TARGET_FILE,
            "features_df": self.base_path / self._FEATURE_FILE,
            "metadata": self.base_path / self._METADATA_FILE,
            "metadata_old": self.base_path / self._OLD_METADATA_FILE,
        }

    def load(self) -> MLResource:
        """外部ファイルをロードしてMLResourceを作成します。旧形式のmetadataにも対応します。"""
        metadata_path = self.path["metadata"]
        if not metadata_path.exists() and self.path["metadata_old"].exists():
            metadata_path = self.path["metadata_old"]
        with metadata_path.open(encoding="utf-8") as f:
            metadata_dict = json.load(f)

        if "train_duration" in metadata_dict:
            pass
        else:
            metadata_dict["train_duration"] = Duration(
                start=pd.to_datetime(metadata_dict.pop("train_start"), unit='ms'),
                end=pd.to_datetime(metadata_dict.pop("train_end"), unit='ms')
            )
            metadata_dict["test_duration"] = Duration(
                start=pd.to_datetime(metadata_dict.pop("test_start"), unit='ms'),
                end=pd.to_datetime(metadata_dict.pop("test_end"), unit='ms')
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


class MLOutputCollectionStorage:
    """MLOutputCollectionインスタンスのセーブ・ロードを司るクラス"""

    _RAW_RETURN_FILE = "raw_returns_df.parquet"
    _PRED_RESULT_FILE = "pred_result_df.parquet"
    _ORDER_PRICE_FILE = "order_price_df.parquet"

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Dict[str, Path]:
        return {
            "raw_returns_df": self.base_path / self._RAW_RETURN_FILE,
            "pred_result_df": self.base_path / self._PRED_RESULT_FILE,
            "order_price_df": self.base_path / self._ORDER_PRICE_FILE,
        }

    def load(self) -> MLOutputCollection:
        """外部ファイルをロードしてMLOutputCollectionを作成します。"""
        return MLOutputCollection(
            raw_returns_df=pd.read_parquet(self.path["raw_returns_df"]),
            pred_result_df=pd.read_parquet(self.path["pred_result_df"]),
            order_price_df=pd.read_parquet(self.path["order_price_df"]),
        )

    def save(self, output_collection: MLOutputCollection) -> None:
        """MLOutputCollectionのプロパティを外部ファイルに出力します。"""
        self._atomic_write_parquet(output_collection.raw_returns_df, self.path["raw_returns_df"])
        self._atomic_write_parquet(output_collection.pred_result_df, self.path["pred_result_df"])
        self._atomic_write_parquet(output_collection.order_price_df, self.path["order_price_df"])

    @staticmethod
    def _atomic_write_parquet(obj: pd.DataFrame, dest: Path, compression: str = "zstd") -> None:
        if obj is not None:
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            obj.to_parquet(tmp_path, compression=compression)
            tmp_path.replace(dest)


class MLAssetsContainerStorage:
    """MLAssetsContainerインスタンスのセーブ・ロードを司るクラス"""

    _ML_ASSETS_FILE = "ml_assets.pkl"
    _ASSETS_METADATA_FILE = "assets_metadata.json"
    _OLD_METADATA_FILE = "metadata.json"

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Dict[str, Path]:
        return {
            "ml_assets": self.base_path / self._ML_ASSETS_FILE,
            "metadata": self.base_path / self._ASSETS_METADATA_FILE,
            "metadata_old": self.base_path / self._OLD_METADATA_FILE,
        }

    def load(self) -> MLAssetsContainer:
        """外部ファイルをロードしてMLAssetsContainerを作成します。旧形式のmetadataにも対応します。"""
        with self.path["ml_assets"].open("rb") as f:
            assets = pickle.load(f)

        metadata_path = self.path["metadata"]
        if not metadata_path.exists() and self.path["metadata_old"].exists():
            metadata_path = self.path["metadata_old"]

        with metadata_path.open(encoding="utf-8") as f:
            metadata_dict = json.load(f)

        metadata = MLAssetsMetadata(**metadata_dict)

        return MLAssetsContainer(
            assets=assets,
            metadata=metadata,
        )

    def save(self, ml_assets_container: MLAssetsContainer) -> None:
        """MLAssetsContainerを外部ファイルに出力します。"""
        if ml_assets_container.assets is not None:
            tmp_path = self.path["ml_assets"].with_suffix(self.path["ml_assets"].suffix + ".tmp")
            with tmp_path.open("wb") as f:
                pickle.dump(ml_assets_container.assets, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(self.path["ml_assets"])

        metadata_dict = {
            "is_model_divided": ml_assets_container.metadata.is_model_divided,
        }
        pd.Series(metadata_dict).to_json(self.path["metadata"], indent=2, force_ascii=False)


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
