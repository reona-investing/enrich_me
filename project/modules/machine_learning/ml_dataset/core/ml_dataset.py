from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from typing import Optional, Union, List, Dict
import json
import pickle

from machine_learning.ml_dataset.components import MachineLearningAsset, TrainTestData
from machine_learning.models import BaseTrainer
from utils.timeseries import Duration


@dataclass
class MLDataset:
    """
    機械学習用データセットを統合的に管理するクラス
    
    Args:
        dataset_path (Path | str): データセットの保存先パス
        target_df (pd.DataFrame): 目的変数データフレーム
        features_df (pd.DataFrame): 特徴量データフレーム
        raw_returns_df (pd.DataFrame):
    """

    dataset_path: Path | str
    target_df: pd.DataFrame = field(repr=False)
    features_df: pd.DataFrame = field(repr=False)
    raw_returns_df: pd.DataFrame = field(repr=False)
    pred_result_df: pd.DataFrame = field(repr=False)
    order_price_df: pd.DataFrame = field(repr=False)

    train_duration: Duration = field(repr=False)
    test_duration: Duration = field(repr=False)
    date_column: str = field(repr=False)
    sector_column: str = field(repr=False)
    is_model_divided: bool = field(repr=False)
    outlier_threshold: int | float = field(repr=False)
    no_shift_features: list[str] = field(default_factory=list)

    ml_assets: Union[MachineLearningAsset, List[MachineLearningAsset]] = field(default_factory=list)

    # ---------------------------------------------------------------------
    # ファクトリメソッド
    # ---------------------------------------------------------------------
    @classmethod
    def from_files(cls, dataset_path: str | Path) -> "MLDataset":
        """
        ファクトリ：指定パスからファイルを読み込み、インスタンスを生成する。
        
        Args:
            dataset_path (str): データセットの保存先パス
        """
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
        ml_assets: MachineLearningAsset,
        outlier_threshold: int | float,
        no_shift_features: list[str],
        save: bool = True,
    ) -> "MLDataset":
        """
        ファクトリ：既存オブジェクトからインスタンスを構築する。
        
        Args:
            dataset_path (str | Path): データセットの保存先パス
            target_df (pd.DataFrame): 目的変数データフレーム
            features_df (pd.DataFrame): 特徴量データフレーム
            raw_returns_df (pd.DataFrame): 生リターンのデータフレーム
            pred_return_df (pd.DataFrame): リターン予測結果のデータフレーム
            order_price_df (pd.DataFrame): 銘柄別の見込み発注価格のデータフレーム
            train_duration (Duration): 学習に使用する期間
            test_duration (Duration): 予測に使用する期間
            date_column (str): 日付列の列名
            sector_column (str): セクター列の列名
            is_model_divided (bool): モデルをセクターごとに分割するかどうか
            ml_assets (MachineLearningAsset): モデルとスケーラーを格納したオブジェクト
            outlier_threshold (int | float): 外れ値除去の閾値。0のときは除外なし
            no_shift_features (list[str]): 1日シフトの対象外とする特徴量を指定
            save (bool): 生成したクラスを保存するか（デフォルト：保存する）
        """
        ds = cls(dataset_path=Path(dataset_path),
                 target_df=target_df,
                 features_df=features_df,
                 raw_returns_df=raw_returns_df,
                 pred_result_df=pred_return_df,
                 order_price_df=order_price_df,
                 train_duration=train_duration,
                 test_duration=test_duration,
                 date_column=date_column,
                 sector_column=sector_column,
                 is_model_divided=is_model_divided,
                 outlier_threshold=outlier_threshold,
                 no_shift_features=no_shift_features,
                 ml_assets=ml_assets,
                 )
        # 必要に応じて永続化
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
        """
        新しい日次データを追加する。

        Args:
            new_targets (pd.DataFrame):アップデート後の目的変数データフレーム
            new_features (pd.DataFrame):アップデート後の特徴量データフレーム
            new_raw_returns (pd.DataFrame):アップデート後の生リターンデータフレーム
            new_order_price (pd.DataFrame):アップデート後の銘柄別発注価格データフレーム
            save (bool): 生成したクラスを保存するか（デフォルト：保存する）
        """
        self.target_df = new_targets.sort_index()
        self.features_df = new_features.sort_index()
        self.order_price_df = new_order_price.sort_index()
        self.raw_returns_df = new_raw_returns.sort_index()
        
        if save:
            self.save()

    def train(self, trainer: BaseTrainer, save: bool = True, **kwargs):
        """
        学習期間のデータを用いてモデルを学習する。
        学習済みモデルやスケーラーは ``ml_assets`` に格納され、指定パスに保存される。

        Args:
            trainer (BaseTrainer): 任意のトレーナークラス（fit/predict を実装しているもの）
            save (bool): 生成したクラスを保存するか（デフォルト：保存する）
        """

        print("学習を開始します...")

        # ---- 事前準備 --------------------------------------------------------------------
        # 日付とセクターをインデックスに指定
        index_cols = [self.date_column]
        if self.sector_column:
            index_cols.append(self.sector_column)
        ttd = self._split_train_test()
        target_train = ttd.target_train_df.reset_index(drop=False).set_index(index_cols, drop=True)
        features_train = ttd.features_train_df.reset_index(drop=False).set_index(index_cols, drop=True)

        # ---- モデル学習 ------------------------------------------------------------------
        if self.is_model_divided:
            # セクター別モデル
            self.ml_assets = []  # 初期化
            sectors = target_train.index.get_level_values(self.sector_column).unique()

            for sector in sectors:
                print(f"セクター '{sector}' のモデルを学習中...")
                sector_mask = target_train.index.get_level_values(self.sector_column) == sector
                sector_target = target_train[sector_mask].copy()
                sector_features = features_train[sector_mask].copy()
                ml_asset = trainer.train(model_name=sector, target_df=sector_target, features_df=sector_features, **kwargs)
                self.ml_assets.append(ml_asset)
        else:
            # 単一モデル
            print("単一モデルを学習中...")
            ml_asset = trainer.train(model_name='Grobal', target_df=target_train, features_df=features_train, **kwargs)
            self.ml_assets = ml_asset
        if save:
            self.save()
        print("学習が完了しました。")

    def predict(self, save: bool = True):
        """
        テスト期間のデータを用いて予測を行う。
        事前に ``ml_assets`` にモデルが存在する必要がある。
        完了後、結果は ``pred_result_df`` に格納される。

        Args:
            save (bool): 生成したクラスを保存するか（デフォルト：保存する）
        """
        print("予測を開始します...")

        if not self.ml_assets:
            raise ValueError("モデルが学習されていません。またはロードパスが指定されていません。")

        ttd = self._split_train_test()
        index_cols = ttd.target_test_df.index.names
        target_test = ttd.target_test_df.reset_index(drop=False).set_index(index_cols, drop=True)
        features_test = ttd.features_test_df.reset_index(drop=False).set_index(index_cols, drop=True)

        # ---- 予測 ------------------------------------------------------------------------
        if isinstance(self.ml_assets, list):
            # セクター別モデル
            print("複数モデルで予測中...")
            all_predictions_df = []
            print(self.features_df)
            for ml_asset_item in self.ml_assets:
                print(f"セクター '{ml_asset_item.name}' で予測中...")
                sector_mask = target_test.index.get_level_values(self.sector_column) == ml_asset_item.name
                target_sector = target_test[sector_mask].copy()
                features_sector = features_test[sector_mask].copy()
                predictions_sector = ml_asset_item.predict(features_sector)
                predictions_sector = pd.concat([target_sector, predictions_sector], axis=1)
                all_predictions_df.append(predictions_sector)
            self.pred_result_df = pd.concat(all_predictions_df, axis=0).sort_index()
        else:  
            # 単一モデル
            print("単一モデルで予測中...")
            predictions = self.ml_assets.predict(features_test)
            self.pred_result_df = pd.concat([target_test, predictions], axis=1).sort_index()
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
        """
        訓練期間・テスト期間にデータを分割し、外れ値処理等の前処理を行います。

        Returns:
            TrainTestData: target_train, target_test, features_train, features_testを保持するインスタンス
        """
        return TrainTestData().archive(
            target_df=self.target_df, features_df=self.features_df,
            train_duration=self.train_duration, test_duration=self.test_duration,
            datetime_column=self.date_column, outlier_threshold=self.outlier_threshold,
            no_shift_features=self.no_shift_features, reuse_features_df=False)

#=====================================
#以下、セーブ・ロード用のクラス。
#=====================================

class MLDatasetStorage:
    """
    MLDatasetインスタンスのセーブ・ロードを司るクラス
    
    Args:
        base_path (str | Path): データセットのディレクトリパス
    """

    _TARGET_FILE = "target_df.parquet"
    _FEATURE_FILE = "features_df.parquet"
    _RAW_RETURN_FILE = "raw_returns_df.parquet"
    _PRED_RESULT_FILE = "pred_result_df.parquet"
    _ORDER_PRICE_FILE = "order_price_df.parquet"
    _META_FILE = "metadata.json"
    _ML_ASSETS_FILE = "ml_assets.pkl"

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)


    @property
    def path(self) -> Dict[str, Path]:
        return {
            "target_df": self.base_path / self._TARGET_FILE,
            "features_df": self.base_path / self._FEATURE_FILE,
            "raw_returns_df": self.base_path / self._RAW_RETURN_FILE,
            "pred_result_df": self.base_path / self._PRED_RESULT_FILE,
            "order_price_df": self.base_path / self._ORDER_PRICE_FILE,
            "metadata": self.base_path / self._META_FILE,
            "ml_assets": self.base_path / self._ML_ASSETS_FILE,
            }


    def load(self) -> "MLDataset":
        """
        外部ファイルをロードしてMLDatasetを作成します。
        
        Returns:
            MLdataset: 機械学習のデータセットインスタンス
        """
        with self.path["metadata"].open(encoding="utf-8") as f:
            metadata = json.load(f)

        with self.path["ml_assets"].open("rb") as f:
            ml_assets = pickle.load(f)

        return MLDataset(
            dataset_path=self.base_path,
            target_df=pd.read_parquet(self.path["target_df"]),
            features_df=pd.read_parquet(self.path["features_df"]),
            raw_returns_df=pd.read_parquet(self.path["raw_returns_df"]),
            pred_result_df=pd.read_parquet(self.path["pred_result_df"]),
            order_price_df=pd.read_parquet(self.path["order_price_df"]),
            train_duration = Duration(start=pd.to_datetime(metadata["train_start"], unit='ms'), 
                                      end=pd.to_datetime(metadata["train_end"], unit='ms')),
            test_duration =  Duration(start=pd.to_datetime(metadata["test_start"], unit='ms'), 
                                      end=pd.to_datetime(metadata["test_end"], unit='ms')),
            date_column = metadata["date_column"],
            sector_column = metadata["sector_column"],
            is_model_divided = metadata["is_model_divided"],
            outlier_threshold = metadata["outlier_threshold"],
            no_shift_features = metadata["no_shift_features"],
            ml_assets = ml_assets,
        )

    def save(self, ds: "MLDataset") -> None:
        """
        MLDatasetのプロパティを外部ファイルに出力します。
        
        Args:
            ds (MLDataset): 機械学習のデータセットインスタンス
        """
        self._atomic_write_parquet(ds.target_df, self.path["target_df"])
        self._atomic_write_parquet(ds.features_df, self.path["features_df"])
        self._atomic_write_parquet(ds.raw_returns_df, self.path["raw_returns_df"])
        self._atomic_write_parquet(ds.pred_result_df, self.path["pred_result_df"])
        self._atomic_write_parquet(ds.order_price_df, self.path["order_price_df"])

        meta = {
            "train_start": ds.train_duration.start,
            "train_end": ds.train_duration.end,
            "test_start": ds.test_duration.start,
            "test_end": ds.test_duration.end,
            "date_column": ds.date_column,
            "sector_column": ds.sector_column,
            "is_model_divided": ds.is_model_divided,
            "outlier_threshold": ds.outlier_threshold,
            "no_shift_features": ds.no_shift_features,
        }
        pd.Series(meta).to_json(self.path["metadata"], indent=2, force_ascii=False)

        if ds.ml_assets is not None:
            # 一時ファイルとして保存することで、中断したときのファイル破損を防ぐ。
            tmp_path = self.path["ml_assets"].with_suffix(self.path["ml_assets"].suffix + ".tmp")
            with tmp_path.open("wb") as f:
                pickle.dump(ds.ml_assets, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(self.path["ml_assets"])

    # ------------------------------------------------------------------
    # ヘルパーメソッド
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write_parquet(obj: pd.DataFrame, dest: Path, compression: str = "zstd") -> None:
        """
        一時ファイルとして保存し、保存中断した場合のファイル破損を防ぎます。
        """
        if obj is not None:
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            obj.to_parquet(tmp_path, compression=compression)
            tmp_path.replace(dest)