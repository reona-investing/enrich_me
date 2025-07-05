from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, List, Dict
import json
import pickle
import pandas as pd

from machine_learning.ml_dataset.components import MLModelAsset
from machine_learning.models import BaseTrainer


@dataclass
class MLAssetsMetadata:
    """MLModelAssetの管理方針に関する設定"""

    is_model_divided: bool


@dataclass
class MLModelAssetCollection:
    """MLModelAssetの管理を担当するクラス"""

    metadata: MLAssetsMetadata = field(repr=False)
    assets: Union[MLModelAsset, List[MLModelAsset]] = field(default_factory=list)

    def train_models(
        self,
        trainer: BaseTrainer,
        target_train: pd.DataFrame,
        features_train: pd.DataFrame,
        sector_column: str,
        **kwargs,
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
        sector_column: str,
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


class MLModelAssetCollectionStorage:
    """MLModelAssetCollectionインスタンスのセーブ・ロードを司るクラス"""

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

    def load(self) -> MLModelAssetCollection:
        """外部ファイルをロードしてMLModelAssetCollectionを作成します。旧形式のmetadataにも対応します。"""
        with self.path["ml_assets"].open("rb") as f:
            assets = pickle.load(f)

        metadata_path = self.path["metadata"]
        if not metadata_path.exists() and self.path["metadata_old"].exists():
            metadata_path = self.path["metadata_old"]

        with metadata_path.open(encoding="utf-8") as f:
            metadata_dict = json.load(f)

        metadata = MLAssetsMetadata(**metadata_dict)

        return MLModelAssetCollection(
            assets=assets,
            metadata=metadata,
        )

    def save(self, ml_assets_container: MLModelAssetCollection) -> None:
        """MLModelAssetCollectionを外部ファイルに出力します。"""
        if ml_assets_container.assets is not None:
            tmp_path = self.path["ml_assets"].with_suffix(self.path["ml_assets"].suffix + ".tmp")
            with tmp_path.open("wb") as f:
                pickle.dump(ml_assets_container.assets, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(self.path["ml_assets"])

        metadata_dict = {
            "is_model_divided": ml_assets_container.metadata.is_model_divided,
        }
        pd.Series(metadata_dict).to_json(self.path["metadata"], indent=2, force_ascii=False)