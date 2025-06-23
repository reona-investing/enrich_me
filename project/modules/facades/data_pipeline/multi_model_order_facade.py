from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from models.machine_learning.loaders import DatasetLoader
from trading import TradingFacade
from models.machine_learning.models import LassoModel
from models.machine_learning.ml_dataset import MLDatasets, SingleMLDataset

@dataclass
class ModelOrderConfig:
    dataset_path: str
    sector_csv: str
    trading_sector_num: int
    candidate_sector_num: int
    top_slope: float
    margin_weight: float = 0.5

class MultiModelOrderExecutionFacade:
    """複数モデルの予測結果を用いた発注を管理するファサード"""

    def __init__(self, mode: Literal['new', 'none'], trade_facade: TradingFacade, configs: List[ModelOrderConfig]):
        self.mode = mode
        self.trade_facade = trade_facade
        self.configs = configs

    def _predict_lasso(self, dataset_path: str) -> MLDatasets:
        """LASSOモデルで予測を実行し ``MLDatasets`` を返す"""
        loader = DatasetLoader(dataset_path)
        ml_datasets = loader.load_datasets()
        model = LassoModel()
        for _, single_ml in ml_datasets.items():
            pred_df = model.predict(
                single_ml.train_test_materials.target_test_df,
                single_ml.train_test_materials.features_test_df,
                single_ml.ml_object_materials.model,
                single_ml.ml_object_materials.scaler,
            )
            single_ml.archive_pred_result(pred_df)
            single_ml.save()
            ml_datasets.replace_model(single_ml_dataset=single_ml)
        return ml_datasets

    async def execute(self) -> None:
        if self.mode == 'none':
            return
        if self.mode != 'new':
            raise NotImplementedError("現在のところ 'new' モードのみ対応しています。")

        await self.trade_facade.margin_provider.refresh()
        total_margin = await self.trade_facade.margin_provider.get_available_margin()

        for cfg in self.configs:
            ml_datasets = self._predict_lasso(cfg.dataset_path)
            alloc_margin = total_margin * cfg.margin_weight
            await self.trade_facade.take_positions(
                order_price_df=ml_datasets.get_order_price(),
                pred_result_df=ml_datasets.get_pred_result(),
                SECTOR_REDEFINITIONS_CSV=cfg.sector_csv,
                num_sectors_to_trade=cfg.trading_sector_num,
                num_candidate_sectors=cfg.candidate_sector_num,
                top_slope=cfg.top_slope,
                margin_power=alloc_margin,
            )

