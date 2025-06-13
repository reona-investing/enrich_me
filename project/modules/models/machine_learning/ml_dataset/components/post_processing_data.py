import pandas as pd

from models.machine_learning.ml_dataset.components.base_data_component import BaseDataComponent
from models.machine_learning.outputs import EvaluationMaterials, StockSelectionMaterials

class PostProcessingData(BaseDataComponent):
    """後処理データの管理"""
    
    instance_vars = {
        'raw_target_df': '.parquet',
        'order_price_df': '.parquet',
        'pred_result_df': '.parquet',
    }

    def archive_raw_target(self, raw_target_df: pd.DataFrame):
        """生の目的変数を格納"""
        self._raw_target_df = raw_target_df

    def archive_order_price(self, order_price_df: pd.DataFrame):
        """個別銘柄の発注価格を格納"""
        self._order_price_df = order_price_df

    def archive_pred_result(self, pred_result_df: pd.DataFrame):
        """予測結果を格納"""
        self._pred_result_df = pred_result_df

    def getter_stock_selection(self) -> StockSelectionMaterials:
        """株式選択用データを返却"""
        return StockSelectionMaterials(
            order_price_df=self._order_price_df,
            pred_result_df=self._pred_result_df
        )

    def getter_evaluation(self) -> EvaluationMaterials:
        """評価用データを返却"""
        return EvaluationMaterials(
            pred_result_df=self._pred_result_df,
            raw_target_df=self._raw_target_df
        )

    def getter(self) -> EvaluationMaterials:
        """デフォルトのgetter（評価用データを返却）"""
        return self.getter_evaluation()