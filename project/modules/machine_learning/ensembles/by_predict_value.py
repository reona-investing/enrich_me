import pandas as pd
from typing import List, Tuple
from machine_learning.ensembles.base_ensemble_method import BaseEnsembleMethod

class ByPredictValueMethod(BaseEnsembleMethod):
    """予測値ベースでのアンサンブル手法"""
    
    def ensemble(self, inputs: List[Tuple[pd.DataFrame, float]]) -> pd.DataFrame:
        """
        予測値の重み付き平均でアンサンブルを実行
        
        Args:
            inputs (List[Tuple[pd.DataFrame, float]]): (予測結果データフレーム, 重み)のタプルのリスト
            
        Returns:
            pd.DataFrame: アンサンブル後の予測値を格納したデータフレーム
        """
        self._validate_inputs(inputs)
        normalized_inputs = self._normalize_weights(inputs)
        
        ensembled_pred = None
        for pred_result_df, weight in normalized_inputs:
            weighted_pred = pred_result_df['Pred'] * weight
            ensembled_pred = weighted_pred if ensembled_pred is None else ensembled_pred + weighted_pred

        ensembled_pred_df = pd.DataFrame(ensembled_pred, index=normalized_inputs[0][0].index, columns=['Pred'])
        return ensembled_pred_df