import pandas as pd
from typing import List, Tuple
from machine_learning.ensembles.base_ensemble_method import BaseEnsembleMethod

class ByRankMethod(BaseEnsembleMethod):
    """予測順位ベースでのアンサンブル手法"""
    
    def ensemble(self, inputs: List[Tuple[pd.DataFrame, float]]) -> pd.DataFrame:
        """
        予測順位ベースでアンサンブルを実行
        
        Args:
            inputs (List[Tuple[pd.DataFrame, float]]): (予測結果データフレーム, 重み)のタプルのリスト
            
        Returns:
            pd.DataFrame: アンサンブル後の予測順位を格納したデータフレーム
        """
        self._validate_inputs(inputs)
        normalized_inputs = self._normalize_weights(inputs)
        
        ensembled_rank = None
        for pred_result_df, weight in normalized_inputs:
            # 各データフレームの予測値を順位に変換し、重み付け
            rank = pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * weight
            ensembled_rank = rank if ensembled_rank is None else ensembled_rank + rank

        # 重み付けされた順位を基に最終的な順位を計算
        ensembled_rank_df = pd.DataFrame(ensembled_rank, index=normalized_inputs[0][0].index, columns=['Pred'])
        return ensembled_rank_df.groupby('Date')[['Pred']].rank(ascending=False)