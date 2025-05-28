import pandas as pd
from typing import List, Tuple
from models.machine_learning.ensembles.base_ensemble_method import BaseEnsembleMethod

class ByVotingMethod(BaseEnsembleMethod):
    """投票ベースでのアンサンブル手法"""
    
    def __init__(self, top_n: int = 5):
        """
        Args:
            top_n (int): 各モデルの上位何位までを考慮するか
        """
        self.top_n = top_n
    
    def ensemble(self, inputs: List[Tuple[pd.DataFrame, float]]) -> pd.DataFrame:
        """
        各モデルの上位予測に投票を行いアンサンブルを実行
        
        Args:
            inputs (List[Tuple[pd.DataFrame, float]]): (予測結果データフレーム, 重み)のタプルのリスト
            
        Returns:
            pd.DataFrame: アンサンブル後の投票スコアを格納したデータフレーム
        """
        self._validate_inputs(inputs)
        normalized_inputs = self._normalize_weights(inputs)
        
        vote_scores = pd.Series(0.0, index=normalized_inputs[0][0].index)
        
        for pred_result_df, weight in normalized_inputs:
            # 各日付での順位を計算
            ranks = pred_result_df.groupby('Date')['Pred'].rank(ascending=False)
            
            # 上位top_nに投票（順位が高いほど高いスコア）
            top_mask = ranks <= self.top_n
            vote_values = (self.top_n + 1 - ranks) * weight
            vote_scores += vote_values * top_mask

        ensembled_vote_df = pd.DataFrame(vote_scores, columns=['Pred'])
        return ensembled_vote_df