from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Tuple

class BaseEnsembleMethod(ABC):
    """アンサンブル手法の抽象基底クラス"""
    
    @abstractmethod
    def ensemble(self, inputs: List[Tuple[pd.DataFrame, float]]) -> pd.DataFrame:
        """
        アンサンブルを実行する抽象メソッド
        
        Args:
            inputs (List[Tuple[pd.DataFrame, float]]): (予測結果データフレーム, 重み)のタプルのリスト
            
        Returns:
            pd.DataFrame: アンサンブル後の予測結果を格納したデータフレーム
        """
        pass
    
    def _validate_inputs(self, inputs: List[Tuple[pd.DataFrame, float]]) -> None:
        """入力の妥当性をチェック"""
        assert len(inputs) > 0, 'inputsには1つ以上の要素を指定してください。'
        
        # 全てのデータフレームが同じインデックスを持つかチェック
        base_index = inputs[0][0].index
        for pred_df, _ in inputs[1:]:
            assert pred_df.index.equals(base_index), 'すべての予測データフレームは同じインデックスを持つ必要があります。'
        
        # 重みが正の値であることをチェック
        total_weight = sum(weight for _, weight in inputs)
        assert total_weight > 0, '重みの合計は0より大きい必要があります。'
    
    def _normalize_weights(self, inputs: List[Tuple[pd.DataFrame, float]]) -> List[Tuple[pd.DataFrame, float]]:
        """重みを正規化（合計を1.0にする）"""
        total_weight = sum(weight for _, weight in inputs)
        
        normalized_inputs = [
            (pred_df, weight / total_weight) 
            for pred_df, weight in inputs
        ]
        
        return normalized_inputs