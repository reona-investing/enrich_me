import pandas as pd
from typing import List, Tuple

from machine_learning.collection import ModelCollection

class Ensemble:

    @staticmethod
    def ensemble_by_rank(collections: List[Tuple[ModelCollection, float]]) -> pd.DataFrame:
        """
        複数のModelCollectionの予測結果を予測順位ベースでアンサンブルする。
        
        Args:
            collections: (モデルコレクション, 重み)のタプルを格納したリスト
            
        Returns:
            pd.DataFrame: アンサンブル後の予測順位を格納したデータフレーム
            
        Raises:
            ValueError: collections が空の場合、または予測結果が存在しないコレクションがある場合
        """
        if not collections:
            raise ValueError("少なくとも1つのコレクションを指定してください。")
        
        # 各コレクションの予測結果を取得
        ensembled_rank = None
        
        for collection, weight in collections:
            # 予測結果を取得
            try:
                pred_result_df = collection.get_result_df()
            except ValueError as e:
                raise ValueError(f"コレクションの予測結果を取得できません: {e}")
            
            # 日付ごとに順位付け
            rank = pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * weight
            
            # 集計
            ensembled_rank = rank if ensembled_rank is None else ensembled_rank + rank
        
        # 結果をデータフレームに変換
        first_collection = collections[0][0]
        first_pred_df = first_collection.get_result_df()
        
        ensembled_rank_df = pd.DataFrame(ensembled_rank, index=first_pred_df.index, columns=['Pred'])
        
        # 日付ごとに順位付け
        return ensembled_rank_df.groupby('Date')[['Pred']].rank(ascending=False)