import pandas as pd
from typing import List, Tuple
from trading.sbi.selection.interface import ISectorAnalyzer, SectorPrediction

class SectorAnalyzer(ISectorAnalyzer):
    """セクター分析クラス"""
    
    def analyze_sector_predictions(self, predictions_df: pd.DataFrame) -> List[SectorPrediction]:
        """セクター予測を分析"""
        # ランク付け
        predictions_df = predictions_df.copy()
        predictions_df['Rank'] = predictions_df['Pred'].rank(ascending=False).astype(int)
        
        # SectorPredictionオブジェクトのリストに変換
        sector_predictions = []
        for _, row in predictions_df.iterrows():
            sector_predictions.append(
                SectorPrediction(
                    Sector=row['Sector'],
                    PredictionValue=row['Pred'],
                    Rank=row['Rank'],
                    isBuyable=bool(row.get('Buyable', 0)),
                    isSellable=bool(row.get('Sellable', 0))
                )
            )
        
        return sector_predictions
    
    def select_sectors_to_trade(self, 
                              sector_predictions: List[SectorPrediction], 
                              num_sectors: int, 
                              num_candidates: int) -> Tuple[List[str], List[str]]:
        """取引対象セクターを選択"""
        # 予測値でソート
        sorted_predictions = sorted(sector_predictions, key=lambda x: x.PredictionValue)
        
        # 売り候補（下位）
        sell_candidates = sorted_predictions[:num_candidates]
        sell_sectors = [sp.Sector for sp in sell_candidates if sp.isSellable][:num_sectors]
        
        # 買い候補（上位）
        buy_candidates = sorted_predictions[-num_candidates:]
        buy_candidates.reverse()  # 予測値が高い順に
        buy_sectors = [sp.Sector for sp in buy_candidates if sp.isBuyable][:num_sectors]
        
        return buy_sectors, sell_sectors