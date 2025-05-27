import pandas as pd
from typing import List
from trading.sbi.selection.interface import IWeightCalculator, StockWeight

class WeightCalculator(IWeightCalculator):
    """ウェイト計算クラス"""
    
    def calculate_weights(self, sector_definitions: pd.DataFrame, price_data: pd.DataFrame) -> List[StockWeight]:
        """銘柄ウェイトを計算"""
        # 1. 過去5日間の出来高平均と株価を計算
        price_data = self._calc_index_weight_components(price_data)
        
        # 2. セクター情報と結合
        merged_df = self._merge_sector_and_price_data(sector_definitions, price_data)
        
        # 3. ウェイト計算
        weight_df = self._calculate_sector_weights(merged_df)
        
        # 4. StockWeightオブジェクトリストに変換
        return [
            StockWeight(
                Code=row['Code'], 
                Sector=row['Sector'],
                CompanyName=row['CompanyName'],
                Weight=row['Weight'],
                EstimatedCost=row['EstimatedCost']
            )
            for _, row in weight_df.iterrows()
        ]
    
    def _calc_index_weight_components(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """終値時点での単位株購入コスト、過去5営業日の出来高平均、時価総額を算出"""
        price_df = price_df.copy()
        price_df['Past5daysVolume'] = price_df.groupby('Code')['Volume'].rolling(5).mean().reset_index(level=0, drop=True)
        price_df['EstimatedCost'] = price_df['Close'] * 100
        latest_date = price_df['Date'].max()
        return price_df.loc[price_df['Date'] == latest_date, ['Code', 'EstimatedCost', 'Past5daysVolume', 'MarketCapClose']]

    def _merge_sector_and_price_data(self, sector_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """セクター情報と価格データを結合"""
        sector_df = sector_df.copy()
        sector_df['Code'] = sector_df['Code'].astype(str)
        merged_df = pd.merge(
            sector_df[['Code', 'CompanyName', 'Sector']], 
            price_df, 
            how='right', 
            on='Code'
        )
        return merged_df[merged_df['Sector'].notnull()]
    
    def _calculate_sector_weights(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """セクター内での銘柄ウェイトを計算"""
        # セクターごとの時価総額合計を計算
        sector_totals = merged_df.groupby('Sector')['MarketCapClose'].sum()
        
        # 各銘柄のウェイトを計算
        merged_df['Weight'] = merged_df.apply(
            lambda row: row['MarketCapClose'] / sector_totals[row['Sector']] 
            if row['Sector'] in sector_totals.index else 0, 
            axis=1
        )
        
        return merged_df