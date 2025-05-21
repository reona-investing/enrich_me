import pandas as pd
from typing import Tuple
from trading.sbi.interface.selection import IPriceProvider

class ETFAllocator:
    """ETF配分クラス"""
    
    def __init__(self, price_provider: IPriceProvider):
        """
        Args:
            price_provider: 価格情報プロバイダー
        """
        self.price_provider = price_provider
    
    def allocate_etfs(self, 
                     long_df: pd.DataFrame, 
                     short_df: pd.DataFrame, 
                     buy_adjuster_num: int, 
                     sell_adjuster_num: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ETFを配分"""
        long_df, short_df = long_df.copy(), short_df.copy()
        
        if buy_adjuster_num > 0:
            # TOPIX ETFを追加
            for i in range(buy_adjuster_num):
                etf_price = self.price_provider.get_etf_price('1308')
                etf_row = pd.DataFrame({
                    'Code': ['1308'],
                    'CompanyName': ['上場インデックスファンドTOPIX'],
                    'Sector': [f'ETF{i}'],
                    'Weight': [1.0],
                    'EstimatedCost': [etf_price * 10]
                })
                long_df = pd.concat([long_df, etf_row], ignore_index=True)
        
        if sell_adjuster_num > 0:
            # TOPIX ベア2倍 ETFを追加
            for i in range(sell_adjuster_num):
                etf_price = self.price_provider.get_etf_price('1356')
                etf_row = pd.DataFrame({
                    'Code': ['1356'],
                    'CompanyName': ['TOPIXベア2倍上場投信'],
                    'Sector': [f'ETF{i}'],
                    'Weight': [0.5],  # ベア2倍のため半分
                    'EstimatedCost': [etf_price * 10]
                })
                short_df = pd.concat([short_df, etf_row], ignore_index=True)
        
        # 欠損値を削除
        long_df = long_df.dropna(subset=['Weight', 'EstimatedCost'])
        short_df = short_df.dropna(subset=['Weight', 'EstimatedCost'])
        
        return long_df, short_df