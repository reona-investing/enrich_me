import pandas as pd
from trading.sbi.selection.interface import ISectorProvider, ITradeLimitProvider

class TradabilityAnalyzer:
    """取引可能性分析クラス"""
    
    def __init__(self, sector_provider: ISectorProvider, trade_limit_provider: ITradeLimitProvider):
        """
        Args:
            sector_provider: セクター情報プロバイダー
            trade_limit_provider: 取引制限情報プロバイダー
        """
        self.sector_provider = sector_provider
        self.trade_limit_provider = trade_limit_provider
    
    async def analyze_sector_tradability(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """セクターごとの取引可能性を分析"""
        print("[INFO] TradabilityAnalyzer.analyze_sector_tradability を開始します")
        
        # セクター定義を取得
        sector_df = self.sector_provider.get_sector_definitions()
        
        # 買い・売り可能銘柄を取得 - このメソッド呼び出しではキャッシュが使われるようにする
        print("[INFO] TradabilityAnalyzer: 買い・売り可能銘柄を取得します")
        buyable_df = await self.trade_limit_provider.get_buyable_symbols()
        sellable_df = await self.trade_limit_provider.get_sellable_symbols()
        
        # セクターごとの取引可能銘柄数を集計
        buy_counts = self._count_tradable_by_sector(sector_df, buyable_df)
        sell_counts = self._count_tradable_by_sector(sector_df, sellable_df)
        total_counts = self._count_total_by_sector(sector_df)
        
        # 予測結果と結合
        result_df = pred_df.copy()
        result_df = pd.merge(result_df, buy_counts, how='left', on='Sector')
        result_df = pd.merge(result_df, sell_counts, how='left', on='Sector')
        result_df = pd.merge(result_df, total_counts, how='left', on='Sector')
        
        # 欠損値を0で埋める
        result_df[['Buy', 'Sell']] = result_df[['Buy', 'Sell']].fillna(0).astype(int)
        
        # 取引可能フラグを設定（セクター内の50%以上の銘柄が取引可能な場合）
        result_df['Buyable'] = (result_df['Buy'] / result_df['Total'] >= 0.5).astype(int)
        result_df['Sellable'] = (result_df['Sell'] / result_df['Total'] >= 0.5).astype(int)
        
        return result_df
    
    def _count_tradable_by_sector(self, sector_df: pd.DataFrame, tradable_df: pd.DataFrame) -> pd.DataFrame:
        """セクターごとの取引可能銘柄数を集計"""
        # 取引可能な銘柄をセクター情報と結合
        tradable_with_sector = pd.merge(
            tradable_df, 
            sector_df[['Code', 'Sector']], 
            how='inner', 
            on='Code'
        )
        
        # セクターごとに集計
        counts = tradable_with_sector.groupby('Sector')['Code'].count().reset_index()
        
        # カラム名を変更
        column_name = 'Buy' if 'MaxUnit' in tradable_df.columns and 'isBorrowingStock' not in tradable_df.columns else 'Sell'
        counts = counts.rename(columns={'Code': column_name})
        
        return counts
    
    def _count_total_by_sector(self, sector_df: pd.DataFrame) -> pd.DataFrame:
        """セクターごとの合計銘柄数を集計"""
        total_counts = sector_df.groupby('Sector')['Code'].count().reset_index()
        total_counts = total_counts.rename(columns={'Code': 'Total'})
        return total_counts