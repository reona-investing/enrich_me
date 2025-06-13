from typing import Optional, Dict
import pandas as pd
from calculation.features.implementation import IndexFeatures, PriceFeatures

class FeaturesSet:
    """特徴量データの結合管理に特化したクラス"""
    
    def __init__(self):
        """初期化時はインスタンス生成のみ"""
        pass
    
    def combine_features(self, 
                        index_calculator: Optional[IndexFeatures] = None,
                        price_calculator: Optional[PriceFeatures] = None) -> Optional[pd.DataFrame]:
        """
        計算済み特徴量を持つ計算器インスタンスから特徴量を結合
        
        Args:
            index_calculator: インデックス系特徴量計算器（計算済み）
            price_calculator: 価格系特徴量計算器（計算済み）
            
        Returns:
            結合された特徴量データフレーム
        """
        valid_features = []
        
        if index_calculator is not None and index_calculator.has_features():
            valid_features.append(index_calculator.get_features())
        
        if price_calculator is not None and price_calculator.has_features():
            valid_features.append(price_calculator.get_features())
        
        if len(valid_features) == 0:
            return None
        elif len(valid_features) == 1:
            result = valid_features[0].copy()
        else:
            result = self._merge_features(valid_features[0], valid_features[1])
        
        return result.sort_index()
    
    def combine_from_calculators(self,
                               index_calculator: Optional[IndexFeatures] = None,
                               price_calculator: Optional[PriceFeatures] = None,
                               # 共通パラメータ
                               new_sector_price: Optional[pd.DataFrame] = None,
                               new_sector_list: Optional[pd.DataFrame] = None,
                               stock_dfs_dict: Optional[Dict] = None,
                               # インデックス系パラメータ
                               index_params: Optional[Dict] = None,
                               # 価格系パラメータ
                               price_params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        計算器インスタンスから特徴量を計算して結合
        
        Args:
            index_calculator: インデックス特徴量計算器
            price_calculator: 価格特徴量計算器
            new_sector_price: セクター価格データ（価格系計算時に必要）
            new_sector_list: セクターリストデータ（価格系計算時に必要）
            stock_dfs_dict: 株価データ辞書（価格系計算時に必要）
            index_params: インデックス系特徴量計算パラメータ
            price_params: 価格系特徴量計算パラメータ
            
        Returns:
            結合された特徴量データフレーム
        """
        # デフォルトパラメータの設定
        if index_params is None:
            index_params = {}
        if price_params is None:
            price_params = {}
        
        # インデックス系特徴量の計算
        if index_calculator is not None:
            index_calculator.calculate_features(**index_params)
        
        # 価格系特徴量の計算
        if price_calculator is not None:
            if all(data is not None for data in [new_sector_price, new_sector_list, stock_dfs_dict]):
                price_calculator.calculate_features(
                    new_sector_price=new_sector_price,
                    new_sector_list=new_sector_list,
                    stock_dfs_dict=stock_dfs_dict,
                    **price_params
                )
        
        # 特徴量の結合
        return self.combine_features(index_calculator, price_calculator)
    
    def _merge_features(self, 
                       indices_features: pd.DataFrame, 
                       price_features: pd.DataFrame) -> pd.DataFrame:
        """
        インデックス系と価格系特徴量の結合
        
        Args:
            indices_features: インデックス系特徴量データフレーム
            price_features: 価格系特徴量データフレーム
            
        Returns:
            結合された特徴量データフレーム
        """
        # インデックスをリセットしてマージ
        merged_features = pd.merge(
            indices_features.reset_index(), 
            price_features.reset_index(), 
            on=['Date'], 
            how='outer'
        )
        
        # Sector列がない行を除去
        merged_features = merged_features.dropna(subset=['Sector'])
        
        # Sector列の復元とffill処理
        sector_values = merged_features['Sector'].values
        merged_features = merged_features.groupby('Sector').ffill()
        merged_features['Sector'] = sector_values
        
        # インデックスを再設定
        return merged_features.set_index(['Date', 'Sector'], drop=True)


# 使用例とパターン
if __name__ == '__main__':
    from acquisition.jquants_api_operations import StockAcquisitionFacade
    from calculation.sector_index.sector_index import SectorIndex
    from utils.paths import Paths

    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv' #別でファイルを作っておく
    SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet' #出力のみなのでファイルがなくてもOK

    saf = StockAcquisitionFacade(filter="(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))")
    stock_dfs = saf.get_stock_data_dict()
    sic = SectorIndex()
    sector_df, _ = sic.calc_sector_index(stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)
    sector_list_df = pd.read_csv(SECTOR_REDEFINITIONS_CSV)

    i_features = IndexFeatures()
    p_features = PriceFeatures()
    i_features.calculate_features()
    p_features.calculate_features(sector_df, sector_list_df, stock_dfs)
    fs = FeaturesSet()
    df = fs.combine_features(i_features, p_features)
    print(df)