from typing import Optional, Dict, List, Literal
import pandas as pd
from acquisition.jquants_api_operations import StockAcquisitionFacade
from calculation.sector_index_calculator import SectorIndex
from calculation.features.implementation import IndexFeatures, PriceFeatures
from calculation.features.integration.features_set import FeaturesSet
from preprocessing import PreprocessingPipeline


class FeaturesFacade:
    """特徴量データフレーム作成のファサードクラス"""
    
    def __init__(self):
        """初期化"""
        pass
    
    def create_features_dataframe(self,
                                # データ取得・セクター計算パラメータ
                                stock_filter: str = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))",
                                sector_redefinitions_csv: str = None,
                                sector_index_parquet: str = None,
                                
                                # 特徴量種別選択
                                use_index_features: bool = True,
                                use_price_features: bool = True,
                                
                                # インデックス系特徴量パラメータ
                                groups_setting: Dict = {},
                                names_setting: Dict = {},
                                currencies_type: Literal['relative', 'raw'] = 'relative',
                                commodity_type: Literal['JPY', 'raw'] = 'raw',
                                
                                # 価格系特徴量パラメータ
                                adopt_1d_return: bool = True,
                                mom_duration: List[int] = [5, 21],
                                vola_duration: List[int] = [5, 21],
                                adopt_size_factor: bool = True,
                                adopt_eps_factor: bool = True,
                                adopt_sector_categorical: bool = True,
                                add_rank: bool = True,
                                
                                # 前処理パラメータ
                                index_preprocessing: Optional[PreprocessingPipeline] = None,
                                price_preprocessing: Optional[PreprocessingPipeline] = None) -> pd.DataFrame:
        """
        特徴量データフレームを作成
        
        Args:
            stock_filter: 株式データフィルタ条件
            sector_redefinitions_csv: セクター再定義CSVファイルパス
            sector_index_parquet: セクターインデックス出力パーケットファイルパス
            use_index_features: インデックス系特徴量を使用するか
            use_price_features: 価格系特徴量を使用するか
            groups_setting: インデックス特徴量グループ設定
            names_setting: インデックス特徴量名称設定
            currencies_type: 通貨処理タイプ
            commodity_type: コモディティ処理タイプ
            adopt_1d_return: 1日リターンを採用するか
            mom_duration: モメンタム計算期間
            vola_duration: ボラティリティ計算期間
            adopt_size_factor: サイズファクターを採用するか
            adopt_eps_factor: EPSファクターを採用するか
            adopt_sector_categorical: セクターカテゴリを採用するか
            add_rank: ランキングを追加するか
            index_preprocessing: インデックス系前処理パイプライン
            price_preprocessing: 価格系前処理パイプライン
            
        Returns:
            結合された特徴量データフレーム
            
        Raises:
            ValueError: 必要なパラメータが不足している場合
        """
        # 基本データの取得
        stock_dfs_dict = self._get_stock_data(stock_filter)
        
        # セクターデータの準備（価格系特徴量が必要な場合のみ）
        sector_df = None
        sector_list_df = None
        
        if use_price_features:
            if sector_redefinitions_csv is None:
                raise ValueError("価格系特徴量を使用する場合、sector_redefinitions_csvは必須です")
            
            sector_df, sector_list_df = self._prepare_sector_data(
                stock_dfs_dict, sector_redefinitions_csv, sector_index_parquet
            )
        
        # 特徴量計算器の準備
        index_calculator = None
        price_calculator = None
        
        if use_index_features:
            index_calculator = self._create_index_features(
                groups_setting, names_setting, currencies_type, commodity_type, index_preprocessing
            )
        
        if use_price_features:
            price_calculator = self._create_price_features(
                sector_df, sector_list_df, stock_dfs_dict,
                adopt_1d_return, mom_duration, vola_duration,
                adopt_size_factor, adopt_eps_factor, adopt_sector_categorical,
                add_rank, price_preprocessing
            )
        
        # 特徴量の結合
        features_set = FeaturesSet()
        result = features_set.combine_features(index_calculator, price_calculator)
        
        if result is None:
            raise ValueError("特徴量データフレームの作成に失敗しました")
        
        print("特徴量データフレームの作成が完了しました。")
        return result
    
    def _get_stock_data(self, stock_filter: str) -> Dict:
        """株式データの取得"""
        print("株式データを取得中...")
        saf = StockAcquisitionFacade(filter=stock_filter)
        return saf.get_stock_data_dict()
    
    def _prepare_sector_data(self, 
                           stock_dfs_dict: Dict,
                           sector_redefinitions_csv: str,
                           sector_index_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """セクターデータの準備"""
        print("セクターデータを準備中...")
        sic = SectorIndex()
        sector_df, _ = sic.calc_sector_index(
            stock_dfs_dict, sector_redefinitions_csv, sector_index_parquet
        )
        sector_list_df = pd.read_csv(sector_redefinitions_csv)
        return sector_df, sector_list_df
    
    def _create_index_features(self,
                             groups_setting: Dict,
                             names_setting: Dict,
                             currencies_type: str,
                             commodity_type: str,
                             preprocessing: Optional[PreprocessingPipeline]) -> IndexFeatures:
        """インデックス系特徴量の作成"""
        print("インデックス系特徴量を計算中...")
        index_calculator = IndexFeatures()
        index_calculator.calculate_features(
            groups_setting=groups_setting,
            names_setting=names_setting,
            currencies_type=currencies_type,
            commodity_type=commodity_type
        )
        
        if preprocessing is not None:
            index_calculator.apply_preprocessing(preprocessing)
        
        return index_calculator
    
    def _create_price_features(self,
                             sector_df: pd.DataFrame,
                             sector_list_df: pd.DataFrame,
                             stock_dfs_dict: Dict,
                             adopt_1d_return: bool,
                             mom_duration: List[int],
                             vola_duration: List[int],
                             adopt_size_factor: bool,
                             adopt_eps_factor: bool,
                             adopt_sector_categorical: bool,
                             add_rank: bool,
                             preprocessing: Optional[PreprocessingPipeline]) -> PriceFeatures:
        """価格系特徴量の作成"""
        print("価格系特徴量を計算中...")
        price_calculator = PriceFeatures()
        price_calculator.calculate_features(
            new_sector_price=sector_df,
            new_sector_list=sector_list_df,
            stock_dfs_dict=stock_dfs_dict,
            adopt_1d_return=adopt_1d_return,
            mom_duration=mom_duration,
            vola_duration=vola_duration,
            adopt_size_factor=adopt_size_factor,
            adopt_eps_factor=adopt_eps_factor,
            adopt_sector_categorical=adopt_sector_categorical,
            add_rank=add_rank
        )
        
        if preprocessing is not None:
            price_calculator.apply_preprocessing(preprocessing)
        
        return price_calculator


# 使用例
if __name__ == '__main__':
    from utils.paths import Paths
    
    # ファサードを使用したシンプルな特徴量データフレーム作成
    facade = FeaturesFacade()
    
    # 基本的な使用例
    features_df = facade.create_features_dataframe(
        sector_redefinitions_csv=f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv',
        sector_index_parquet=f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet'
    )
    
    print(f"特徴量データフレーム形状: {features_df.shape}")
    print(f"特徴量列数: {len(features_df.columns)}")
    print(features_df.head())
    
    # カスタマイズされた使用例
    custom_features_df = facade.create_features_dataframe(
        sector_redefinitions_csv=f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv',
        sector_index_parquet=f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet',
        stock_filter="(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))",
        use_index_features=True,
        use_price_features=True,
        mom_duration=[10, 30],
        vola_duration=[10, 30],
    )
    
    print(f"カスタム特徴量データフレーム形状: {custom_features_df.shape}")