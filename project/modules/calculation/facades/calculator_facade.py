import pandas as pd
from typing import Literal, Optional, Tuple
from calculation.sector_index.sector_index import SectorIndex
from calculation.features_calculator import FeaturesCalculator
from preprocessing.pipeline import PreprocessingPipeline


class CalculatorFacade:
    """
    SectorIndexとFeaturesCalculatorを組み合わせて
    一連の処理を実行するファサードクラス
    """
    
    @staticmethod
    def calculate_all(
        stock_dfs: dict,
        sector_redefinitions_csv: str,
        sector_index_parquet: str,
        # FeaturesCalculatorのパラメータ
        adopts_features_indices: bool = True,
        adopts_features_price: bool = True,
        groups_setting: Optional[dict] = None,
        names_setting: Optional[dict] = None,
        currencies_type: Literal['relative', 'raw'] = 'relative',
        commodity_type: Literal['JPY', 'raw'] = 'raw',
        adopt_1d_return: bool = True,
        mom_duration: Optional[list] = None,
        vola_duration: Optional[list] = None,
        adopt_size_factor: bool = True,
        adopt_eps_factor: bool = True,
        adopt_sector_categorical: bool = True,
        add_rank: bool = True,
        # 前処理パイプラインのパラメータ
        indices_preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        price_preprocessing_pipeline: Optional[PreprocessingPipeline] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        セクターインデックス計算と特徴量計算を一連で実行する
        
        Args:
            stock_dfs (dict): 'list', 'fin', 'price'キーを持つ株価データ辞書
            sector_redefinitions_csv (str): セクター定義CSVファイルのパス
            sector_index_parquet (str): セクターインデックス出力用parquetファイルのパス
            adopts_features_indices (bool): インデックス系特徴量の採否
            adopts_features_price (bool): 価格系特徴量の採否
            groups_setting (dict, optional): 特徴量グループの採否設定
            names_setting (dict, optional): 特徴量の採否設定
            currencies_type (str): 通貨処理方法 ('relative' or 'raw')
            commodity_type (str): コモディティ処理方法 ('JPY' or 'raw')
            adopt_1d_return (bool): 1日リターンを特徴量とするか
            mom_duration (list, optional): モメンタム算出日数リスト
            vola_duration (list, optional): ボラティリティ算出日数リスト
            adopt_size_factor (bool): サイズファクターを特徴量とするか
            adopt_eps_factor (bool): EPSを特徴量とするか
            adopt_sector_categorical (bool): セクターをカテゴリ変数として採用するか
            add_rank (bool): 各日・各指標の業種別ランキングを追加するか
            indices_preprocessing_pipeline (PreprocessingPipeline, optional): インデックス系特徴量の前処理パイプライン
            price_preprocessing_pipeline (PreprocessingPipeline, optional): 価格系特徴量の前処理パイプライン
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - new_sector_price: セクターインデックス価格データ
                - order_price_df: 発注用個別銘柄データ
                - features_df: 特徴量データ
        """
        # デフォルト値の設定
        if mom_duration is None:
            mom_duration = [5, 21]
        if vola_duration is None:
            vola_duration = [5, 21]
        if groups_setting is None:
            groups_setting = {}
        if names_setting is None:
            names_setting = {}
        
        # 1. セクターインデックスの計算
        sic = SectorIndex()
        new_sector_price, order_price_df = sic.calc_sector_index(
            stock_dfs_dict=stock_dfs,
            SECTOR_REDEFINITIONS_CSV=sector_redefinitions_csv,
            SECTOR_INDEX_PARQUET=sector_index_parquet
        )
        
        # 2. セクター定義の読み込み（特徴量計算用）
        new_sector_list = pd.read_csv(sector_redefinitions_csv)
        
        # 3. 特徴量の計算（前処理パイプライン付き）
        features_df = FeaturesCalculator.calculate_features(
            new_sector_price=new_sector_price,
            new_sector_list=new_sector_list,
            stock_dfs_dict=stock_dfs,
            adopts_features_indices=adopts_features_indices,
            adopts_features_price=adopts_features_price,
            groups_setting=groups_setting,
            names_setting=names_setting,
            currencies_type=currencies_type,
            commodity_type=commodity_type,
            adopt_1d_return=adopt_1d_return,
            mom_duration=mom_duration,
            vola_duration=vola_duration,
            adopt_size_factor=adopt_size_factor,
            adopt_eps_factor=adopt_eps_factor,
            adopt_sector_categorical=adopt_sector_categorical,
            add_rank=add_rank,
            indices_preprocessing_pipeline=indices_preprocessing_pipeline,
            price_preprocessing_pipeline=price_preprocessing_pipeline
        )
        
        return new_sector_price, order_price_df, features_df
