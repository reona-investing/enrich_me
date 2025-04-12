#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
異なるモデルコレクションのアンサンブル機能をテストするスクリプト
"""

import os
import sys
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from facades.stock_acquisition_facade import StockAcquisitionFacade
from calculation.sector_index_calculator import SectorIndexCalculator
from calculation.target_calculator import TargetCalculator
from calculation.features_calculator import FeaturesCalculator
from utils.paths import Paths

from machine_learning.factory import CollectionFactory
from machine_learning.ensemble import Ensemble


def create_and_train_collection(target_df, features_df, model_type, train_start_date, train_end_date):
    """
    指定されたタイプのモデルコレクションを作成し学習する
    
    Args:
        target_df: 目的変数のデータフレーム
        features_df: 特徴量のデータフレーム
        model_type: モデルタイプ ('lasso' または 'lgbm')
        train_start_date: 学習開始日
        train_end_date: 学習終了日
        
    Returns:
        作成・学習済みのモデルコレクション
    """
    # セクターのリストを取得
    sectors = target_df.index.get_level_values('Sector').unique().tolist()
    
    # コレクションの作成
    collection = CollectionFactory.get_collection()
    
    # セクターごとにモデルを生成
    for sector in sectors:
        target_for_sector = target_df[target_df.index.get_level_values('Sector') == sector]
        features_for_sector = features_df[features_df.index.get_level_values('Sector') == sector]
        
        # モデル生成
        collection.generate_model(name=sector, type=model_type)
        single_model = collection.get_model(name=sector)
        
        # データセットの読み込み
        single_model.load_dataset(
            target_df=target_for_sector,
            features_df=features_for_sector,
            train_start_date=train_start_date,
            train_end_date=train_end_date
        )
        
        # モデルをコレクションに設定
        collection.set_model(model=single_model)
    
    # 全モデルの学習
    collection.train_all()
    
    # 全モデルの予測実行
    collection.predict_all()
    
    return collection


def main():
    """メイン実行関数"""
    print("モデルコレクションのアンサンブルテストを開始します...")
    
    # テストデータの準備
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet'
    train_start_date = datetime(2014, 1, 1)
    train_end_date = datetime(2021, 12, 31)

    print("データを読み込み中...")
    stock_dfs = StockAcquisitionFacade(filter=universe_filter).get_stock_data_dict()
    sector_index_df, _ = SectorIndexCalculator.calc_new_sector_price(stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)

    raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(
        sector_index_df, reduce_components=1, 
        train_start_day=train_start_date, train_end_day=train_end_date
    )
    
    features_df = FeaturesCalculator.calculate_features(
        sector_index_df, None, None,
        adopts_features_indices=True, adopts_features_price=False,
        groups_setting=None, names_setting=None, currencies_type='relative',
        adopt_1d_return=True, mom_duration=None, vola_duration=None,
        adopt_size_factor=False, adopt_eps_factor=False,
        adopt_sector_categorical=False, add_rank=False
    )
    
    print("LASSOモデルコレクションを学習中...")
    lasso_collection = create_and_train_collection(
        target_df, features_df, 'lasso', train_start_date, train_end_date
    )
    
    print("LightGBMモデルコレクションを学習中...")
    lgbm_collection = create_and_train_collection(
        target_df, features_df, 'lgbm', train_start_date, train_end_date
    )
    
    # 各モデルコレクションの予測結果
    lasso_pred_df = lasso_collection.get_result_df()
    lgbm_pred_df = lgbm_collection.get_result_df()
    
    print(f"LASSOモデル予測結果: {lasso_pred_df.shape}")
    print(f"LightGBMモデル予測結果: {lgbm_pred_df.shape}")
    
    # アンサンブル
    print("予測結果をアンサンブルします...")
    collections_with_weights = [
        (lasso_collection, 0.7),  # LASSOモデルには0.7の重みを付ける
        (lgbm_collection, 0.3)    # LightGBMモデルには0.3の重みを付ける
    ]
    
    ensembled_df = Ensemble.ensemble_by_rank(collections_with_weights)
    
    print(f"アンサンブル後の予測結果: {ensembled_df.shape}")
    print("\nアンサンブル結果サンプル:")
    print(ensembled_df.head(10))
    
    print("\nアンサンブルテスト完了")


if __name__ == "__main__":
    main()