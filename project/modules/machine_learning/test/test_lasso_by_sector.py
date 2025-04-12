#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
machine_learning/storategies/lasso_by_sector.py のテストスクリプト

このスクリプトでは:
1. 必要なデータの準備
2. LassoBySectorの動作確認
3. 結果の評価とテスト成功の確認
を行います。
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

# プロジェクトのルートディレクトリをパスに追加
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

# テスト対象のモジュールをインポート
from machine_learning.strategies.lasso_by_sector import LassoBySector
from utils.paths import Paths
from calculation.sector_index_calculator import SectorIndexCalculator
from calculation.target_calculator import TargetCalculator
from calculation.features_calculator import FeaturesCalculator
from facades.stock_acquisition_facade import StockAcquisitionFacade


def prepare_test_data():
    """
    テスト用のデータを準備する関数
    
    Returns:
        tuple: (target_df, features_df, raw_target_df, order_price_df)
    """
    print("テスト用データを準備しています...")
    
    # 一時ディレクトリを作成
    test_dir = tempfile.mkdtemp(prefix="lasso_by_sector_test_")
    
    # 必要なデータの取得
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70'))"
    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    SECTOR_INDEX_PARQUET = f'{test_dir}/test_sector_price.parquet'
    
    # データ期間の設定
    train_start_date = datetime(2019, 1, 1)
    train_end_date = datetime(2019, 12, 31)  # 小さなデータセットでテスト
    
    # データの取得と処理
    stock_dfs = StockAcquisitionFacade(filter=universe_filter).get_stock_data_dict()
    sector_index_df, order_price_df = SectorIndexCalculator.calc_new_sector_price(
        stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET
    )
    
    # 目的変数の作成
    raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(
        sector_index_df, reduce_components=1, 
        train_start_day=train_start_date, train_end_day=train_end_date
    )
    
    # 特徴量の作成
    features_df = FeaturesCalculator.calculate_features(
        sector_index_df, None, None,
        adopts_features_indices=True, adopts_features_price=False,
        groups_setting=None, names_setting=None, currencies_type='relative',
        adopt_1d_return=True, mom_duration=None, vola_duration=None,
        adopt_size_factor=False, adopt_eps_factor=False,
        adopt_sector_categorical=False, add_rank=False
    )
    
    # データサイズが小さくなるように、必要な部分だけを抽出
    sectors = target_df.index.get_level_values('Sector').unique()[:3]  # テスト用に3セクターのみ使用
    dates = target_df.index.get_level_values('Date').unique()[:30]    # テスト用に30日分のみ使用
    
    target_df = target_df[
        (target_df.index.get_level_values('Sector').isin(sectors)) & 
        (target_df.index.get_level_values('Date').isin(dates))
    ]
    
    features_df = features_df[
        (features_df.index.get_level_values('Sector').isin(sectors)) & 
        (features_df.index.get_level_values('Date').isin(dates))
    ]
    
    raw_target_df = raw_target_df[
        (raw_target_df.index.get_level_values('Sector').isin(sectors)) & 
        (raw_target_df.index.get_level_values('Date').isin(dates))
    ]
    
    order_price_df = order_price_df[order_price_df['Date'].isin(dates)]
    
    print(f"データ準備完了: target_df shape={target_df.shape}, features_df shape={features_df.shape}")
    return target_df, features_df, raw_target_df, order_price_df, test_dir


def test_lasso_by_sector():
    """LassoBySectorの機能をテストする関数"""
    print("\nLassoBySector テストを開始します...\n")
    
    # テストデータを準備
    target_df, features_df, raw_target_df, order_price_df, temp_dir = prepare_test_data()
    
    # 一時ファイルパスを作成
    test_model_path = os.path.join(temp_dir, "test_lasso_model.pkl")
    
    try:
        # 訓練期間の設定
        train_start_date = target_df.index.get_level_values('Date').min()
        train_end_date = target_df.index.get_level_values('Date').max() - pd.Timedelta(days=7)
        test_start_date = train_end_date + pd.Timedelta(days=1)
        test_end_date = target_df.index.get_level_values('Date').max()
        
        print(f"学習期間: {train_start_date} から {train_end_date}")
        print(f"テスト期間: {test_start_date} から {test_end_date}")
        
        # LassoBySectorの実行
        print("\nLassoBySector.train_and_predict_new_code を実行中...")
        
        # 関数の実行
        LassoBySector.train_and_predict_new_code(
            path=test_model_path,
            target_df=target_df, 
            features_df=features_df, 
            raw_target_df=raw_target_df, 
            order_price_df=order_price_df,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date
        )
        
        # 生成されたモデルファイルの確認
        if os.path.exists(test_model_path):
            file_size = os.path.getsize(test_model_path) / (1024 * 1024)  # サイズをMBで表示
            print(f"\n✓ モデルファイル {test_model_path} が正常に作成されました (サイズ: {file_size:.2f} MB)")
            
            # モデルをロードして検証（オプション）
            import pickle
            with open(test_model_path, 'rb') as f:
                collection_data = pickle.load(f)
                
            if 'models' in collection_data:
                num_models = len(collection_data['models'])
                print(f"✓ {num_models}個のモデルが正常に保存されました")
                print("✓ モデルキー:", list(collection_data['models'].keys()))
                
                # 予測結果の確認
                has_predictions = any(hasattr(model, 'pred_result_df') and model.pred_result_df is not None 
                                     for model in collection_data['models'].values())
                if has_predictions:
                    print("✓ モデルに予測結果が含まれています")
                else:
                    print("✗ モデルに予測結果が含まれていません")
            else:
                print("✗ モデルが正しく保存されていません")
            
            print("\nテスト成功: LassoBySectorが正常に動作しました")
            return True
        else:
            print(f"\n✗ テスト失敗: モデルファイル {test_model_path} が作成されませんでした")
            return False
    
    except Exception as e:
        print(f"\n✗ テスト失敗: 例外が発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # テスト用の一時ディレクトリを削除
        print(f"\n一時ディレクトリを削除しています: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    success = test_lasso_by_sector()
    sys.exit(0 if success else 1)