#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
リファクタリング前後のコードが同じ結果を返すかテストするスクリプト
全セクターを単一のLightGBMモデルで学習・予測するケース
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from typing import Tuple, Dict, List, Any

# ユーティリティ関数
def setup_test_environment():
    """テスト環境のセットアップと必要なパスの設定"""
    import sys
    # プロジェクトのルートディレクトリをパスに追加
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.append(project_root)
    
    # 必要なディレクトリのパスを作成
    test_output_dir = os.path.join(project_root, 'test_output_lgbm')
    os.makedirs(test_output_dir, exist_ok=True)
    
    return {
        'project_root': project_root,
        'test_output_dir': test_output_dir
    }

def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    テストデータの読み込み
    
    既存のテストコードと同様の方法でデータを取得
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: target_df, features_df
    """
    from facades.stock_acquisition_facade import StockAcquisitionFacade
    from calculation.sector_index_calculator import SectorIndexCalculator
    from calculation.target_calculator import TargetCalculator
    from calculation.features_calculator import FeaturesCalculator
    from utils.paths import Paths
    
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet'
    train_start_date = datetime(2014, 1, 1)
    train_end_date = datetime(2021, 12, 31)

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
    
    return target_df, features_df

def save_data_for_testing(target_df: pd.DataFrame, features_df: pd.DataFrame, paths: Dict[str, str]):
    """テスト用にデータを一時的に保存"""
    target_path = os.path.join(paths['test_output_dir'], 'target_df.pkl')
    features_path = os.path.join(paths['test_output_dir'], 'features_df.pkl')
    
    with open(target_path, 'wb') as f:
        pickle.dump(target_df, f)
    
    with open(features_path, 'wb') as f:
        pickle.dump(features_df, f)
    
    return {
        'target_path': target_path,
        'features_path': features_path
    }

def load_saved_test_data(data_paths: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """保存したテストデータを読み込む"""
    with open(data_paths['target_path'], 'rb') as f:
        target_df = pickle.load(f)
    
    with open(data_paths['features_path'], 'rb') as f:
        features_df = pickle.load(f)
    
    return target_df, features_df

# リファクタリング前のコードを使用した学習と予測（全セクター単一モデル）
def train_and_predict_old_code_lgbm_single(target_df: pd.DataFrame, features_df: pd.DataFrame, 
                                          train_start_date: datetime, train_end_date: datetime) -> pd.DataFrame:
    """
    リファクタリング前のコードを使用して全セクターを1つのLightGBMモデルで学習・予測
    
    Args:
        target_df: 目的変数のデータフレーム
        features_df: 特徴量のデータフレーム
        train_start_date: 学習データの開始日
        train_end_date: 学習データの終了日
        
    Returns:
        pd.DataFrame: 予測結果のデータフレーム
    """
    from models import MLDataset
    from models.machine_learning import LgbmModel
    
    # テスト用のディレクトリ作成
    test_dataset_path = 'test_output_lgbm/old_model'
    os.makedirs(test_dataset_path, exist_ok=True)
    
    # データセットの準備
    ml_dataset = MLDataset(test_dataset_path, init_load=False)
    
    # 訓練データとテストデータの設定
    ml_dataset.archive_train_test_data(
        target_df=target_df,
        features_df=features_df,
        train_start_day=train_start_date,
        train_end_day=train_end_date,
        test_start_day=train_end_date,  # テストは学習期間の最後からスタート
        test_end_day=target_df.index.get_level_values('Date').max(),
        outlier_threshold=0  # 外れ値除去なし
    )
    
    # モデルの学習
    lgbm_model = LgbmModel()
    train_materials = ml_dataset.train_test_materials
    
    # カスタムパラメータの設定
    trainer_outputs = lgbm_model.train(
        target_train_df=train_materials.target_train_df,
        features_train_df=train_materials.features_train_df,
        num_leaves=31,
        learning_rate=0.05
    )
    
    # 学習結果の保存
    ml_dataset.archive_ml_objects(
        models=trainer_outputs.models,
        scalers=[]  # LightGBMはスケーラーを使用しない
    )
    
    # 予測の実行
    pred_result_df = lgbm_model.predict(
        target_test_df=train_materials.target_test_df,
        features_test_df=train_materials.features_test_df,
        models=trainer_outputs.models
    )
    
    # 予測結果の保存
    ml_dataset.archive_pred_result(pred_result_df=pred_result_df)
    
    return pred_result_df

# リファクタリング後のコードを使用した学習と予測（全セクター単一モデル）
def train_and_predict_new_code_lgbm_single(target_df: pd.DataFrame, features_df: pd.DataFrame, 
                                          train_start_date: datetime, train_end_date: datetime) -> pd.DataFrame:
    """
    リファクタリング後のコードを使用して全セクターを1つのLightGBMモデルで学習・予測
    
    Args:
        target_df: 目的変数のデータフレーム
        features_df: 特徴量のデータフレーム
        train_start_date: 学習データの開始日
        train_end_date: 学習データの終了日
        
    Returns:
        pd.DataFrame: 予測結果のデータフレーム
    """
    from machine_learning.factory import CollectionFactory
    from machine_learning.params import LgbmParams
    
    # コレクションの作成
    lgbm_collection = CollectionFactory.get_collection()
    
    # カスタムパラメータの設定
    custom_params = LgbmParams(
        num_leaves=31,
        learning_rate=0.05,
    )
    
    # モデル生成
    lgbm_collection.generate_model(name="all_sectors", type='lgbm', params=custom_params)
    single_model = lgbm_collection.get_model(name="all_sectors")
    
    # データセットの読み込み
    single_model.load_dataset(
        target_df=target_df,
        features_df=features_df,
        train_start_date=train_start_date,
        train_end_date=train_end_date
    )
    
    # モデルをコレクションに設定
    lgbm_collection.set_model(model=single_model)
    
    # モデルの学習
    lgbm_collection.train_all()
    
    # モデルの予測
    lgbm_collection.predict_all()
    
    # 予測結果の取得
    pred_result_df = lgbm_collection.get_result_df()
    
    # テスト用に一時保存
    collection_path = 'test_output_lgbm/new_model.pkl'
    os.makedirs(os.path.dirname(collection_path), exist_ok=True)
    lgbm_collection.save(path=collection_path)
    
    return pred_result_df

def compare_results(old_pred_df: pd.DataFrame, new_pred_df: pd.DataFrame) -> Dict[str, Any]:
    """
    リファクタリング前後の予測結果を比較する
    
    Args:
        old_pred_df: リファクタリング前のコードによる予測結果
        new_pred_df: リファクタリング後のコードによる予測結果
        
    Returns:
        Dict[str, Any]: 比較結果の統計情報
    """
    # 両方のデータフレームが同じインデックスを持っているか確認
    old_indices = set(old_pred_df.index)
    new_indices = set(new_pred_df.index)
    
    common_indices = old_indices.intersection(new_indices)
    old_only_indices = old_indices - new_indices
    new_only_indices = new_indices - old_indices
    
    # 予測値の差異を計算
    common_old_pred = old_pred_df.loc[list(common_indices)]
    common_new_pred = new_pred_df.loc[list(common_indices)]
    
    # 予測値の差の統計
    pred_diff = common_old_pred['Pred'] - common_new_pred['Pred']
    
    # 統計情報
    comparison_stats = {
        'common_indices_count': len(common_indices),
        'old_only_indices_count': len(old_only_indices),
        'new_only_indices_count': len(new_only_indices),
        'pred_difference_mean': pred_diff.mean(),
        'pred_difference_std': pred_diff.std(),
        'pred_difference_max': pred_diff.abs().max(),
        'pred_difference_min': pred_diff.abs().min(),
        'correlation': common_old_pred['Pred'].corr(common_new_pred['Pred']),
        'are_identical': np.allclose(common_old_pred['Pred'].values, common_new_pred['Pred'].values, rtol=1e-5, atol=1e-8)
    }
    
    return comparison_stats

def main():
    """メイン実行関数"""
    print("リファクタリング前後のLightGBM単一モデル比較テストを開始します...")
    
    # テスト環境のセットアップ
    paths = setup_test_environment()
    print(f"テスト環境を設定しました: {paths['test_output_dir']}")
    
    try:
        # データを取得
        print("テストデータを読み込みます...")
        target_df, features_df = load_test_data()
        
        # データを保存（必要に応じて後で再利用）
        data_paths = save_data_for_testing(target_df, features_df, paths)
        print(f"テストデータを一時保存しました: {data_paths}")
    except Exception as e:
        print(f"テストデータの読み込みに失敗しました: {e}")
        print("保存済みのテストデータを利用します...")
        # 既に保存されているデータを使用
        data_paths = {
            'target_path': os.path.join(paths['test_output_dir'], 'target_df.pkl'),
            'features_path': os.path.join(paths['test_output_dir'], 'features_df.pkl')
        }
        target_df, features_df = load_saved_test_data(data_paths)
    
    # 訓練期間の設定
    train_start_date = datetime(2014, 1, 1)
    train_end_date = datetime(2021, 12, 31)
    
    print(f"データ情報: target_df shape={target_df.shape}, features_df shape={features_df.shape}")
    print(f"学習期間: {train_start_date} から {train_end_date}")
    
    # リファクタリング前のコードで学習と予測
    print("\nリファクタリング前のコード（LightGBM単一モデル）で学習と予測を実行します...")
    try:
        old_pred_df = train_and_predict_old_code_lgbm_single(
            target_df, features_df, train_start_date, train_end_date
        )
        print(f"リファクタリング前のコードの予測結果: shape={old_pred_df.shape}")
        
        # 予測結果の簡単な統計情報
        print(f"予測値の統計: min={old_pred_df['Pred'].min()}, max={old_pred_df['Pred'].max()}, mean={old_pred_df['Pred'].mean()}")
    except Exception as e:
        print(f"リファクタリング前のコードの実行に失敗しました: {e}")
        old_pred_df = None
    
    # リファクタリング後のコードで学習と予測
    print("\nリファクタリング後のコード（LightGBM単一モデル）で学習と予測を実行します...")
    try:
        new_pred_df = train_and_predict_new_code_lgbm_single(
            target_df, features_df, train_start_date, train_end_date
        )
        print(f"リファクタリング後のコードの予測結果: shape={new_pred_df.shape}")
        
        # 予測結果の簡単な統計情報
        print(f"予測値の統計: min={new_pred_df['Pred'].min()}, max={new_pred_df['Pred'].max()}, mean={new_pred_df['Pred'].mean()}")
    except Exception as e:
        print(f"リファクタリング後のコードの実行に失敗しました: {e}")
        new_pred_df = None
    
    # 予測結果の比較
    if old_pred_df is not None and new_pred_df is not None:
        print("\n予測結果を比較します...")
        comparison_stats = compare_results(old_pred_df, new_pred_df)
        
        print("\n比較結果:")
        for key, value in comparison_stats.items():
            print(f"{key}: {value}")
        
        if comparison_stats['are_identical']:
            print("\n✓ テスト結果: リファクタリング前後のコードは同じ結果を生成します。")
        else:
            print("\n✗ テスト結果: リファクタリング前後のコードは異なる結果を生成します。")
            print(f"  - 差の平均: {comparison_stats['pred_difference_mean']}")
            print(f"  - 差の標準偏差: {comparison_stats['pred_difference_std']}")
            print(f"  - 相関係数: {comparison_stats['correlation']}")
            
            # LightGBMは乱数の影響を受けるため、高い相関があれば実質的に同等と見なせる場合がある
            if comparison_stats['correlation'] > 0.99:
                print("  - 相関係数が非常に高いため、実質的に同等の結果と見なせます（LightGBMの乱数の影響を考慮）")
    else:
        print("\n予測結果の比較ができませんでした。どちらかまたは両方のコードの実行に失敗しています。")
    
    print("\nテスト完了")

if __name__ == "__main__":
    main()