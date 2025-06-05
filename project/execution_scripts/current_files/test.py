#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
リファクタリング前後のアンサンブル結果比較テスト

このスクリプトでは以下を実行します：
1. 共通テストデータの準備
2. 元のモジュール（models）を使用したLASSOとLightGBMのアンサンブル
3. 新しいモジュール（リファクタリング後）を使用したアンサンブル
4. 結果の比較と検証
"""

import os
import sys
import shutil
import tempfile
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

# 一時ディレクトリの作成（テスト用）
TEST_DIR = tempfile.mkdtemp(prefix="refactoring_ensemble_test_")


def setup_environment():
    """テスト環境のセットアップ"""
    print("テスト環境をセットアップしています...")
    
    # テスト用のディレクトリを作成
    os.makedirs(os.path.join(TEST_DIR, 'original'), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, 'refactored'), exist_ok=True)
    
    # プロジェクトのルートディレクトリをパスに追加
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    print(f"テスト用一時ディレクトリ: {TEST_DIR}")
    return root_dir


def prepare_test_data() -> Dict[str, pd.DataFrame]:
    """テストデータの準備"""
    print("テストデータを準備しています...")
    
    # プロジェクト内のモジュールをインポート
    from acquisition.jquants_api_operations.facades.stock_acquisition_facade import StockAcquisitionFacade
    from calculation.sector_index_calculator import SectorIndex
    from calculation.target_calculator import TargetCalculator
    from calculation.features_calculator import FeaturesCalculator
    from utils.paths import Paths
    
    # テスト用のパラメータ
    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    SECTOR_INDEX_PARQUET = os.path.join(TEST_DIR, 'New48sectors_price.parquet')
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70'))"  # テスト用に範囲を絞る
    
    # サンプル期間（短めの期間を指定）
    train_start_day = datetime(2020, 1, 1)
    train_end_day = datetime(2020, 12, 31)
    
    try:
        # 株価データの取得（範囲を限定）
        print("株価データを取得しています...")
        stock_dfs = StockAcquisitionFacade(filter=universe_filter).get_stock_data_dict()
        
        # セクターインデックスの計算
        print("セクターインデックスを計算しています...")
        sic = SectorIndex()
        sector_index_df, order_price_df = sic.calc_sector_index(
            stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET
        )
        
        # 目的変数の計算
        print("目的変数を計算しています...")
        raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(
            sector_index_df, reduce_components=1, 
            train_start_day=train_start_day, train_end_day=train_end_day
        )
        
        # LASSO用の特徴量を計算
        print("LASSO用特徴量を計算しています...")
        lasso_features_df = FeaturesCalculator.calculate_features(
            sector_index_df, None, None,
            adopts_features_indices=True, adopts_features_price=False,
            groups_setting=None, names_setting=None, currencies_type='relative',
            adopt_1d_return=True, mom_duration=None, vola_duration=None,
            adopt_size_factor=False, adopt_eps_factor=False,
            adopt_sector_categorical=False, add_rank=False
        )
        
        # LightGBM用の特徴量を計算
        print("LightGBM用特徴量を計算しています...")
        lgbm_features_df = FeaturesCalculator.calculate_features(
            sector_index_df, 
            pd.read_csv(SECTOR_REDEFINITIONS_CSV), stock_dfs,
            adopts_features_indices=True, adopts_features_price=True,
            groups_setting=None, names_setting=None, currencies_type='relative',
            adopt_1d_return=True, mom_duration=[5, 21], vola_duration=[5, 21],
            adopt_size_factor=True, adopt_eps_factor=True,
            adopt_sector_categorical=True, add_rank=True
        )
        
        # データを一時保存
        test_data = {
            'sector_index_df': sector_index_df,
            'order_price_df': order_price_df,
            'raw_target_df': raw_target_df,
            'target_df': target_df,
            'lasso_features_df': lasso_features_df,
            'lgbm_features_df': lgbm_features_df,
            'redefinitions_csv': SECTOR_REDEFINITIONS_CSV,
            'stock_dfs': stock_dfs
        }
        
        # ファイルに保存（後で再利用するため）
        with open(os.path.join(TEST_DIR, 'test_data.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        
        return test_data
    
    except Exception as e:
        print(f"テストデータの準備中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_original_ensemble(test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """元のモジュールを使用したアンサンブル処理を実行"""
    print("\n--- オリジナルコードでアンサンブルを実行 ---")
    
    # 元のモジュールをインポート
    from models import MLDataset
    from models.machine_learning import LassoModel, LgbmModel
    import models.ensemble as ensembles
    from models.loader import load_datasets
    
    # テスト用のパス
    ML_DATASET_PATH1 = os.path.join(TEST_DIR, 'original', 'lasso_model')
    ML_DATASET_PATH2 = os.path.join(TEST_DIR, 'original', 'lgbm_model')
    ML_DATASET_ENSEMBLED_PATH = os.path.join(TEST_DIR, 'original', 'ensemble_model')
    
    # パラメータ
    train_start_day = datetime(2020, 1, 1)
    train_end_day = datetime(2020, 6, 30)  # 半年分を学習に使用
    test_start_day = datetime(2020, 7, 1)  # それ以降をテストに
    test_end_day = datetime(2020, 12, 31)
    
    # 学習と予測の開始時間を記録
    import time
    start_time = time.time()
    
    # 1. LASSO モデルの学習と予測
    print("LASSOモデルを学習・予測しています...")
    ml_dataset1 = MLDataset(ML_DATASET_PATH1, init_load=False)
    ml_dataset1.archive_train_test_data(
        target_df=test_data['target_df'],
        features_df=test_data['lasso_features_df'],
        train_start_day=train_start_day,
        train_end_day=train_end_day,
        test_start_day=test_start_day,
        test_end_day=test_end_day,
        outlier_threshold=3.0
    )
    ml_dataset1.archive_raw_target(test_data['raw_target_df'])
    ml_dataset1.archive_order_price(test_data['order_price_df'])
    
    lasso_model = LassoModel()
    trainer_outputs = lasso_model.train(
        ml_dataset1.train_test_materials.target_train_df,
        ml_dataset1.train_test_materials.features_train_df
    )
    ml_dataset1.archive_ml_objects(trainer_outputs.models, trainer_outputs.scalers)
    
    pred_result_df1 = lasso_model.predict(
        ml_dataset1.train_test_materials.target_test_df,
        ml_dataset1.train_test_materials.features_test_df,
        ml_dataset1.ml_object_materials.models,
        ml_dataset1.ml_object_materials.scalers
    )
    ml_dataset1.archive_pred_result(pred_result_df1)
    ml_dataset1.save()
    
    # 2. LightGBM モデルの学習と予測
    print("LightGBMモデルを学習・予測しています...")
    
    # LASSO予測を特徴量に追加
    lgbm_features_with_lasso = test_data['lgbm_features_df'].copy()
    lgbm_features_with_lasso = pd.merge(
        lgbm_features_with_lasso, 
        pred_result_df1[['Pred']], 
        how='outer', 
        left_index=True, 
        right_index=True
    )
    lgbm_features_with_lasso = lgbm_features_with_lasso.rename(columns={'Pred': '1stModel_pred'})
    
    ml_dataset2 = MLDataset(ML_DATASET_PATH2, init_load=False)
    ml_dataset2.archive_train_test_data(
        target_df=test_data['target_df'],
        features_df=lgbm_features_with_lasso,
        train_start_day=train_start_day,
        train_end_day=train_end_day,
        test_start_day=test_start_day,
        test_end_day=test_end_day,
        outlier_threshold=3.0,
        no_shift_features=['1stModel_pred'],
        reuse_features_df=True
    )
    ml_dataset2.archive_raw_target(test_data['raw_target_df'])
    ml_dataset2.archive_order_price(test_data['order_price_df'])
    
    lgbm_model = LgbmModel()
    trainer_outputs = lgbm_model.train(
        ml_dataset2.train_test_materials.target_train_df,
        ml_dataset2.train_test_materials.features_train_df,
        categorical_features=['Sector_cat']
    )
    ml_dataset2.archive_ml_objects(trainer_outputs.models, None)  # LightGBMはスケーラーなし
    
    pred_result_df2 = lgbm_model.predict(
        ml_dataset2.train_test_materials.target_test_df,
        ml_dataset2.train_test_materials.features_test_df,
        ml_dataset2.ml_object_materials.models
    )
    ml_dataset2.archive_pred_result(pred_result_df2)
    ml_dataset2.save()
    
    # 3. アンサンブル
    print("アンサンブルを実行しています...")
    ensemble_weights = [6.7, 1.3]
    ensemble_inputs = [(pred_result_df1, ensemble_weights[0]), (pred_result_df2, ensemble_weights[1])]
    ensembled_pred_df = ensembles.by_rank(inputs=ensemble_inputs)
    
    # ターゲット列を追加
    ensembled_result_df = test_data['target_df'].loc[ensembled_pred_df.index, ['Target']].copy()
    ensembled_result_df['Pred'] = ensembled_pred_df['Pred']
    
    # アンサンブル結果を保存
    ml_dataset_ensembled = MLDataset(ML_DATASET_ENSEMBLED_PATH, init_load=False)
    ml_dataset_ensembled.copy_from_other_dataset(ml_dataset1)
    ml_dataset_ensembled.archive_pred_result(ensembled_result_df)
    ml_dataset_ensembled.save()
    
    # 終了時間を記録
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        'lasso_dataset': ml_dataset1,
        'lgbm_dataset': ml_dataset2,
        'ensemble_dataset': ml_dataset_ensembled,
        'lasso_pred_df': pred_result_df1,
        'lgbm_pred_df': pred_result_df2,
        'ensemble_pred_df': ensembled_result_df,
        'execution_time': execution_time
    }


def run_refactored_ensemble(test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """リファクタリング後のモジュールを使用したアンサンブル処理を実行"""
    print("\n--- リファクタリング後のコードでアンサンブルを実行 ---")
    
    # リファクタリング後のモジュールをインポート
    from machine_learning.strategies import SectorLassoStrategy, SingleLgbmStrategy, EnsembleStrategy
    
    # テスト用のパス
    ML_DATASET_PATH1 = os.path.join(TEST_DIR, 'refactored', 'lasso_model.pkl')
    ML_DATASET_PATH2 = os.path.join(TEST_DIR, 'refactored', 'lgbm_model.pkl')
    ML_DATASET_ENSEMBLED_PATH = os.path.join(TEST_DIR, 'refactored', 'ensemble_model.pkl')
    
    # パラメータ
    train_start_day = datetime(2020, 1, 1)
    train_end_day = datetime(2020, 6, 30)  # 半年分を学習に使用
    test_start_day = datetime(2020, 7, 1)  # それ以降をテストに
    test_end_day = datetime(2020, 12, 31)
    
    # 学習と予測の開始時間を記録
    import time
    start_time = time.time()
    
    # 1. LASSO モデルの学習と予測
    print("LASSOモデルを学習・予測しています...")
    lasso_collection = SectorLassoStrategy.run(
        path=ML_DATASET_PATH1,
        target_df=test_data['target_df'],
        features_df=test_data['lasso_features_df'],
        raw_target_df=test_data['raw_target_df'],
        order_price_df=test_data['order_price_df'],
        train_start_date=train_start_day,
        train_end_date=train_end_day,
        test_start_date=test_start_day,
        test_end_date=test_end_day,
        train=True,
        outlier_threshold=3.0
    )
    
    # 2. LightGBM モデルの学習と予測
    print("LightGBMモデルを学習・予測しています...")
    
    # LASSO予測を特徴量に追加
    lasso_pred_df = lasso_collection.get_result_df()
    lgbm_features_with_lasso = test_data['lgbm_features_df'].copy()
    lgbm_features_with_lasso = pd.merge(
        lgbm_features_with_lasso, 
        lasso_pred_df[['Pred']], 
        how='outer', 
        left_index=True, 
        right_index=True
    )
    lgbm_features_with_lasso = lgbm_features_with_lasso.rename(columns={'Pred': '1stModel_pred'})
    
    lgbm_collection = SingleLgbmStrategy.run(
        path=ML_DATASET_PATH2,
        target_df=test_data['target_df'],
        features_df=lgbm_features_with_lasso,
        raw_target_df=test_data['raw_target_df'],
        order_price_df=test_data['order_price_df'],
        train_start_date=train_start_day,
        train_end_date=train_end_day,
        test_start_date=test_start_day,
        test_end_date=test_end_day,
        train=True,
        outlier_threshold=3.0,
        no_shift_features=['1stModel_pred'],
        reuse_features_df=True
    )
    
    # 3. アンサンブル
    print("アンサンブルを実行しています...")
    ensemble_weights = [6.7, 1.3]
    
    ensemble_collection = EnsembleStrategy.run(
        path=ML_DATASET_ENSEMBLED_PATH,
        # これらはダミー引数（アンサンブル戦略では使用しない）
        target_df=pd.DataFrame(),
        features_df=pd.DataFrame(),
        raw_target_df=pd.DataFrame(),
        order_price_df=pd.DataFrame(),
        train_start_date=datetime(2000, 1, 1),
        train_end_date=datetime(2000, 1, 1),
        # 実際に使用するパラメータ
        collection_paths=[ML_DATASET_PATH1, ML_DATASET_PATH2],
        weights=ensemble_weights,
        ensemble_method='rank'
    )
    
    # 終了時間を記録
    end_time = time.time()
    execution_time = end_time - start_time
    
    return {
        'lasso_collection': lasso_collection,
        'lgbm_collection': lgbm_collection,
        'ensemble_collection': ensemble_collection,
        'lasso_pred_df': lasso_pred_df,
        'lgbm_pred_df': lgbm_collection.get_result_df(),
        'ensemble_pred_df': ensemble_collection.get_result_df(),
        'execution_time': execution_time
    }


def compare_ensemble_results(original_results: Dict[str, Any], refactored_results: Dict[str, Any]) -> Dict[str, Any]:
    """元のコードとリファクタリング後のコードのアンサンブル結果を比較"""
    print("\n--- アンサンブル結果の比較 ---")
    
    # 予測結果の取得
    original_pred_df = original_results['ensemble_pred_df']
    refactored_pred_df = refactored_results['ensemble_pred_df']
    
    # インデックスのチェック
    original_indices = set(original_pred_df.index)
    refactored_indices = set(refactored_pred_df.index)
    
    common_indices = original_indices.intersection(refactored_indices)
    original_only = original_indices - refactored_indices
    refactored_only = refactored_indices - original_indices
    
    # 共通のインデックスでの予測結果の比較
    if common_indices:
        original_common = original_pred_df.loc[list(common_indices)]
        refactored_common = refactored_pred_df.loc[list(common_indices)]
        
        # 予測値の差分統計
        pred_diff = original_common['Pred'] - refactored_common['Pred']
        correlation = original_common['Pred'].corr(refactored_common['Pred'])
        
        is_identical = np.allclose(
            original_common['Pred'].values, 
            refactored_common['Pred'].values, 
            rtol=1e-5, atol=1e-8
        )
    else:
        pred_diff = None
        correlation = None
        is_identical = False
    
    # 実行時間の比較
    time_diff = original_results['execution_time'] - refactored_results['execution_time']
    time_ratio = original_results['execution_time'] / refactored_results['execution_time'] if refactored_results['execution_time'] > 0 else float('inf')
    
    comparison_results = {
        'common_indices_count': len(common_indices),
        'original_only_count': len(original_only),
        'refactored_only_count': len(refactored_only),
        'is_identical': is_identical,
        'correlation': correlation,
        'mean_diff': pred_diff.mean() if pred_diff is not None else None,
        'std_diff': pred_diff.std() if pred_diff is not None else None,
        'max_abs_diff': pred_diff.abs().max() if pred_diff is not None else None,
        'original_time': original_results['execution_time'],
        'refactored_time': refactored_results['execution_time'],
        'time_diff': time_diff,
        'time_ratio': time_ratio
    }
    
    return comparison_results


def print_ensemble_comparison_results(comparison_results: Dict[str, Any]) -> None:
    """アンサンブル比較結果を出力"""
    print("\n==== アンサンブル比較結果のサマリー ====")
    
    # インデックスの比較
    print(f"共通インデックス数: {comparison_results['common_indices_count']}")
    print(f"元コードのみ存在するインデックス数: {comparison_results['original_only_count']}")
    print(f"リファクタリング後コードのみ存在するインデックス数: {comparison_results['refactored_only_count']}")
    
    # 予測値の比較
    if comparison_results['is_identical']:
        print("\n✓ アンサンブル予測結果: 完全に一致しています。")
    else:
        print("\n✗ アンサンブル予測結果: 差異があります。")
        print(f"  - 相関係数: {comparison_results['correlation']:.6f}")
        print(f"  - 平均差分: {comparison_results['mean_diff']:.6e}")
        print(f"  - 標準偏差: {comparison_results['std_diff']:.6e}")
        print(f"  - 最大絶対差分: {comparison_results['max_abs_diff']:.6e}")
        
        # 相関係数が非常に高い場合は実質的に同等と見なす
        if comparison_results['correlation'] is not None and comparison_results['correlation'] > 0.9999:
            print("  - 相関係数が非常に高いため、数値的に同等と見なせます。")
    
    # 実行時間の比較
    print("\nアンサンブル処理時間比較:")
    print(f"  - 元コード: {comparison_results['original_time']:.2f} 秒")
    print(f"  - リファクタリング後: {comparison_results['refactored_time']:.2f} 秒")
    print(f"  - 差分: {comparison_results['time_diff']:.2f} 秒")
    
    if comparison_results['time_ratio'] > 1:
        print(f"  - リファクタリング後のコードは {comparison_results['time_ratio']:.2f} 倍高速です。")
    else:
        print(f"  - リファクタリング後のコードは {1/comparison_results['time_ratio']:.2f} 倍遅いです。")


def cleanup():
    """テスト終了後のクリーンアップ"""
    print(f"\nテスト一時ディレクトリ ({TEST_DIR}) を削除しています...")
    shutil.rmtree(TEST_DIR, ignore_errors=True)


def main():
    """テストのメイン関数"""
    try:
        # 環境設定
        setup_environment()
        
        # テストデータの準備
        test_data = prepare_test_data()
        
        # 元のコードを実行
        original_results = run_original_ensemble(test_data)
        
        # リファクタリング後のコードを実行
        refactored_results = run_refactored_ensemble(test_data)
        
        # 結果の比較
        comparison_results = compare_ensemble_results(original_results, refactored_results)
        
        # 比較結果の出力
        print_ensemble_comparison_results(comparison_results)
        
    except Exception as e:
        import traceback
        print(f"\nテスト実行中にエラーが発生しました: {e}")
        traceback.print_exc()
        
    finally:
        # クリーンアップ
        cleanup()


if __name__ == "__main__":
    main()