import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

# モジュールのパスを追加
module_path = str(Path(__file__).parent.parent)
if module_path not in sys.path:
    sys.path.append(module_path)

# リファクタリング前後のモジュールをインポート
from calculation.features_calculator import FeaturesCalculator as OldFeaturesCalculator
from calculation.refactored_features_calculator import main as new_features_calculator_main
from acquisition.jquants_api_operations.facades.stock_acquisition_facade import StockAcquisitionFacade
from utils.paths import Paths
from calculation.sector_index_calculator import SectorIndex


def compare_dataframes(old_df, new_df, name=""):
    """
    二つのデータフレームが同じ値を持っているかチェックする関数
    """
    # カラムをソートして同じ順番にする
    old_df = old_df.sort_index(axis=1)
    new_df = new_df.sort_index(axis=1)
    
    # カラム名のセットが同じか確認
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    
    if old_cols != new_cols:
        print(f"{name} - カラム名が一致しません:")
        print(f"旧実装にしかないカラム: {old_cols - new_cols}")
        print(f"新実装にしかないカラム: {new_cols - old_cols}")
        
        # 共通のカラムだけを比較対象にする
        common_cols = old_cols.intersection(new_cols)
        old_df = old_df[list(common_cols)]
        new_df = new_df[list(common_cols)]
    
    # インデックスの形状を確認
    if old_df.index.names != new_df.index.names:
        print(f"{name} - インデックス名が一致しません:")
        print(f"旧実装: {old_df.index.names}")
        print(f"新実装: {new_df.index.names}")
    
    # インデックスの値を確認
    if len(old_df) != len(new_df) or not old_df.index.equals(new_df.index):
        print(f"{name} - インデックスの値または長さが一致しません")
        print(f"旧実装の行数: {len(old_df)}, 新実装の行数: {len(new_df)}")
        
        # インデックスを再設定して値だけを比較
        old_df = old_df.reset_index(drop=True)
        new_df = new_df.reset_index(drop=True)
    
    # データ値を比較
    try:
        are_equal = old_df.equals(new_df)
        if not are_equal:
            # 差分を詳しく調べる
            diff_mask = ~(old_df == new_df)
            if diff_mask.any().any():
                diff_count = diff_mask.sum().sum()
                print(f"{name} - {diff_count}個の値が一致しません")
                
                # 数値型のカラムについて誤差を許容して比較
                numeric_cols = old_df.select_dtypes(include=['number']).columns
                almost_equal = True
                for col in numeric_cols:
                    if col in new_df.columns:
                        # NaN値を考慮した比較
                        old_vals = old_df[col].fillna(np.nan)
                        new_vals = new_df[col].fillna(np.nan)
                        
                        # NaNを同じ扱いにする
                        old_nans = old_vals.isna()
                        new_nans = new_vals.isna()
                        if not (old_nans == new_nans).all():
                            print(f"{col}: NaN値の位置が異なります")
                            almost_equal = False
                            continue
                        
                        # 非NaN値を数値比較
                        old_nums = old_vals[~old_nans].values
                        new_nums = new_vals[~new_nans].values
                        
                        if len(old_nums) != len(new_nums):
                            print(f"{col}: 非NaN値の数が異なります")
                            almost_equal = False
                        elif not np.allclose(old_nums, new_nums, rtol=1e-5, atol=1e-8, equal_nan=True):
                            # 許容誤差を超える場合
                            print(f"{col}: 数値に有意な差があります")
                            # サンプルの差分を表示
                            diff_idx = (~np.isclose(old_nums, new_nums, rtol=1e-5, atol=1e-8)).nonzero()[0]
                            if len(diff_idx) > 0:
                                sample_idx = diff_idx[0]
                                print(f"  例: 旧={old_nums[sample_idx]}, 新={new_nums[sample_idx]}, 差={old_nums[sample_idx]-new_nums[sample_idx]}")
                            almost_equal = False
                
                if almost_equal:
                    print(f"{name} - 許容誤差の範囲内で一致しています")
                else:
                    print(f"{name} - 許容誤差を超える差異があります")
        else:
            print(f"{name} - 完全に一致しています")
        return are_equal
    except Exception as e:
        print(f"{name} - 比較中にエラーが発生しました: {e}")
        return False


def test_features_calculator():
    """リファクタリング前後の特徴量計算を比較するテスト"""
    print("特徴量計算のリファクタリング前後比較テストを開始します...")
    
    # 共通のデータ準備
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    acq = StockAcquisitionFacade(filter=universe_filter)
    stock_dfs = acq.get_stock_data_dict()
    
    # セクター定義ファイルのパスとセクター価格ファイルのパス
    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    NEW_SECTOR_PRICE_PKLGZ = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/sector_price/New48sectors_price.parquet'
    
    # セクター価格の計算（両方のテストで共通利用）
    si = SectorIndex()
    new_sector_price_df, _ = si.calc_sector_index(
        stock_dfs, SECTOR_REDEFINITIONS_CSV, NEW_SECTOR_PRICE_PKLGZ)
    new_sector_list = pd.read_csv(SECTOR_REDEFINITIONS_CSV)


    print("\n1. 銘柄ベースのテスト（セクターなし）")
    # 旧実装の特徴量計算
    old_features_df = OldFeaturesCalculator.calculate_features(
        new_sector_price=new_sector_price_df,
        new_sector_list=new_sector_list,
        stock_dfs_dict=stock_dfs,
        adopts_features_indices=True,
        adopts_features_price=True,
        groups_setting=None,
        names_setting=None,
        currencies_type='relative',
        commodity_type='raw',
        adopt_1d_return=True,
        mom_duration=[5, 21],
        vola_duration=[5, 21],
        adopt_size_factor=True,
        adopt_eps_factor=True,
        adopt_sector_categorical=True,
        add_rank=True
    )


    sector_index_df = pd.read_parquet(f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet')
    financial_df = pd.read_parquet(f'{Paths.SECTOR_FIN_FOLDER}/New48sectors_fin.parquet').reset_index(drop=False)



    # 新実装の特徴量計算
    new_features_df = new_features_calculator_main(
        sector_index_df, financial_df, 
        is_sector=True, code_col='Sector',
        price_yaml_path=Paths.SECTOR_INDEX_COLUMNS_YAML,
        fin_yaml_path=Paths.SECTOR_FIN_COLUMNS_YAML,
        return_duration=[1, 5, 21],
        vola_duration=[5, 21]
    )
    
    # データフレームを比較
    compare_dataframes(old_features_df, new_features_df, "銘柄ベースの特徴量計算")
    
    print(old_features_df[['TOPIX_1d_return', 'USD_1d_return', '5d_mom', '21d_vola']].tail(2))
    print(new_features_df.set_index(['Date', 'Sector'])[['TOPIX_1d_return', 'USD_1d_return', '5d_return', '21d_vola']].tail(2))


if __name__ == "__main__":
    test_features_calculator()