#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
リファクタリング後のコードを使用した実行スクリプト
元のPO48Ensemble_fixed_241014.pyと同等の機能を実装
"""

import os
import pandas as pd
import asyncio
from datetime import datetime
from typing import Dict, Any

# ユーティリティモジュール
from utils.notifier import SlackNotifier
from utils.flag_manager import flag_manager, Flags
from utils.paths import Paths
from utils.error_handler import error_handler

# データ取得・計算モジュール
from calculation import TargetCalculator, FeaturesCalculator, SectorIndexCalculator
from trading import TradingFacade
from acquisition.jquants_api_operations import StockAcquisitionFacade
from acquisition.features_updater import FeaturesUpdateFacade

# リファクタリング後の新しいモデルモジュール
from machine_learning.strategies import SectorLassoStrategy, SingleLgbmStrategy, EnsembleStrategy
from machine_learning.core.collection import ModelCollection


# Slackの初期化
def init_slack():
    """Slack通知の初期化"""
    slack = SlackNotifier(program_name=os.path.basename(__file__))
    slack.start(
        message='プログラムを開始します。',
        should_send_program_name=True
    )
    return slack


async def read_and_update_data(filter: str) -> dict:
    """
    株価データと金融データを取得
    
    Args:
        filter: ユニバースを絞るフィルタ
        
    Returns:
        株価データの辞書
    """
    stock_dfs_dict = None
    update = process = False
    if flag_manager.flags[Flags.FETCH_DATA]:
        update = process = True
    
    stock_dfs_dict = StockAcquisitionFacade(update=update, process=process, filter=filter).get_stock_data_dict()
    
    if flag_manager.flags[Flags.UPDATE_DATASET] or flag_manager.flags[Flags.FETCH_DATA]:
        '''各種金融データ取得or読み込み'''
        fu = FeaturesUpdateFacade()
        await fu.update_all()
    
    return stock_dfs_dict


def get_necessary_dfs(stock_dfs_dict: dict, train_start_day: datetime, train_end_day: datetime, 
                     SECTOR_REDEFINITIONS_CSV: str, SECTOR_INDEX_PARQUET: str) -> dict:
    """
    必要なデータフレームを準備
    
    Args:
        stock_dfs_dict: 株価データの辞書
        train_start_day: 学習期間の開始日
        train_end_day: 学習期間の終了日
        SECTOR_REDEFINITIONS_CSV: 銘柄と業種の対応リスト
        SECTOR_INDEX_PARQUET: 業種別の株価インデックス
        
    Returns:
        必要なデータフレームの辞書
    """
    # セクターインデックスの計算
    new_sector_price_df, order_price_df = SectorIndexCalculator.calc_sector_index(
        stock_dfs_dict, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET
    )
    
    # 目的変数の算出
    raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(
        new_sector_price_df, reduce_components=1, 
        train_start_day=train_start_day, train_end_day=train_end_day
    )

    return {
        'new_sector_price_df': new_sector_price_df, 
        'order_price_df': order_price_df, 
        'raw_target_df': raw_target_df,
        'target_df': target_df
    }


def update_1st_model(necessary_dfs_dict: Dict[str, pd.DataFrame],
                    ML_DATASET_PATH1: str,
                    train_start_day: datetime, train_end_day: datetime, 
                    test_start_day: datetime, test_end_day: datetime,
                    learn: bool) -> None:
    """
    1番目のモデル（LASSO）の更新
    
    Args:
        necessary_dfs_dict: 必要なデータフレームの辞書
        ML_DATASET_PATH1: LASSOモデルの保存先パス
        train_start_day: 学習期間の開始日
        train_end_day: 学習期間の終了日
        test_start_day: テスト期間の開始日
        test_end_day: テスト期間の終了日
        learn: 学習するか否か
    """
    if flag_manager.flags[Flags.UPDATE_DATASET]:
        # LASSO用特徴量の算出
        features_df = FeaturesCalculator.calculate_features(
            necessary_dfs_dict['new_sector_price_df'], None, None,
            adopts_features_indices=True, adopts_features_price=False,
            groups_setting=None, names_setting=None, currencies_type='relative',
            adopt_1d_return=True, mom_duration=None, vola_duration=None,
            adopt_size_factor=False, adopt_eps_factor=False,
            adopt_sector_categorical=False, add_rank=False
        )
        
        # セクター別LASSOモデル戦略を実行
        SectorLassoStrategy.run(
            path=ML_DATASET_PATH1,
            target_df=necessary_dfs_dict['target_df'],
            features_df=features_df,
            raw_target_df=necessary_dfs_dict['raw_target_df'],
            order_price_df=necessary_dfs_dict['order_price_df'],
            train_start_date=train_start_day,
            train_end_date=train_end_day,
            test_start_date=test_start_day,
            test_end_date=test_end_day,
            train=learn,
            outlier_threshold=3.0
        )


def update_2nd_model(necessary_dfs_dict: Dict[str, pd.DataFrame],
                    ML_DATASET_PATH1: str,
                    ML_DATASET_PATH2: str,
                    stock_dfs_dict: dict,
                    SECTOR_REDEFINITIONS_CSV: str,
                    train_start_day: datetime, train_end_day: datetime, 
                    test_start_day: datetime, test_end_day: datetime,
                    learn: bool) -> None:
    """
    2番目のモデル（LightGBM）の更新
    
    Args:
        necessary_dfs_dict: 必要なデータフレームの辞書
        ML_DATASET_PATH1: LASSOモデルの保存先パス
        ML_DATASET_PATH2: LightGBMモデルの保存先パス
        stock_dfs_dict: 株価データの辞書
        SECTOR_REDEFINITIONS_CSV: 銘柄と業種の対応リスト
        train_start_day: 学習期間の開始日
        train_end_day: 学習期間の終了日
        test_start_day: テスト期間の開始日
        test_end_day: テスト期間の終了日
        learn: 学習するか否か
    """
    if flag_manager.flags[Flags.UPDATE_DATASET]:
        # LightGBM用特徴量の算出
        features_df = FeaturesCalculator.calculate_features(
            necessary_dfs_dict['new_sector_price_df'], 
            pd.read_csv(SECTOR_REDEFINITIONS_CSV), stock_dfs_dict,
            adopts_features_indices=True, adopts_features_price=True,
            groups_setting=None, names_setting=None, currencies_type='relative',
            adopt_1d_return=True, mom_duration=[5, 21], vola_duration=[5, 21],
            adopt_size_factor=True, adopt_eps_factor=True,
            adopt_sector_categorical=True, add_rank=True
        )
        
        # LASSOモデルの予測結果を取得し、特徴量として追加
        lasso_collection = ModelCollection.load(ML_DATASET_PATH1)
        pred_in_1st_model = lasso_collection.get_result_df()
        
        features_df = pd.merge(
            features_df, pred_in_1st_model[['Pred']], 
            how='outer', left_index=True, right_index=True
        )
        features_df = features_df.rename(columns={'Pred': '1stModel_pred'})
        
        # 単一LightGBMモデル戦略を実行
        SingleLgbmStrategy.run(
            path=ML_DATASET_PATH2,
            target_df=necessary_dfs_dict['target_df'],
            features_df=features_df,
            raw_target_df=necessary_dfs_dict['raw_target_df'],
            order_price_df=necessary_dfs_dict['order_price_df'],
            train_start_date=train_start_day,
            train_end_date=train_end_day,
            test_start_date=test_start_day,
            test_end_date=test_end_day,
            train=learn,
            outlier_threshold=3.0,
            no_shift_features=['1stModel_pred'],
            reuse_features_df=True
        )


def update_ensemble_model(ML_DATASET_PATH1: str,
                         ML_DATASET_PATH2: str,
                         ML_DATASET_ENSEMBLED_PATH: str,
                         ensemble_weights: list) -> None:
    """
    アンサンブルモデルの更新
    
    Args:
        ML_DATASET_PATH1: LASSOモデルの保存先パス
        ML_DATASET_PATH2: LightGBMモデルの保存先パス
        ML_DATASET_ENSEMBLED_PATH: アンサンブルモデルの保存先パス
        ensemble_weights: アンサンブルの重み
    """
    # アンサンブル戦略を実行
    EnsembleStrategy.run(
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


async def main(ML_DATASET_PATH1: str, ML_DATASET_PATH2: str, ML_DATASET_ENSEMBLED_PATH: str,
               SECTOR_REDEFINITIONS_CSV: str, SECTOR_INDEX_PARQUET: str,
               universe_filter: str, trading_sector_num: int, candidate_sector_num: int,
               train_start_day: datetime, train_end_day: datetime,
               test_start_day: datetime, test_end_day: datetime,
               top_slope: float = 1.0, learn: bool = False, predict: bool = None):
    """
    メイン関数
    
    Args:
        ML_DATASET_PATH1: 1番目のモデル（LASSO）の保存先パス
        ML_DATASET_PATH2: 2番目のモデル（LightGBM）の保存先パス
        ML_DATASET_ENSEMBLED_PATH: アンサンブルモデルの保存先パス
        SECTOR_REDEFINITIONS_CSV: 銘柄と業種の対応リスト
        SECTOR_INDEX_PARQUET: 業種別の株価インデックス
        universe_filter: ユニバースを絞るフィルタ
        trading_sector_num: 上位・下位何業種を取引対象とするか
        candidate_sector_num: 取引できない業種がある場合、上位・下位何業種を取引対象候補とするか
        train_start_day: 学習期間の開始日
        train_end_day: 学習期間の終了日
        test_start_day: テスト期間の開始日
        test_end_day: テスト期間の終了日
        top_slope: トップ予想の業種にどれほどの傾斜をかけるか
        learn: 学習するか否か
        predict: 予測するか否か
    """
    try:
        # Slack通知の初期化
        slack = init_slack()
        
        # 初期設定
        # 最初に各種フラグをセットしておく。データ更新の要否を引数に入力している場合は、フラグをその値で上書き。
        turn_true = []
        if learn:
            turn_true.append(Flags.LEARN)
        if predict:
            turn_true.append(Flags.PREDICT)
        flag_manager.set_flags(turn_true=turn_true)
        print(flag_manager.get_flags())
        
        # データの更新・読み込み
        if flag_manager.flags[Flags.UPDATE_DATASET] or flag_manager.flags[Flags.UPDATE_MODELS]:
            # 株価データの取得
            stock_dfs_dict = await read_and_update_data(universe_filter)
            slack.send_message(message='データの更新が完了しました。')

            # 学習・予測
            ensemble_weights = [6.7, 1.3]
            
            # 必要なデータフレームの取得
            necessary_dfs_dict = get_necessary_dfs(
                stock_dfs_dict, train_start_day, train_end_day, 
                SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET
            )
            
            # 1番目のモデル（LASSO）の更新
            update_1st_model(
                necessary_dfs_dict, ML_DATASET_PATH1,
                train_start_day, train_end_day, test_start_day, test_end_day,
                learn=flag_manager.flags[Flags.LEARN]
            )
            
            # 2番目のモデル（LightGBM）の更新
            update_2nd_model(
                necessary_dfs_dict, ML_DATASET_PATH1, ML_DATASET_PATH2,
                stock_dfs_dict, SECTOR_REDEFINITIONS_CSV,
                train_start_day, train_end_day, test_start_day, test_end_day,
                learn=flag_manager.flags[Flags.LEARN]
            )
            
            # アンサンブルモデルの更新
            update_ensemble_model(
                ML_DATASET_PATH1, ML_DATASET_PATH2, ML_DATASET_ENSEMBLED_PATH,
                ensemble_weights
            )
            
            slack.send_message(message='予測が完了しました。')
        
        # 取引戦略の実行
        trade_facade = TradingFacade()
        
        # 新規建
        if flag_manager.flags[Flags.TAKE_NEW_POSITIONS]:
            await trade_facade.take_positions(
                ml_dataset_path=ML_DATASET_ENSEMBLED_PATH,  # 修正: ModelCollectionを直接渡すのではなくパスを渡す
                SECTOR_REDEFINITIONS_CSV=SECTOR_REDEFINITIONS_CSV,
                num_sectors_to_trade=trading_sector_num,
                num_candidate_sectors=candidate_sector_num,
                top_slope=top_slope
            )
        
        # 追加建
        if flag_manager.flags[Flags.TAKE_ADDITIONAL_POSITIONS]:
            await trade_facade.take_additionals()
        
        # 決済注文
        if flag_manager.flags[Flags.SETTLE_POSITIONS]:
            await trade_facade.settle_positions()
        
        # 取引結果の取得
        if flag_manager.flags[Flags.FETCH_RESULT]:
            await trade_facade.fetch_invest_result(SECTOR_REDEFINITIONS_CSV)
        
        slack.finish(message='すべての処理が完了しました。')
        
    except Exception as e:
        # エラーログの出力
        error_handler.handle_exception(Paths.ERROR_LOG_CSV)
        error_handler.handle_exception(Paths.ERROR_LOG_BACKUP)
        
        # Slack通知
        try:
            slack.send_error_log(f'エラーが発生しました。\n詳細は{Paths.ERROR_LOG_CSV}を確認してください。')
        except:
            print(f"エラー発生: {e}")
        
        # 例外を再送出
        raise


# メイン処理
if __name__ == '__main__':
    # パラメータ類
    # パス類
    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet'
    ML_DATASET_PATH1 = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250413'
    ML_DATASET_PATH2 = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LGBM_learned_in_250413'
    ML_DATASET_EMSEMBLED_PATH = f'{Paths.ML_DATASETS_FOLDER}/48sectors_Ensembled_learned_in_250413'
    
    # ユニバースを絞るフィルタ
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    
    # 上位・下位何業種を取引対象とするか？
    trading_sector_num = 3
    candidate_sector_num = 5
    
    # トップ予想の業種にどれほどの傾斜をかけるか
    top_slope = 1
    
    # 学習期間
    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2021, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31)  # ずっと先の未来を指定
    
    # 学習するか否か
    learn = True
    predict = True
    
    # 実行
    asyncio.get_event_loop().run_until_complete(
        main(
            ML_DATASET_PATH1, ML_DATASET_PATH2, ML_DATASET_EMSEMBLED_PATH,
            SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET,
            universe_filter, trading_sector_num, candidate_sector_num,
            train_start_day, train_end_day, test_start_day, test_end_day,
            top_slope, learn, predict
        )
    )