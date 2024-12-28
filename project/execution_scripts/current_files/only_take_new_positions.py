#%% 事前準備
#パスを通す
if __name__ == '__main__':
    from pathlib import Path
    import sys
    PROJECT_FOLDER = str(Path(__file__).parents[2])
    ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
    sys.path.append(ORIGINAL_MODULES)

#プログラム開始のライン通知
import LineNortifier
import os
LINE = LineNortifier.LINEnotifier(program_name=os.path.basename(__file__))
LINE.start(
    message = 'プログラムを開始します。',
    should_send_program_name = True
)

#%% モジュールのインポート
from datetime import datetime
import pandas as pd

import paths #パス一覧
import stock_dfs_processor as processor #取得したデータの加工
import stock_dfs_reader as reader #加工したデータの読み込み
import sector_index_calculator
import FlagManager
import set_time_flags #フラグ管理
import features_scraper as scraper
import features_calculator
import target_calculator
import MLDataset
import machine_learning
import sbi_trading_logic
import error_handler
import asyncio


#%% メイン関数
async def main(ML_DATASET_PATH:str, 
               NEW_SECTOR_LIST_CSV:str,
               trading_sector_num:int,
               candidate_sector_num:int,
               top_slope:float = 1.0, DATASET_FOR_ORDER_PATH:str = None
               ):
    '''
    モデルの実装
    ML_DATASET_PATH: 学習済みモデル、スケーラー、予測結果等を格納したデータセットのパスを格納したリスト
    NEW_SECTOR_LIST_CSV: 銘柄と業種の対応リスト
    NEW_SECTOR_PRICE_PKLGZ: 業種別の株価インデックス
    universe_filter: ユニバースを絞るフィルタ
    trading_sector_num: 上位・下位何業種を取引対象とするか
    candidate_sector_num: 取引できない業種がある場合、上位・下位何業種を取引対象候補とするか
    train_start_day: 学習期間の開始日
    train_end_day: 学習期間の終了日
    test_start_day: テスト期間の開始日
    test_end_day: テスト期間の終了日
    top_slope: トップ予想の業種にどれほどの傾斜をかけるか
    should_learn: 学習するか否か
    '''
    try:
        '''1. 初期設定'''
        # ml_datasetは必ず生成するので、最初に生成してしまう。
        ml_dataset = MLDataset.MLDataset(ML_DATASET_PATH)
        if DATASET_FOR_ORDER_PATH is not None:
            dataset_for_order = MLDataset.MLDataset(DATASET_FOR_ORDER_PATH)
            ml_dataset.order_price_df = dataset_for_order.order_price_df
        print(ml_dataset.pred_result_df)
        
        '''10. 新規注文'''
        _, long_orders, short_orders, todays_pred_df = \
            await sbi_trading_logic.select_stocks(ml_dataset.order_price_df, NEW_SECTOR_LIST_CSV, ml_dataset.pred_result_df,
                                            trading_sector_num, candidate_sector_num, 
                                            top_slope=top_slope)
        _, take_position, failed_order_list = await sbi_trading_logic.make_new_order(long_orders, short_orders)
        LINE.send_message(
            message = 
                f'発注が完了しました。\n' +
                f'買： {long_orders["Sector"].unique()}\n' +
                f'売： {short_orders["Sector"].unique()}'
        )
        if len(failed_order_list) > 0:
            LINE.send_message(
                message = 
                    f'以下の注文の発注に失敗しました。\n' +
                    f'{failed_order_list}'
            )

    except:
        '''エラーログの出力'''
        error_log_path = f'{paths.DEBUG_FILES_FOLDER}/error_log.csv'
        error_handler.handle_exception(error_log_path)
        LINE.send_message(f'エラーが発生しました。\n詳細は{error_log_path}を確認してください。')

#%% パラメータ類
if __name__ == '__main__':
    '''パス類'''
    NEW_SECTOR_LIST_CSV = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_list.csv' #別でファイルを作っておく
    ML_DATASET_PATH = f'{paths.ML_DATASETS_FOLDER}/LGBM_New48sectors_Ensembled'
    ORDER_PATH =f'{paths.ML_DATASETS_FOLDER}/New48sectors'
    '''上位・下位何業種を取引対象とするか？'''
    trading_sector_num = 3
    candidate_sector_num = 5
    '''トップ予想の業種にどれほどの傾斜をかけるか'''
    top_slope = 1.0

#%% 実行
if __name__ == '__main__':
    asyncio.run(main(ML_DATASET_PATH, 
                     NEW_SECTOR_LIST_CSV,
                     trading_sector_num, 
                     candidate_sector_num,
                     top_slope,
                     DATASET_FOR_ORDER_PATH=ORDER_PATH))

'''
同じモデルを使っても。こちらで実行すると、モデルの性能が落ちた。
なぜ？学習時に使用したデータが不適切だった？要検討
'''