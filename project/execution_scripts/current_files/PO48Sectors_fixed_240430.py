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

import paths #パス一覧
import jquants_api_fetcher as fetcher #JQuantsAPIでのデータ取得
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
import order_to_SBI
import error_handler
import asyncio


#%% メイン関数
async def main(ML_DATASET_PATH:str, NEW_SECTOR_LIST_CSV:str, NEW_SECTOR_PRICE_PKLGZ:str,
         universe_filter:str, trading_sector_num:int, candidate_sector_num:int,
         train_start_day:datetime, train_end_day:datetime,
         test_start_day:datetime, test_end_day:datetime,
         top_slope:float = 1.0, should_learn:bool = True,
         should_update_historical_data:bool = None):
    '''
    モデルの実装
    ML_DATASET_PATH: 学習済みモデル、スケーラー、予測結果等を格納したデータセット
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
        # 最初に各種フラグをセットしておく
        # データ更新の要否を引数に入力している場合は、フラグをその値で上書き
        now_this_model = FlagManager.launch()
        set_time_flags.set_time_flags(should_update_historical_data=should_update_historical_data)
        # ml_datasetは必ず生成するので、最初に生成してしまう。
        if should_learn:
            ml_dataset = MLDataset.MLDataset()
        else:
            ml_dataset = MLDataset.MLDataset(ML_DATASET_PATH)

        if now_this_model.should_update_historical_data:
            '''2. 個別銘柄のデータ更新（取得→成型）'''
            fetcher.update_stock_dfs()
            processor.process_stock_dfs()

        if should_learn or now_this_model.should_update_historical_data:
            '''3. 個別銘柄のデータ読み込み'''
            stock_dfs_dict = reader.read_stock_dfs(filter = universe_filter)
            '''4. セクターインデックスの計算'''
            new_sector_price_df, order_price_df = \
                sector_index_calculator.calc_new_sector_price(stock_dfs_dict, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PKLGZ)
            '''5. 目的変数の算出'''
            raw_target_df, target_df = \
                target_calculator.daytime_return_PCAresiduals(new_sector_price_df,
                                                            reduce_components=1, train_start_day=train_start_day, train_end_day=train_end_day)

            '''6. 各種金融データ取得or読み込み'''
            await scraper.scrape_all_indices(should_scrape_features=now_this_model.should_update_historical_data)
            '''7. 特徴量の算出'''
            features_df = features_calculator.calculate_features(new_sector_price_df, None, None,
                                                                adopts_features_indices = True, adopts_features_price = False,
                                                                groups_setting = None, names_setting = None, currencies_type = 'relative',
                                                                )
            '''8. 機械学習用データセットの更新'''
            # 目的変数・特徴量dfをデータセットに登録
            ml_dataset.archive_dfs(target_df, features_df,
                                   train_start_day, train_end_day, test_start_day, test_end_day,
                                   outlier_theshold = 3,
                                   raw_target_df=raw_target_df, order_price_df=order_price_df)
            LINE.send_message(
                message = 'データ更新と目的変数・特徴量の計算が完了しました。'
            )

        '''9. 機械学習（学習は必要時、予測は毎回）'''
        ml_dataset = machine_learning.lasso(ml_dataset, dataset_path = ML_DATASET_PATH, learn = should_learn,
                                            min_features = 3, max_features = 5)

        '''10. 新規注文'''
        if now_this_model.should_take_positions:
            _, long_orders, short_orders, todays_pred_df = \
                await order_to_SBI.select_stocks(ml_dataset.order_price_df, NEW_SECTOR_LIST_CSV, ml_dataset.pred_result_df,
                                                trading_sector_num, candidate_sector_num, 
                                                top_slope=top_slope)
            _, take_position, failed_order_list = await order_to_SBI.make_new_order(long_orders, short_orders)
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
        
        '''11. 決済注文'''
        if now_this_model.should_settle_positions:
            _, error_tickers = await order_to_SBI.settle_all_margins()
            if len(error_tickers) == 0:
                LINE.send_message(message = '全銘柄の決済注文が完了しました。')
            else:
                LINE.send_message(
                    message = 
                        f'全銘柄の決済注文を試みました。\n' +
                        f'銘柄コード{error_tickers}の決済注文に失敗しました。'
                        )

        '''12. 取引結果の取得'''
        if now_this_model.should_fetch_invest_result:
            _, trade_history, buying_power_history, deposit_history, return_rate, amount = \
                await order_to_SBI.update_information(NEW_SECTOR_LIST_CSV, 
                                                    paths.TRADE_HISTORY_CSV, 
                                                    paths.BUYING_POWER_HISTORY_CSV, 
                                                    paths.DEPOSIT_HISTORY_CSV)
            LINE.send_message(
                message = 
                    f'取引履歴等の更新が完了しました。\n' +
                    f'{trade_history["日付"].iloc[-1].strftime("%Y-%m-%d")}の取引結果：{amount}円'
                    )
        else:
            '''13. データの読み込み'''
            trade_history, buying_power_history, deposit_history, return_rate, amount = \
                order_to_SBI.load_information(paths.TRADE_HISTORY_CSV, 
                                                    paths.BUYING_POWER_HISTORY_CSV, 
                                                    paths.DEPOSIT_HISTORY_CSV)

        LINE.finish(message = 'すべての処理が完了しました。')

        return ml_dataset
    
    except:
        '''エラーログの出力'''
        error_log_path = f'{paths.DEBUG_FILES_FOLDER}/error_log.csv'
        error_handler.handle_exception(error_log_path)
        LINE.send_message(f'エラーが発生しました。\n詳細は{error_log_path}を確認してください。')

#%% パラメータ類
if __name__ == '__main__':
    '''パス類'''
    NEW_SECTOR_LIST_CSV = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_list.csv' #別でファイルを作っておく
    NEW_SECTOR_PRICE_PKLGZ = f'{paths.SECTOR_PRICE_FOLDER}/New48sectors_price' #出力のみなのでファイルがなくてもOK
    ML_DATASET_PATH = f'{paths.ML_DATASETS_FOLDER}/New48sectors.parquet'
    '''ユニバースを絞るフィルタ'''
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))" #現行のTOPIX500
    '''上位・下位何業種を取引対象とするか？'''
    trading_sector_num = 3
    candidate_sector_num = 5
    '''トップ予想の業種にどれほどの傾斜をかけるか'''
    top_slope = 1.5
    '''学習期間'''
    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2021, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31) #ずっと先の未来を指定
    '''学習するか否か'''
    should_learn = False
    should_update_historical_data = None

#%% 実行
if __name__ == '__main__':
    asyncio.run(main(ML_DATASET_PATH, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PKLGZ,
                universe_filter, trading_sector_num, candidate_sector_num,
                train_start_day, train_end_day, test_start_day, test_end_day,
                top_slope, should_learn, should_update_historical_data))