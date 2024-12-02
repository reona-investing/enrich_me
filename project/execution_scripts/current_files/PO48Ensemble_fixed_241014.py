#%% 事前準備
#パスを通す
if __name__ == '__main__':
    from pathlib import Path
    import sys
    PROJECT_FOLDER = str(Path(__file__).parents[2])
    ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
    sys.path.append(ORIGINAL_MODULES)

#プログラム開始のライン通知
import SlackNotifier
import os
Slack = SlackNotifier.SlackNotifier(program_name=os.path.basename(__file__))
Slack.start(
    message = 'プログラムを開始します。',
    should_send_program_name = True
)

#%% モジュールのインポート
from datetime import datetime
import pandas as pd

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

def set_flags(should_update_historical_data, should_predict):
    now_this_model = FlagManager.launch()
    set_time_flags.set_time_flags(should_update_historical_data=should_update_historical_data)
    if should_predict is None:
        should_predict = now_this_model.should_update_historical_data
    return now_this_model, should_predict

def load_datasets(should_learn, ML_DATASET_PATH1, ML_DATASET_PATH2, ML_DATASET_ENSEMBLED_PATH):
    if should_learn:
        ml_dataset1 = MLDataset.MLDataset()
        ml_dataset2 = MLDataset.MLDataset()
        ml_dataset_ensembled = MLDataset.MLDataset()
    else:
        ml_dataset1 = MLDataset.MLDataset(ML_DATASET_PATH1)
        ml_dataset2 = MLDataset.MLDataset(ML_DATASET_PATH2)
        ml_dataset_ensembled = MLDataset.MLDataset(ML_DATASET_ENSEMBLED_PATH)
    return ml_dataset1, ml_dataset2, ml_dataset_ensembled

async def read_and_update_data(should_learn, should_update_historical_data, universe_filter):
    stock_dfs_dict = None
    if should_update_historical_data:
        '''個別銘柄のデータ更新（取得→成型）'''
        fetcher.update_stock_dfs()
        processor.process_stock_dfs()
    '''個別銘柄のデータ読み込み'''
    stock_dfs_dict = reader.read_stock_dfs(filter = universe_filter)
    if should_learn or should_update_historical_data:    
        '''各種金融データ取得or読み込み'''
        await scraper.scrape_all_indices(should_scrape_features=should_update_historical_data)
        Slack.send_message(message = 'データの更新が完了しました。')
    return stock_dfs_dict

def get_necessary_dfs(stock_dfs_dict, train_start_day, train_end_day, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PARQUET):
    '''セクターインデックスの計算'''
    new_sector_price_df, order_price_df = \
        sector_index_calculator.calc_new_sector_price(stock_dfs_dict, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PARQUET)
    '''目的変数の算出'''
    raw_target_df, target_df = \
        target_calculator.daytime_return_PCAresiduals(new_sector_price_df,
                                                    reduce_components=1, train_start_day=train_start_day, train_end_day=train_end_day)

    return {'new_sector_price_df': new_sector_price_df, 
            'order_price_df': order_price_df, 
            'raw_target_df': raw_target_df,
            'target_df': target_df}

def update_1st_model(ml_dataset, necessary_dfs_dict, 
                     should_learn, should_update_data, should_update_model,
                     train_start_day, train_end_day, test_start_day, test_end_day):
    if should_update_data:
        '''LASSO用特徴量の算出'''
        features_df = features_calculator.calculate_features(necessary_dfs_dict['new_sector_price_df'], None, None,
                                                             adopts_features_indices = True, adopts_features_price = False,
                                                             groups_setting = None, names_setting = None, currencies_type = 'relative',
                                                             adopt_1d_return = True, mom_duration = None, vola_duration = None,
                                                             adopt_size_factor = False, adopt_eps_factor = False,
                                                             adopt_sector_categorical = False, add_rank = False)
        '''LASSO用データセットの更新'''
        ml_dataset.archive_dfs(necessary_dfs_dict['target_df'], features_df, # 目的変数・特徴量dfをデータセットに登録
                                train_start_day, train_end_day, test_start_day, test_end_day,
                                outlier_theshold = 3,
                                raw_target_df=necessary_dfs_dict['raw_target_df'], 
                                order_price_df=necessary_dfs_dict['order_price_df'])
    if should_update_model:
        '''LASSO（学習は必要時、予測は毎回）'''
        ml_dataset = machine_learning.lasso(ml_dataset, dataset_path = ML_DATASET_PATH1, learn = should_learn,
                                            min_features = 3, max_features = 5)
    return ml_dataset

def update_2nd_model(ml_dataset1, ml_dataset2, stock_dfs_dict, necessary_dfs_dict, 
                     should_learn, should_update_data, should_update_model,
                     train_start_day, train_end_day, test_start_day, test_end_day):
    if should_update_data:
        '''lightGBM用特徴量の算出'''
        features_df = features_calculator.calculate_features(necessary_dfs_dict['new_sector_price_df'], 
                                                              pd.read_csv(NEW_SECTOR_LIST_CSV), stock_dfs_dict,
                                                              adopts_features_indices = True, adopts_features_price = True,
                                                              groups_setting = None, names_setting = None, currencies_type = 'relative',
                                                              adopt_1d_return = True, mom_duration = [5, 21], vola_duration = [5, 21],
                                                              adopt_size_factor = True, adopt_eps_factor = True,
                                                              adopt_sector_categorical = True, add_rank = True)
        features_df = pd.merge(features_df, ml_dataset1.pred_result_df[['Pred']], how='outer',
                            left_index=True, right_index=True) # LASSOでの予測結果をlightGBMの特徴量として追加
        features_df = features_df.rename(columns={'Pred':'1stModel_pred'})

        print(necessary_dfs_dict['target_df'])
        print(features_df)

        '''lightGBM用データセットの更新'''
        # 目的変数・特徴量dfをデータセットに登録
        ml_dataset2.archive_dfs(necessary_dfs_dict['target_df'], features_df,
                                train_start_day, train_end_day, test_start_day, test_end_day,
                                outlier_theshold = 3, 
                                raw_target_df=necessary_dfs_dict['raw_target_df'], 
                                order_price_df=necessary_dfs_dict['order_price_df'],
                                no_shift_features=['1stModel_pred'], reuse_features_df_of_others=True)

    if should_update_model:
        '''lightGBM（学習は必要時、予測は毎回）'''
        ml_dataset2 = machine_learning.lgbm(ml_dataset = ml_dataset2, dataset_path = ML_DATASET_PATH2, 
                                            learn = should_learn, categorical_features = ['Sector_cat'])
    return ml_dataset2

def ensemble_pred_results(dataset_ensembled, datasets, ensemble_rates, ENSEMBLED_DATASET_PATH):
    if len(datasets) == 0 or len(datasets) == 0:
        raise ValueError('datasetsとensemble_ratesには1つ以上の要素を指定してください。')
    if len(datasets) != len(ensemble_rates):
        raise ValueError('datasetsとensemble_ratesの要素数を同じにしてください。')
    ensembled_pred_df = datasets[0].pred_result_df[['Target']]
    ensembled_pred_df['Pred'] = machine_learning.ensemble_by_rank(ml_datasets = datasets, 
                                                    ensemble_rates = ensemble_rates)
    dataset_ensembled.archive_raw_target(datasets[0].raw_target_df)
    dataset_ensembled.archive_pred_result(ensembled_pred_df)
    dataset_ensembled.save_instance(ENSEMBLED_DATASET_PATH)
    dataset_ensembled = MLDataset.MLDataset(ENSEMBLED_DATASET_PATH)
    return dataset_ensembled

async def take_positions(order_price_df, NEW_SECTOR_LIST_CSV, pred_result_df, 
                         trading_sector_num, candidate_sector_num,
                         top_slope):
    tab, long_orders, short_orders, todays_pred_df = \
        await order_to_SBI.select_stocks(order_price_df, NEW_SECTOR_LIST_CSV, pred_result_df,
                                        trading_sector_num, candidate_sector_num, 
                                        top_slope=top_slope)
    _, failed_order_list = await order_to_SBI.make_new_order(long_orders, short_orders, tab)
    Slack.send_message(
        message = 
            f'発注が完了しました。\n' +
            f'買： {long_orders["Sector"].unique()}\n' +
            f'売： {short_orders["Sector"].unique()}'
    )
    if len(failed_order_list) > 0:
        Slack.send_message(
            message = 
                f'以下の注文の発注に失敗しました。\n' +
                f'{failed_order_list}'
        )

async def take_additionals():
    _, failed_order_list = await order_to_SBI.make_additional_order()
    Slack.send_message(
        message = 
            f'追加発注が完了しました。'
    )
    if len(failed_order_list) > 0:
        Slack.send_message(
            message = 
                f'以下の注文の発注に失敗しました。\n' +
                f'{failed_order_list}'
        )

async def settle_positions():
    _, error_tickers = await order_to_SBI.settle_all_margins()
    if len(error_tickers) == 0:
        Slack.send_message(message = '全銘柄の決済注文が完了しました。')
    else:
        Slack.send_message(
            message = 
                f'全銘柄の決済注文を試みました。\n' +
                f'銘柄コード{error_tickers}の決済注文に失敗しました。'
                )

async def fetch_invest_result(NEW_SECTOR_LIST_CSV):
    _, trade_history, _, _, _, amount = \
        await order_to_SBI.update_information(NEW_SECTOR_LIST_CSV,
                                              paths.TRADE_HISTORY_CSV, 
                                              paths.BUYING_POWER_HISTORY_CSV,
                                              paths.DEPOSIT_HISTORY_CSV)
    Slack.send_result(
        message = 
            f'取引履歴等の更新が完了しました。\n' +
            f'{trade_history["日付"].iloc[-1].strftime("%Y-%m-%d")}の取引結果：{amount}円'
            )

#%% メイン関数
async def main(ML_DATASET_PATH1:str, ML_DATASET_PATH2:str, ML_DATASET_ENSEMBLED_PATH:str,
               NEW_SECTOR_LIST_CSV:str, NEW_SECTOR_PRICE_PARQUET:str,
               universe_filter:str, trading_sector_num:int, candidate_sector_num:int,
               train_start_day:datetime, train_end_day:datetime,
               test_start_day:datetime, test_end_day:datetime,
               top_slope:float = 1.0, should_learn:bool = True, should_predict:bool = None,
               should_update_historical_data:bool = None):
    '''
    モデルの実装
    ML_DATASET_PATH: 学習済みモデル、スケーラー、予測結果等を格納したデータセットのパスを格納したリスト
    NEW_SECTOR_LIST_CSV: 銘柄と業種の対応リスト
    NEW_SECTOR_PRICE_PARQUET: 業種別の株価インデックス
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
        '''初期設定'''
        # 最初に各種フラグをセットしておく。データ更新の要否を引数に入力している場合は、フラグをその値で上書き。
        now_this_model, should_predict = set_flags(should_update_historical_data, should_predict)
        # データセットの読み込み
        ml_dataset1, ml_dataset2, ml_dataset_ensembled = \
            load_datasets(should_learn, ML_DATASET_PATH1, ML_DATASET_PATH2, ML_DATASET_ENSEMBLED_PATH)
        '''データの更新・読み込み'''
        stock_dfs_dict = await read_and_update_data(should_learn, now_this_model.should_update_historical_data, universe_filter)
        '''学習・予測'''
        should_update_data = should_learn or now_this_model.should_update_historical_data
        should_update_model = should_learn or should_predict or \
            now_this_model.should_take_positions or now_this_model.should_take_additionals
        ensemble_rates = [6.7, 1.3]
        if should_update_data or should_update_model:
            necessary_dfs_dict = get_necessary_dfs(stock_dfs_dict, train_start_day, train_end_day, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PARQUET)
            
            ml_dataset1 = update_1st_model(ml_dataset1, necessary_dfs_dict, 
                                           should_learn, should_update_data, should_update_model,
                                           train_start_day, train_end_day, test_start_day, test_end_day)
            ml_dataset2 = update_2nd_model(ml_dataset1, ml_dataset2, stock_dfs_dict, necessary_dfs_dict, 
                                           should_learn, should_update_data, should_update_model,
                                           train_start_day, train_end_day, test_start_day, test_end_day)
            ml_dataset_ensembled = ensemble_pred_results(dataset_ensembled = ml_dataset_ensembled,
                                                         datasets = [ml_dataset1, ml_dataset2], 
                                                         ensemble_rates = ensemble_rates,
                                                         ENSEMBLED_DATASET_PATH = ML_DATASET_ENSEMBLED_PATH)          
            Slack.send_message(message = f'予測が完了しました。')
        '''新規建'''
        if now_this_model.should_take_positions:
            await take_positions(order_price_df = necessary_dfs_dict['order_price_df'],
                                 NEW_SECTOR_LIST_CSV = NEW_SECTOR_LIST_CSV,
                                 pred_result_df = ml_dataset_ensembled.pred_result_df,
                                 trading_sector_num = trading_sector_num,
                                 candidate_sector_num = candidate_sector_num,
                                 top_slope = top_slope)
        '''追加建'''
        if now_this_model.should_take_additionals:
            await take_additionals()
        '''決済注文'''
        if now_this_model.should_settle_positions:
            await settle_positions()
        '''取引結果の取得'''
        if now_this_model.should_fetch_invest_result:    
            await fetch_invest_result(NEW_SECTOR_LIST_CSV)
        Slack.finish(message = 'すべての処理が完了しました。')
    except:
        '''エラーログの出力'''
        error_handler.handle_exception(paths.ERROR_LOG_CSV)
        Slack.send_error_log(f'エラーが発生しました。\n詳細は{paths.ERROR_LOG_CSV}を確認してください。')

#%% パラメータ類
if __name__ == '__main__':
    '''パス類'''
    NEW_SECTOR_LIST_CSV = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv' #別でファイルを作っておく
    NEW_SECTOR_PRICE_PARQUET = f'{paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet' #出力のみなのでファイルがなくてもOK
    ML_DATASET_PATH1 = f'{paths.ML_DATASETS_FOLDER}/New48sectors'
    ML_DATASET_PATH2 = f'{paths.ML_DATASETS_FOLDER}/LGBM_after_New48sectors'
    ML_DATASET_EMSEMBLED_PATH = f'{paths.ML_DATASETS_FOLDER}/LGBM_New48sectors_Ensembled'
    '''ユニバースを絞るフィルタ'''
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))" #現行のTOPIX500
    '''上位・下位何業種を取引対象とするか？'''
    trading_sector_num = 3
    candidate_sector_num = 5
    '''トップ予想の業種にどれほどの傾斜をかけるか'''
    top_slope = 1
    '''学習期間'''
    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2021, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31) #ずっと先の未来を指定
    '''学習するか否か'''
    should_learn = False
    should_predict = None
    should_update_historical_data = None

#%% 実行
if __name__ == '__main__':
    asyncio.run(main(ML_DATASET_PATH1, ML_DATASET_PATH2, ML_DATASET_EMSEMBLED_PATH, 
                     NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PARQUET,
                     universe_filter, trading_sector_num, candidate_sector_num,
                     train_start_day, train_end_day, test_start_day, test_end_day,
                     top_slope, should_learn, should_predict, should_update_historical_data))