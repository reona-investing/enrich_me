#プログラム開始のライン通知
from utils.notifier import SlackNotifier
import os
Slack = SlackNotifier(program_name=os.path.basename(__file__))
Slack.start(
    message = 'プログラムを開始します。',
    should_send_program_name = True
)

#%% モジュールのインポート
from datetime import datetime
import pandas as pd
from typing import Tuple
from utils.flag_manager import flag_manager, Flags
from utils.paths import Paths
from acquisition import features_scraper as scraper
from calculation import TargetCalculator, FeaturesCalculator, SectorIndexCalculator
from models.dataset import MLDataset
from models.machine_learning import LassoModel, lgbm
import models.ensemble as ensemble
from facades import TradingFacade, StockAcquisitionFacade
from utils.error_handler import error_handler
import asyncio

def load_datasets(ML_DATASET_PATH1: str, ML_DATASET_PATH2: str, ML_DATASET_ENSEMBLED_PATH: str) \
    -> Tuple[MLDataset, MLDataset, MLDataset]:
    if flag_manager.flags[Flags.LEARN]:
        ml_dataset1 = MLDataset(ML_DATASET_PATH1, init_load=False)
        ml_dataset2 = MLDataset(ML_DATASET_PATH2, init_load=False)
        ml_dataset_ensembled = MLDataset(ML_DATASET_ENSEMBLED_PATH, init_load=False)
    else:
        ml_dataset1 = MLDataset(ML_DATASET_PATH1)
        ml_dataset2 = MLDataset(ML_DATASET_PATH2)
        ml_dataset_ensembled = MLDataset(ML_DATASET_ENSEMBLED_PATH)
    return ml_dataset1, ml_dataset2, ml_dataset_ensembled

async def read_and_update_data(filter: str) -> dict:
    stock_dfs_dict = None
    update = process = False
    if flag_manager.flags[Flags.FETCH_DATA]:
        update = process = True
    stock_dfs_dict = StockAcquisitionFacade(update=update, process=process, filter = filter).get_stock_data_dict()
    if flag_manager.flags[Flags.UPDATE_DATASET]:    
        '''各種金融データ取得or読み込み'''
        await scraper.scrape_all_indices(
            should_scrape_features=flag_manager.flags[Flags.FETCH_DATA])
        Slack.send_message(message = 'データの更新が完了しました。')
    return stock_dfs_dict

def get_necessary_dfs(stock_dfs_dict: dict, train_start_day: datetime, train_end_day: datetime, 
                      SECTOR_REDEFINITIONS_CSV: str, SECTOR_INDEX_PARQUET: str) -> dict:
    '''セクターインデックスの計算'''
    new_sector_price_df, order_price_df = \
        SectorIndexCalculator.calc_new_sector_price(stock_dfs_dict, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)
    '''目的変数の算出'''
    raw_target_df, target_df = \
        TargetCalculator.daytime_return_PCAresiduals(new_sector_price_df,
                                                    reduce_components=1, train_start_day=train_start_day, train_end_day=train_end_day)

    return {'new_sector_price_df': new_sector_price_df, 
            'order_price_df': order_price_df, 
            'raw_target_df': raw_target_df,
            'target_df': target_df}

def update_1st_model(ml_dataset: MLDataset, necessary_dfs_dict: dict,
                     train_start_day: datetime, train_end_day: datetime, test_start_day: datetime, test_end_day: datetime) \
                        -> MLDataset:
    if flag_manager.flags[Flags.UPDATE_DATASET]:
        '''LASSO用特徴量の算出'''
        features_df = FeaturesCalculator.calculate_features(necessary_dfs_dict['new_sector_price_df'], None, None,
                                                             adopts_features_indices = True, adopts_features_price = False,
                                                             groups_setting = None, names_setting = None, currencies_type = 'relative',
                                                             adopt_1d_return = True, mom_duration = None, vola_duration = None,
                                                             adopt_size_factor = False, adopt_eps_factor = False,
                                                             adopt_sector_categorical = False, add_rank = False)
        '''LASSO用データセットの更新'''
        ml_dataset.archive_train_test_data(necessary_dfs_dict['target_df'], features_df,
                                           train_start_day, train_end_day, test_start_day, test_end_day,
                                           outlier_threshold = 3,)
        ml_dataset.archive_raw_target(necessary_dfs_dict['raw_target_df'])
        ml_dataset.archive_order_price(necessary_dfs_dict['order_price_df'])
    lasso_model = LassoModel()
    if flag_manager.flags[Flags.UPDATE_MODELS]:
        '''LASSO（学習は必要時、予測は毎回）'''
        trainer_outputs = lasso_model.train(ml_dataset.train_test_materials.target_train_df, ml_dataset.train_test_materials.features_train_df)
        ml_dataset.archive_ml_objects(trainer_outputs.models, trainer_outputs.scalers)
    pred_result_df = lasso_model.predict(ml_dataset.train_test_materials.target_test_df, ml_dataset.train_test_materials.features_test_df, 
                                     ml_dataset.ml_object_materials.models, ml_dataset.ml_object_materials.scalers)
    ml_dataset.archive_pred_result(pred_result_df)
    return ml_dataset

def update_2nd_model(ml_dataset1: MLDataset, ml_dataset2: MLDataset, 
                     stock_dfs_dict: dict, necessary_dfs_dict: dict, 
                     train_start_day: datetime, train_end_day: datetime, test_start_day: datetime, test_end_day: datetime) \
                        -> MLDataset:
    if flag_manager.flags[Flags.UPDATE_DATASET]:
        '''lightGBM用特徴量の算出'''
        features_df = FeaturesCalculator.calculate_features(necessary_dfs_dict['new_sector_price_df'], 
                                                              pd.read_csv(SECTOR_REDEFINITIONS_CSV), stock_dfs_dict,
                                                              adopts_features_indices = True, adopts_features_price = True,
                                                              groups_setting = None, names_setting = None, currencies_type = 'relative',
                                                              adopt_1d_return = True, mom_duration = [5, 21], vola_duration = [5, 21],
                                                              adopt_size_factor = True, adopt_eps_factor = True,
                                                              adopt_sector_categorical = True, add_rank = True)
        pred_in_1st_model = ml_dataset1.evaluation_materials.pred_result_df
        features_df = pd.merge(features_df, pred_in_1st_model[['Pred']], how='outer',
                            left_index=True, right_index=True) # LASSOでの予測結果をlightGBMの特徴量として追加
        features_df = features_df.rename(columns={'Pred':'1stModel_pred'})

        '''lightGBM用データセットの更新'''
        # 目的変数・特徴量dfをデータセットに登録
        ml_dataset2.archive_train_test_data(necessary_dfs_dict['target_df'], features_df,
                                            train_start_day, train_end_day, test_start_day, test_end_day,
                                            outlier_threshold = 3,
                                            no_shift_features=['1stModel_pred'], 
                                            reuse_features_df=True)
        ml_dataset2.archive_raw_target(necessary_dfs_dict['raw_target_df'])
        ml_dataset2.archive_order_price(necessary_dfs_dict['order_price_df'])

    if flag_manager.flags[Flags.UPDATE_MODELS]:
        '''lightGBM（学習は必要時、予測は毎回）'''
        ml_dataset2 = lgbm(ml_dataset = ml_dataset2, learn = flag_manager.flags[Flags.LEARN], categorical_features = ['Sector_cat'])
    return ml_dataset2


def ensemble_pred_results(ensemble_inputs: list[pd.DataFrame, float]) -> pd.DataFrame:
    ensembled_pred_df = ensemble_inputs[0][0][['Target']]
    ensembled_pred_df['Pred'] = ensemble.by_rank(inputs = ensemble_inputs)
    return ensembled_pred_df


def update_ensembled_model(ENSEMBLED_DATASET_PATH: str | os.PathLike[str], ensembled_pred_df: pd.DataFrame, copy_from: MLDataset) -> MLDataset:
    dataset_ensembled = MLDataset(ENSEMBLED_DATASET_PATH)
    dataset_ensembled.copy_from_other_dataset(copy_from)
    dataset_ensembled.archive_pred_result(ensembled_pred_df)
    dataset_ensembled.save()
    return dataset_ensembled


#%% メイン関数
async def main(ML_DATASET_PATH1:str, ML_DATASET_PATH2:str, ML_DATASET_ENSEMBLED_PATH:str,
               SECTOR_REDEFINITIONS_CSV:str, SECTOR_INDEX_PARQUET:str,
               universe_filter:str, trading_sector_num:int, candidate_sector_num:int,
               train_start_day:datetime, train_end_day:datetime,
               test_start_day:datetime, test_end_day:datetime,
               top_slope:float = 1.0, learn:bool = False, predict:bool = None):
    '''
    モデルの実装
    ML_DATASET_PATH: 学習済みモデル、スケーラー、予測結果等を格納したデータセットのパスを格納したリスト
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
    '''
    try:
        '''初期設定'''
        # 最初に各種フラグをセットしておく。データ更新の要否を引数に入力している場合は、フラグをその値で上書き。
        turn_true = []
        if learn:
            turn_true.append(Flags.LEARN)
        if predict:
            turn_true.append(Flags.PREDICT)
        # TODO turn_trueの設定をヘルパー関数として切り出す！
        flag_manager.set_flags(turn_true=turn_true)
        print(flag_manager.get_flags())
        # データセットの読み込み
        if flag_manager.flags[Flags.UPDATE_DATASET] or flag_manager.flags[Flags.UPDATE_MODELS]:
            ml_dataset1, ml_dataset2, ml_dataset_ensembled = load_datasets(ML_DATASET_PATH1, ML_DATASET_PATH2, ML_DATASET_ENSEMBLED_PATH)
            '''データの更新・読み込み'''
            stock_dfs_dict = await read_and_update_data(universe_filter)
            '''学習・予測'''
            ensemble_weights = [6.7, 1.3]
            necessary_dfs_dict = get_necessary_dfs(stock_dfs_dict, train_start_day, train_end_day, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)
            
            ml_dataset1 = update_1st_model(ml_dataset1, necessary_dfs_dict, 
                                           train_start_day, train_end_day, test_start_day, test_end_day)
            ml_dataset2 = update_2nd_model(ml_dataset1, ml_dataset2, stock_dfs_dict, necessary_dfs_dict,
                                           train_start_day, train_end_day, test_start_day, test_end_day)
            pred_result_df1 = ml_dataset1.evaluation_materials.pred_result_df
            pred_result_df2 = ml_dataset2.evaluation_materials.pred_result_df
            ensembled_pred_df = ensemble_pred_results(ensemble_inputs = [(pred_result_df1, ensemble_weights[0]),
                                                                         (pred_result_df2, ensemble_weights[1]),] )
            update_ensembled_model(ENSEMBLED_DATASET_PATH = ML_DATASET_ENSEMBLED_PATH, 
                                   ensembled_pred_df = ensembled_pred_df,
                                   copy_from = ml_dataset1)  
            Slack.send_message(message = f'予測が完了しました。')
        trade_facade = TradingFacade()
        '''新規建'''
        if flag_manager.flags[Flags.TAKE_NEW_POSITIONS]:
            await trade_facade.take_positions(
                ml_dataset= ml_dataset_ensembled,
                SECTOR_REDEFINITIONS_CSV = SECTOR_REDEFINITIONS_CSV,
                num_sectors_to_trade = trading_sector_num,
                num_candidate_sectors = candidate_sector_num,
                top_slope = top_slope)
        '''追加建'''
        if flag_manager.flags[Flags.TAKE_ADDITIONAL_POSITIONS]:
            await trade_facade.take_additionals()
        '''決済注文'''
        if flag_manager.flags[Flags.SETTLE_POSITIONS]:
            await trade_facade.settle_positions()
        '''取引結果の取得'''
        if flag_manager.flags[Flags.FETCH_RESULT]:
            await trade_facade.fetch_invest_result(SECTOR_REDEFINITIONS_CSV)
        Slack.finish(message = 'すべての処理が完了しました。')
    except:
        '''エラーログの出力'''
        error_handler.handle_exception(Paths.ERROR_LOG_CSV)
        Slack.send_error_log(f'エラーが発生しました。\n詳細は{Paths.ERROR_LOG_CSV}を確認してください。')

#%% パラメータ類
if __name__ == '__main__':
    '''パス類'''
    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv' #別でファイルを作っておく
    SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet' #出力のみなのでファイルがなくてもOK
    ML_DATASET_PATH1 = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250125'
    ML_DATASET_PATH2 = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LGBM_learned_in_250125'
    ML_DATASET_EMSEMBLED_PATH = f'{Paths.ML_DATASETS_FOLDER}/48sectors_Ensembled_learned_in_250125'
    '''ユニバースを絞るフィルタ'''
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))" #現行のTOPIX500
    '''上位・下位何業種を取引対象とするか？'''
    trading_sector_num = 3
    candidate_sector_num = 5
    '''トップ予想の業種にどれほどの傾斜をかけるか'''
    top_slope = 1
    '''学習期間'''
    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2024, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31) #ずっと先の未来を指定
    '''学習するか否か'''
    learn = False
    predict = None

#%% 実行
if __name__ == '__main__':
    asyncio.run(main(ML_DATASET_PATH1, ML_DATASET_PATH2, ML_DATASET_EMSEMBLED_PATH, 
                     SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET,
                     universe_filter, trading_sector_num, candidate_sector_num,
                     train_start_day, train_end_day, test_start_day, test_end_day,
                     top_slope, learn, predict))