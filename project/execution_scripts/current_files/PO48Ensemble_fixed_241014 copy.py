#プログラム開始のライン通知
from utils.notifier import SlackNotifier
import os
import numpy as np
Slack = SlackNotifier(program_name=os.path.basename(__file__))
Slack.start(
    message = 'プログラムを開始します。',
    should_send_program_name = True
)

#%% モジュールのインポート
from datetime import datetime
import pandas as pd
from utils.flag_manager import flag_manager, Flags
from utils.paths import Paths
from acquisition.features_updater import FeaturesUpdater
from calculation import TargetCalculator, FeaturesCalculator, SectorIndexCalculator
from utils.error_handler import error_handler
import asyncio

# models2関連のインポート
from models2 import LassoModel, LgbmModel, EnsembleModel, ModelFactory 
from models2.datasets import DatasetManager, ModelDatasetConnector
from models2.base import LassoParams, LgbmParams
from models2.base.base_container import BaseContainer
from models2.base.base_model import BaseModel

# プログラムの先頭、インポート部分の後に追加

# ダミーの予測モデルクラス - モジュールレベルで定義する
class PredictionModel(BaseModel):
    """
    既存の予測結果を利用するダミーモデル
    """
    def __init__(self, predictions: pd.DataFrame = None):
        self.predictions = predictions
        self._feature_importances_df = None
    
    def train(self, X, y, **kwargs):
        return self
    
    def predict(self, X):
        if self.predictions is None:
            return np.zeros(len(X))
        return self.predictions['Pred'].values
    
    @property
    def feature_importances(self):
        return self._feature_importances_df
    
    # pickleのサポート
    def __getstate__(self):
        return {'predictions': self.predictions}
    
    def __setstate__(self, state):
        self.predictions = state['predictions']
        self._feature_importances_df = None

async def read_and_update_data(filter: str) -> dict:
    stock_dfs_dict = None
    update = process = False
    if flag_manager.flags[Flags.FETCH_DATA]:
        update = process = True
    stock_dfs_dict = StockAcquisitionFacade(update=update, process=process, filter = filter).get_stock_data_dict()
    if flag_manager.flags[Flags.UPDATE_DATASET] or flag_manager.flags[Flags.FETCH_DATA]:    
        '''各種金融データ取得or読み込み'''
        fu = FeaturesUpdater()
        await fu.update_all()
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

def update_1st_model(dataset_manager: DatasetManager, necessary_dfs_dict: dict,
                     train_start_day: datetime, train_end_day: datetime, test_start_day: datetime, test_end_day: datetime,
                     force_learn: bool = False) \
                        -> tuple[DatasetManager, pd.DataFrame]:
    if flag_manager.flags[Flags.UPDATE_DATASET]:
        '''LASSO用特徴量の算出'''
        features_df = FeaturesCalculator.calculate_features(necessary_dfs_dict['new_sector_price_df'], None, None,
                                                             adopts_features_indices = True, adopts_features_price = False,
                                                             groups_setting = None, names_setting = None, currencies_type = 'relative',
                                                             adopt_1d_return = True, mom_duration = None, vola_duration = None,
                                                             adopt_size_factor = False, adopt_eps_factor = False,
                                                             adopt_sector_categorical = False, add_rank = False)
        '''LASSO用データセットの更新'''
        dataset_manager.prepare_dataset(necessary_dfs_dict['target_df'], features_df,
                                        train_start_day, train_end_day, test_start_day, test_end_day,
                                        outlier_threshold = 3)
        dataset_manager.archive_raw_target(necessary_dfs_dict['raw_target_df'])
        dataset_manager.archive_order_price(necessary_dfs_dict['order_price_df'])
    
    # セクター一覧を取得してLassoモデルコンテナを作成
    sectors = dataset_manager.get_sectors()
    lasso_container = ModelFactory.create_lasso_container(sectors)
    
    # 学習フラグに基づいてモデルを学習または読み込み
    should_learn = flag_manager.flags[Flags.LEARN] or force_learn
    
    # 学習用パラメータの設定
    lasso_params = LassoParams(max_features=5, min_features=3)
    
    # 学習または読み込み
    ModelDatasetConnector.train_models(
        dataset_manager, 
        lasso_container, 
        params={"params": lasso_params},
        force_learn=should_learn
    )
    
    # 予測の実行
    pred_result_df = ModelDatasetConnector.predict_with_models(dataset_manager, lasso_container)
    
    # 予測結果を保存
    dataset_manager.set_pred_result(pred_result_df)
    dataset_manager.save()
    
    return dataset_manager, pred_result_df


def update_2nd_model(dataset_lasso: DatasetManager, dataset_lgbm: DatasetManager, 
                     stock_dfs_dict: dict, necessary_dfs_dict: dict, pred_in_1st_model: pd.DataFrame,
                     train_start_day: datetime, train_end_day: datetime, test_start_day: datetime, test_end_day: datetime,
                     SECTOR_REDEFINITIONS_CSV: str, force_learn: bool = False) \
                        -> tuple[DatasetManager, pd.DataFrame]:
    if flag_manager.flags[Flags.UPDATE_DATASET]:
        '''lightGBM用特徴量の算出'''
        features_df = FeaturesCalculator.calculate_features(necessary_dfs_dict['new_sector_price_df'], 
                                                              pd.read_csv(SECTOR_REDEFINITIONS_CSV), stock_dfs_dict,
                                                              adopts_features_indices = True, adopts_features_price = True,
                                                              groups_setting = None, names_setting = None, currencies_type = 'relative',
                                                              adopt_1d_return = True, mom_duration = [5, 21], vola_duration = [5, 21],
                                                              adopt_size_factor = True, adopt_eps_factor = True,
                                                              adopt_sector_categorical = True, add_rank = True)
        
        # LASSOでの予測結果をlightGBMの特徴量として追加
        features_df = pd.merge(features_df, pred_in_1st_model[['Pred']], how='outer',
                            left_index=True, right_index=True)
        features_df = features_df.rename(columns={'Pred':'1stModel_pred'})

        '''lightGBM用データセットの更新'''
        # 目的変数・特徴量dfをデータセットに登録
        dataset_lgbm.prepare_dataset(necessary_dfs_dict['target_df'], features_df,
                                    train_start_day, train_end_day, test_start_day, test_end_day,
                                    outlier_threshold = 3,
                                    no_shift_features=['1stModel_pred'], 
                                    reuse_features_df=True)
        dataset_lgbm.archive_raw_target(necessary_dfs_dict['raw_target_df'])
        dataset_lgbm.archive_order_price(necessary_dfs_dict['order_price_df'])

    # セクター一覧を取得してLGBMモデルコンテナを作成
    sectors = dataset_lgbm.get_sectors()
    lgbm_container = ModelFactory.create_from_existing({sector: LgbmModel() for sector in sectors}, "LGBM_MultiSector")
    
    # 学習フラグに基づいてモデルを学習または読み込み
    should_learn = flag_manager.flags[Flags.LEARN] or force_learn
    
    # 学習用パラメータの設定
    lgbm_params = LgbmParams(categorical_features=['Sector_cat'])
    
    # 学習または読み込み
    ModelDatasetConnector.train_models(
        dataset_lgbm, 
        lgbm_container, 
        params={"params": lgbm_params},
        force_learn=should_learn
    )
    
    # 予測の実行
    pred_result_df = ModelDatasetConnector.predict_with_models(dataset_lgbm, lgbm_container)
    
    # 予測結果を保存
    dataset_lgbm.set_pred_result(pred_result_df)
    dataset_lgbm.save()
    
    return dataset_lgbm, pred_result_df

def ensemble_pred_results(pred_result_df1: pd.DataFrame, pred_result_df2: pd.DataFrame, 
                          weights: list[float]) -> tuple[pd.DataFrame, EnsembleModel]:
    """
    複数モデルの予測結果をアンサンブル
    
    Args:
        pred_result_df1: 1つ目のモデルの予測結果
        pred_result_df2: 2つ目のモデルの予測結果
        weights: 各モデルの重み
        
    Returns:
        tuple: (アンサンブル予測結果のDataFrame, アンサンブルモデル)
    """
    # アンサンブルモデルを作成
    ensemble_model = EnsembleModel(name="Sector_Ensemble")
    
    # モジュールレベルで定義された PredictionModel を使用
    model1 = PredictionModel(pred_result_df1)
    model2 = PredictionModel(pred_result_df2)
    
    # モデルをアンサンブルに追加
    ensemble_model.add_model("lasso", model1, weight=weights[0])
    ensemble_model.add_model("lgbm", model2, weight=weights[1])
    
    # アンサンブル予測を実行
    # どちらかの予測結果のインデックスを使用
    X_dummy = pd.DataFrame(index=pred_result_df1.index)
    ensembled_pred_df = ensemble_model.predict(X_dummy)
    
    # 結果にTarget列を追加（存在する場合）
    if 'Target' in pred_result_df1.columns:
        ensembled_pred_df = pd.merge(
            ensembled_pred_df,
            pred_result_df1[['Target']],
            left_index=True,
            right_index=True,
            how='left'
        )
    
    return ensembled_pred_df, ensemble_model

def update_ensembled_model(ensembled_dataset_path: str | os.PathLike[str], 
                           ensembled_pred_df: pd.DataFrame, 
                           source_dataset: DatasetManager,
                           ensemble_model: EnsembleModel) -> DatasetManager:
    """
    アンサンブルモデルとその予測結果を保存する
    
    Args:
        ensembled_dataset_path: アンサンブル予測結果を保存するパス
        ensembled_pred_df: アンサンブル予測結果のDataFrame
        source_dataset: 元となるデータセット（特徴量やターゲットを取得するため）
        ensemble_model: 保存するアンサンブルモデル
        
    Returns:
        DatasetManager: 更新されたデータセット
    """
    # アンサンブル用データセットマネージャーを作成
    dataset_ensembled = DatasetManager(ensembled_dataset_path)
    
    # 訓練データ、テストデータなどを元のデータセットからコピー
    dataset_ensembled.set_train_test_data(
        source_dataset.target_train,
        source_dataset.target_test,
        source_dataset.features_train,
        source_dataset.features_test
    )
    
    # 生の目的変数とオーダー価格もコピー
    dataset_ensembled.archive_raw_target(source_dataset.raw_target)
    dataset_ensembled.archive_order_price(source_dataset.order_price)
    
    # アンサンブル予測結果を保存
    dataset_ensembled.set_pred_result(ensembled_pred_df)
    dataset_ensembled.save()
    
    # アンサンブルモデル自体を保存
    model_export_dir = os.path.join(ensembled_dataset_path, "ensemble_model")
    ensemble_model.export(model_export_dir)
    
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
        flag_manager.set_flags(turn_true=turn_true)
        print(flag_manager.get_flags())
        
        # データセットの読み込み
        if flag_manager.flags[Flags.UPDATE_DATASET] or flag_manager.flags[Flags.UPDATE_MODELS]:
            # 各モデル用のDatasetManagerを作成
            dataset_lasso = DatasetManager(ML_DATASET_PATH1)
            dataset_lgbm = DatasetManager(ML_DATASET_PATH2)
            dataset_ensembled = DatasetManager(ML_DATASET_ENSEMBLED_PATH)
            
            '''データの更新・読み込み'''
            stock_dfs_dict = await read_and_update_data(universe_filter)
            
            '''学習・予測'''
            ensemble_weights = [6.7, 1.3]
            necessary_dfs_dict = get_necessary_dfs(stock_dfs_dict, train_start_day, train_end_day, 
                                                  SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)
            
            # LASSO モデルの更新と予測 (学習/読み込みを制御)
            dataset_lasso, pred_result_df1 = update_1st_model(
                dataset_lasso, necessary_dfs_dict,
                train_start_day, train_end_day, test_start_day, test_end_day,
                force_learn=learn  # 引数で明示的に学習が指定された場合は強制学習
            )
            
            # LightGBM モデルの更新と予測 (学習/読み込みを制御)
            dataset_lgbm, pred_result_df2 = update_2nd_model(
                dataset_lasso, dataset_lgbm, stock_dfs_dict, necessary_dfs_dict, pred_result_df1,
                train_start_day, train_end_day, test_start_day, test_end_day,
                SECTOR_REDEFINITIONS_CSV,
                force_learn=learn  # 引数で明示的に学習が指定された場合は強制学習
            )
            
            # アンサンブルモデルを使用して予測結果を結合
            ensembled_pred_df = ensemble_pred_results(
                pred_result_df1, pred_result_df2, ensemble_weights
            )
            
            # アンサンブルモデルを取得 - この関数はアンサンブルモデルも返す必要があります
            # pred_result_df1, pred_result_df2 および weights から再度モデルを作成
            ensemble_model = EnsembleModel(name="Sector_Ensemble")
            
                
            # モデルをアンサンブルに追加
            ensemble_model.add_model("lasso", PredictionModel(pred_result_df1), weight=ensemble_weights[0])
            ensemble_model.add_model("lgbm", PredictionModel(pred_result_df2), weight=ensemble_weights[1])
            
            # アンサンブル結果とモデルの保存
            dataset_ensembled = update_ensembled_model(
                ML_DATASET_ENSEMBLED_PATH, ensembled_pred_df, dataset_lasso, ensemble_model
            )
            
            Slack.send_message(message = f'予測が完了しました。')
        
        # トレード実行
        from facades import TradingFacade
        trade_facade = TradingFacade()
        
        '''新規建'''
        if flag_manager.flags[Flags.TAKE_NEW_POSITIONS]:
            # DatasetManagerをMLDatasetに変換するか、TradingFacadeを更新する必要がある
            # ここでは互換性のためにDatasetManagerのプロパティを使用
            await trade_facade.take_positions(
                ml_dataset=dataset_ensembled,  # DatasetManagerはMLDatasetと同じインターフェースを提供
                SECTOR_REDEFINITIONS_CSV=SECTOR_REDEFINITIONS_CSV,
                num_sectors_to_trade=trading_sector_num,
                num_candidate_sectors=candidate_sector_num,
                top_slope=top_slope)
        
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
        error_handler.handle_exception(Paths.ERROR_LOG_BACKUP)
        Slack.send_error_log(f'エラーが発生しました。\n詳細は{Paths.ERROR_LOG_CSV}を確認してください。')

#%% パラメータ類
if __name__ == '__main__':
    '''パス類'''
    SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv' #別でファイルを作っておく
    SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet' #出力のみなのでファイルがなくてもOK
    ML_DATASET_PATH1 = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_to_2023'
    ML_DATASET_PATH2 = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LGBM_to_2023'
    ML_DATASET_EMSEMBLED_PATH = f'{Paths.ML_DATASETS_FOLDER}/48sectors_Ensembled_to_2023'
    '''ユニバースを絞るフィルタ'''
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))" #現行のTOPIX500
    '''上位・下位何業種を取引対象とするか？'''
    trading_sector_num = 3
    candidate_sector_num = 5
    '''トップ予想の業種にどれほどの傾斜をかけるか'''
    top_slope = 1
    '''学習期間'''
    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2023, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31) #ずっと先の未来を指定
    '''学習するか否か'''
    learn = True
    predict = True

#%% 実行
if __name__ == '__main__':
    # StockAcquisitionFacadeのインポート
    from facades import StockAcquisitionFacade
    
    asyncio.get_event_loop().run_until_complete(main(ML_DATASET_PATH1, ML_DATASET_PATH2, ML_DATASET_EMSEMBLED_PATH, 
                                                     SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET,
                                                     universe_filter, trading_sector_num, candidate_sector_num,
                                                     train_start_day, train_end_day, test_start_day, test_end_day,
                                                     top_slope, learn, predict))