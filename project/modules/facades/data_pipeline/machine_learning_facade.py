from typing import Dict, List, Literal
import pandas as pd
from datetime import datetime
import os

from calculation import TargetCalculator, FeaturesCalculator, SectorIndex
from models.machine_learning.ml_dataset import MLDatasets, SingleMLDataset
from models.machine_learning.ensembles import EnsembleMethodFactory
from models.machine_learning.models import LassoModel, LgbmModel
from models.machine_learning.loaders import DatasetLoader
from utils.notifier import SlackNotifier


class MachineLearningFacade:
    def __init__(self, mode: Literal['train_and_predict', 'predict_only', 'none'],
                 stock_dfs_dict: Dict[str, pd.DataFrame],
                 sector_redef_csv_path: str, sector_index_parquet_path: str,
                 datasets1_path: str, datasets2_path:str,
                 ensembled_datasets_path: str, model1_weight: float, model2_weight: float,
                 train_start_day: datetime, train_end_day: datetime, test_start_day: datetime, test_end_day: datetime):
        self.mode = mode
        self.stock_dfs_dict = stock_dfs_dict
        self.sector_redef_csv_path = sector_redef_csv_path
        self.sector_index_parquet_path = sector_index_parquet_path
        self.datasets1_path = datasets1_path
        self.datasets2_path = datasets2_path
        self.ensembled_datasets_path = ensembled_datasets_path
        self.model1_weight = model1_weight
        self.model2_weight = model2_weight
        self.train_start_day = train_start_day
        self.train_end_day = train_end_day
        self.test_start_day = test_start_day
        self.test_end_day = test_end_day

        # Slack通知用
        self.slack = SlackNotifier(program_name=os.path.basename(__file__))

        self.necessary_dfs_dict: Dict[str, pd.DataFrame] | None = None
        self.target_df: pd.DataFrame | None = None
        self.raw_target_df: pd.DataFrame | None = None
        self.order_price_df: pd.DataFrame | None = None
        self.new_sector_price_df: pd.DataFrame | None = None
        self.features_df1: pd.DataFrame | None = None
        self.features_df2: pd.DataFrame | None = None
        self.ml_datasets1: MLDatasets | None = None
        self.ml_datasets2: MLDatasets | None = None
        self.ensembled_pred_df: pd.DataFrame | None = None
    
    def execute(self) -> MLDatasets | None:
        if self.mode == 'none':
            return None
        self._get_necessary_dfs()
        if self.mode == 'train_and_predict':
            self._train_1st_model()
            self._predict_1st_model()
            self._train_2nd_model()
            self._predict_2nd_model()
            self._ensemble()
            self._update_ensembled_model()
            result = self.ensembled_ml_datasets
        elif self.mode == 'predict_only':
            self.ml_datasets1 = self._load_model(dataset_root=self.datasets1_path)
            self.features_df1 = self._get_features_df(
                adopt_features_price=False,
                adopt_size_factor=False,
                adopt_eps_factor=False,
                adopt_sector_categorical=False,
                add_rank=False,
            )
            self._update_test_data(self.ml_datasets1, self.target_df, self.features_df1)
            self._predict_1st_model()
            self.ml_datasets2 = self._load_model(dataset_root=self.datasets2_path)
            self.features_df2 = self._get_features_df(
                adopt_features_price=True,
                adopt_size_factor=True,
                adopt_eps_factor=True,
                adopt_sector_categorical=True,
                add_rank=True,
                mom_duration=[5, 21],
                vola_duration=[5, 21],
            )
            self._append_pred_in_1st_model()
            self._update_test_data(self.ml_datasets2, self.target_df, self.features_df2)
            self._predict_2nd_model()
            self._ensemble()
            self._update_ensembled_model()
            result = self.ensembled_ml_datasets
        elif self.mode == 'load_only':
            self.ml_datasets1 = self._load_model(dataset_root=self.datasets1_path)
            self.ml_datasets2 = self._load_model(dataset_root=self.datasets2_path)
            self.ensembled_ml_datasets = self._load_model(dataset_root=self.ensembled_datasets_path)
            result = self.ensembled_ml_datasets
        else:
            return None

        if result is not None:
            self._notify_latest_prediction_date(result)
        return result


    def _get_necessary_dfs(self):
        sic = SectorIndex(self.stock_dfs_dict, self.sector_redef_csv_path, self.sector_index_parquet_path)
        new_sector_price_df, order_price_df = sic.calc_sector_index()
        raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(
            new_sector_price_df,
            reduce_components=1,
            train_start_day=self.train_start_day,
            train_end_day=self.train_end_day,
        )
        self.target_df = target_df
        self.raw_target_df = raw_target_df
        self.order_price_df = order_price_df
        self.new_sector_price_df = new_sector_price_df


    def _load_model(self, dataset_root: str) -> MLDatasets:
        dsl = DatasetLoader(dataset_root = dataset_root)
        return dsl.load_datasets()


    def _train_1st_model(self):
        dsl = DatasetLoader(dataset_root = self.datasets1_path)

        self.features_df1 = self._get_features_df(adopt_features_price = False,
                                adopt_size_factor = False,
                                adopt_eps_factor = False,
                                adopt_sector_categorical = False,
                                add_rank = False
                                )
        self.ml_datasets1 = dsl.create_grouped_datasets(
            target_df = self.target_df, 
            features_df = self.features_df1,
            train_start_day = self.train_start_day,
            train_end_day = self.train_end_day,
            test_start_day = self.test_start_day,
            test_end_day = self.test_end_day,
            raw_target_df = self.raw_target_df,
            order_price_df = self.order_price_df,
            outlier_threshold=3,
        )

        # 学習
        lasso_model = LassoModel()
        for _, single_ml in self.ml_datasets1.items():
            print(single_ml.get_name())
            trainer_outputs = lasso_model.train(
                single_ml.train_test_materials.target_train_df,
                single_ml.train_test_materials.features_train_df,
            )
            single_ml.archive_ml_objects(trainer_outputs.model, trainer_outputs.scaler)
            single_ml.save()
            self.ml_datasets1.replace_model(single_ml_dataset = single_ml)
    

    def _predict_1st_model(self):
        lasso_model = LassoModel()
        for _, single_ml in self.ml_datasets1.items():
            pred_result_df = lasso_model.predict(
                single_ml.train_test_materials.target_test_df,
                single_ml.train_test_materials.features_test_df,
                single_ml.ml_object_materials.model,
                single_ml.ml_object_materials.scaler,
            )
            single_ml.archive_pred_result(pred_result_df)
            single_ml.save()
            self.ml_datasets1.replace_model(single_ml_dataset = single_ml)
        print('第一モデル予測完了')


    def _train_2nd_model(self):
        single_ml = SingleMLDataset(dataset_folder_path = f'{self.datasets2_path}/LightGBM', name='LightGBM')
        
        self.features_df2 = self._get_features_df(adopt_features_price = True,
                                adopt_size_factor = True,
                                adopt_eps_factor = True,
                                adopt_sector_categorical = True,
                                add_rank = True,
                                mom_duration = [5, 21],
                                vola_duration = [5, 21]
                                )
        self._append_pred_in_1st_model()
        single_ml.archive_train_test_data(target_df = self.target_df,
                                          features_df = self.features_df2,
                                          train_start_day = self.train_start_day,
                                          train_end_day = self.train_end_day,
                                          test_start_day = self.test_start_day,
                                          test_end_day = self.test_end_day,
                                          outlier_threshold = 3,
                                          no_shift_features=['1stModel_pred'], 
                                          reuse_features_df=True)
        single_ml.archive_raw_target(raw_target_df = self.raw_target_df)
        single_ml.archive_order_price(order_price_df = self.order_price_df)

        lgbm_model = LgbmModel()
        trainer_outputs = lgbm_model.train(single_ml.train_test_materials.target_train_df, 
                                           single_ml.train_test_materials.features_train_df,
                                           categorical_features = ['Sector_cat'])
        single_ml.archive_ml_objects(model = trainer_outputs.model, scaler = None)
        self.ml_datasets2 = MLDatasets()
        self.ml_datasets2.append_model(single_ml_dataset=single_ml)
        self.ml_datasets2.save_all()


    def _predict_2nd_model(self):
        lgbm_model = LgbmModel()
        for _, single_ml in self.ml_datasets2.items():
            pred_result_df = lgbm_model.predict(
                single_ml.train_test_materials.target_test_df,
                single_ml.train_test_materials.features_test_df,
                single_ml.ml_object_materials.model,
            )
            single_ml.archive_pred_result(pred_result_df)
            single_ml.save()
            self.ml_datasets2.replace_model(single_ml_dataset = single_ml)
        print('第二モデル予測完了')


    def _ensemble(self):
        emf = EnsembleMethodFactory()
        rank_method = emf.create_method(method_name='by_rank')
        ensemble_inputs = [(self.ml_datasets1.get_pred_result(), self.model1_weight),
                           (self.ml_datasets2.get_pred_result(), self.model2_weight)]
        self.ensembled_pred_df =  rank_method.ensemble(ensemble_inputs)


    def _update_ensembled_model(self):
        self.ensembled_ml_datasets = MLDatasets()
        single_ml_dataset = SingleMLDataset(self.ensembled_datasets_path, 'Ensembled')
        single_ml_dataset.archive_pred_result(self.ensembled_pred_df)
        single_ml_dataset.archive_order_price(self.ml_datasets1.get_order_price())
        single_ml_dataset.archive_raw_target(self.ml_datasets1.get_raw_target())
        self.ensembled_ml_datasets.append_model(single_ml_dataset=single_ml_dataset)
        self.ensembled_ml_datasets.save_all()

    def _notify_latest_prediction_date(self, ml_datasets: MLDatasets) -> None:
        df = ml_datasets.get_pred_result()
        latest_date = df.index.get_level_values('Date')[-1]
        self.slack.send_message(f'最新予測日: {latest_date}')


    def _get_features_df(self, adopt_features_price: bool, adopt_size_factor: bool, adopt_eps_factor: bool,
                         adopt_sector_categorical: bool, add_rank: bool,
                         mom_duration: List[int] | None = None, 
                         vola_duration: List[int] | None = None) -> pd.DataFrame:
        return FeaturesCalculator.calculate_features(
                new_sector_price = self.new_sector_price_df,
                new_sector_list = pd.read_csv(self.sector_redef_csv_path),
                stock_dfs_dict = self.stock_dfs_dict,
                adopts_features_indices=True,
                adopts_features_price=adopt_features_price, #TODO LASSO: False, LightGBM: True
                groups_setting=None,
                names_setting=None,
                currencies_type='relative',
                adopt_1d_return=True,
                mom_duration=mom_duration, #TODO LightGBM [5, 21]
                vola_duration=vola_duration, #TODO LightGBM [5, 21]
                adopt_size_factor=adopt_size_factor, #TODO LASSO: False, LightGBM: True
                adopt_eps_factor=adopt_eps_factor, #TODO LASSO: False, LightGBM: True
                adopt_sector_categorical=adopt_sector_categorical, #TODO LASSO: False, LightGBM: True
                add_rank=add_rank, #TODO LASSO: False, LightGBM: True
                )
    
    def _append_pred_in_1st_model(self):
        if not '1stModel_pred' in self.features_df2.columns:
            pred_in_1st_model = self.ml_datasets1.get_pred_result()
            self.features_df2 = pd.merge(self.features_df2, pred_in_1st_model[['Pred']], how='outer',
                                left_index=True, right_index=True) # LASSOでの予測結果をlightGBMの特徴量として追加
            self.features_df2 = self.features_df2.rename(columns={'Pred':'1stModel_pred'})

    def _update_test_data(self, ml_datasets: MLDatasets, target_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
        for name, single_ml in ml_datasets.items():
            sector_target = target_df[target_df.index.get_level_values('Sector') == name]
            sector_features = features_df[features_df.index.get_level_values('Sector') == name]
            sector_target = sector_target[(sector_target.index.get_level_values('Date') >= self.test_start_day) &
                                         (sector_target.index.get_level_values('Date') <= self.test_end_day)]
            sector_features = sector_features[(sector_features.index.get_level_values('Date') >= self.test_start_day) &
                                             (sector_features.index.get_level_values('Date') <= self.test_end_day)]

            if sector_target.empty or sector_features.empty:
                continue

            single_ml.train_test_data._target_test_df = sector_target
            single_ml.train_test_data._features_test_df = sector_features
            single_ml.save()
            ml_datasets.replace_model(single_ml_dataset=single_ml)
    

if __name__ == '__main__':
    from facades.data_pipeline.data_update_facade import DataUpdateFacade
    from utils.paths import Paths

    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))" #現行のTOPIX500
    sector_redef_csv_path = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv' #別でファイルを作っておく
    sector_index_parquet_path = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet' #出力のみなのでファイルがなくてもOK
    datasets1_path = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250607'
    datasets2_path = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LightGBMlearned_in_250607'
    ensembled_datasets_path = f'{Paths.ML_DATASETS_FOLDER}/48sectors_Ensembled_learned_in_250607'

    async def main():
        duf = DataUpdateFacade(mode='load_only', universe_filter=universe_filter)
        stock_dfs_dict = await duf.execute()
        mlf = MachineLearningFacade(mode='train_and_predict', 
                                    stock_dfs_dict=stock_dfs_dict,
                                    sector_redef_csv_path=sector_redef_csv_path, 
                                    sector_index_parquet_path=sector_index_parquet_path,
                                    datasets1_path=datasets1_path, 
                                    datasets2_path=datasets2_path,
                                    ensembled_datasets_path=ensembled_datasets_path,
                                    model1_weight=6.7,
                                    model2_weight=1.3,
                                    train_start_day=datetime(2014,1,1),
                                    train_end_day=datetime(2021,12,31),
                                    test_start_day=datetime(2022,1,1),
                                    test_end_day=datetime.today()
                                    )
        mlf.execute()
    
    import asyncio
    asyncio.get_event_loop().run_until_complete(main())