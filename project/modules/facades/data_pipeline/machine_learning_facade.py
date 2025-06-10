from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Literal
import pandas as pd

from calculation import TargetCalculator, FeaturesCalculator, SectorIndex
from models.machine_learning.ml_dataset import MLDatasets, SingleMLDataset
from models.machine_learning.ensembles import EnsembleMethodFactory
from models.machine_learning.models import LassoModel, LgbmModel
from models.machine_learning.loaders import DatasetLoader


class MachineLearningFacade:
    """Facade for training and predicting models."""

    def __init__(
        self,
        mode: Literal['train_and_predict', 'predict_only', 'none'],
        dataset_path1: str,
        dataset_path2: str,
        ensembled_dataset_path: str,
        sector_redef_csv: str,
        sector_index_parquet: str,
        train_start_day: datetime,
        train_end_day: datetime,
        test_start_day: datetime,
        test_end_day: datetime,
    ) -> None:
        self.mode = mode
        self.dataset_path1 = dataset_path1
        self.dataset_path2 = dataset_path2
        self.ensembled_dataset_path = ensembled_dataset_path
        self.sector_redef_csv = sector_redef_csv
        self.sector_index_parquet = sector_index_parquet
        self.train_start_day = train_start_day
        self.train_end_day = train_end_day
        self.test_start_day = test_start_day
        self.test_end_day = test_end_day

    def _get_necessary_dfs(self, stock_dfs_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        sic = SectorIndex(stock_dfs_dict, self.sector_redef_csv, self.sector_index_parquet)
        new_sector_price_df, order_price_df = sic.calc_sector_index()
        raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(
            new_sector_price_df,
            reduce_components=1,
            train_start_day=self.train_start_day,
            train_end_day=self.train_end_day,
        )
        return {
            'new_sector_price_df': new_sector_price_df,
            'order_price_df': order_price_df,
            'raw_target_df': raw_target_df,
            'target_df': target_df,
        }

    def _update_1st_model(self, necessary: Dict[str, pd.DataFrame]) -> MLDatasets:
        dsl = DatasetLoader(self.dataset_path1)
        target_df = necessary['target_df']
        if self.mode == 'train_and_predict':
            features_df = FeaturesCalculator.calculate_features(
                necessary['new_sector_price_df'],
                None,
                None,
                adopts_features_indices=True,
                adopts_features_price=False,
                groups_setting=None,
                names_setting=None,
                currencies_type='relative',
                adopt_1d_return=True,
                mom_duration=None,
                vola_duration=None,
                adopt_size_factor=False,
                adopt_eps_factor=False,
                adopt_sector_categorical=False,
                add_rank=False,
            )
            ml_datasets = dsl.create_grouped_datasets(
                target_df,
                features_df,
                self.train_start_day,
                self.train_end_day,
                self.test_start_day,
                self.test_end_day,
                raw_target_df=necessary['raw_target_df'],
                order_price_df=necessary['order_price_df'],
                outlier_threshold=3,
            )
        else:
            ml_datasets = dsl.load_datasets()

        lasso_model = LassoModel()
        for key, single_ml in ml_datasets.items():
            if self.mode == 'train_and_predict':
                trainer_outputs = lasso_model.train(
                    single_ml.train_test_materials.target_train_df,
                    single_ml.train_test_materials.features_train_df,
                )
                single_ml.archive_ml_objects(trainer_outputs.model, trainer_outputs.scaler)
            pred_result_df = lasso_model.predict(
                single_ml.train_test_materials.target_test_df,
                single_ml.train_test_materials.features_test_df,
                single_ml.ml_object_materials.model,
                single_ml.ml_object_materials.scaler,
            )
            single_ml.archive_pred_result(pred_result_df)
            single_ml.save()
        return ml_datasets

    def _update_2nd_model(self, ml_datasets1: MLDatasets, stock_dfs_dict: Dict[str, pd.DataFrame], necessary: Dict[str, pd.DataFrame]) -> MLDatasets:
        dsl = DatasetLoader(self.dataset_path2)
        ml_datasets2 = dsl.load_datasets()
        for key, single_ml in ml_datasets2.items():
            if self.mode == 'train_and_predict':
                features_df = FeaturesCalculator.calculate_features(
                    necessary['new_sector_price_df'],
                    pd.read_csv(self.sector_redef_csv),
                    stock_dfs_dict,
                    adopts_features_indices=True,
                    adopts_features_price=True,
                    groups_setting=None,
                    names_setting=None,
                    currencies_type='relative',
                    adopt_1d_return=True,
                    mom_duration=[5, 21],
                    vola_duration=[5, 21],
                    adopt_size_factor=True,
                    adopt_eps_factor=True,
                    adopt_sector_categorical=True,
                    add_rank=True,
                )
                pred_in_1st_model = ml_datasets1.get_pred_result()
                features_df = pd.merge(
                    features_df,
                    pred_in_1st_model[['Pred']],
                    how='outer',
                    left_index=True,
                    right_index=True,
                )
                features_df = features_df.rename(columns={'Pred': '1stModel_pred'})
                single_ml.archive_train_test_data(
                    necessary['target_df'],
                    features_df,
                    self.train_start_day,
                    self.train_end_day,
                    self.test_start_day,
                    self.test_end_day,
                    outlier_threshold=3,
                    no_shift_features=['1stModel_pred'],
                    reuse_features_df=True,
                )
                single_ml.archive_raw_target(necessary['raw_target_df'])
                single_ml.archive_order_price(necessary['order_price_df'])

            lgbm_model = LgbmModel()
            if self.mode == 'train_and_predict':
                trainer_outputs = lgbm_model.train(
                    single_ml.train_test_materials.target_train_df,
                    single_ml.train_test_materials.features_train_df,
                    categorical_features=['Sector_cat'],
                )
                single_ml.archive_ml_objects(model=trainer_outputs.model, scaler=None)
            pred_result_df = lgbm_model.predict(
                single_ml.train_test_materials.target_test_df,
                single_ml.train_test_materials.features_test_df,
                single_ml.ml_object_materials.model,
            )
            single_ml.archive_pred_result(pred_result_df)
            ml_datasets2.append_model(single_ml)
            ml_datasets2.save_all()
            return ml_datasets2

    def _ensemble(self, pred1: pd.DataFrame, pred2: pd.DataFrame) -> pd.DataFrame:
        emf = EnsembleMethodFactory()
        rank_method = emf.create_method('by_rank')
        return rank_method.ensemble([(pred1, 6.7), (pred2, 1.3)])

    def _update_ensembled(self, ensembled_pred_df: pd.DataFrame) -> SingleMLDataset:
        single = SingleMLDataset(self.ensembled_dataset_path, 'Ensembled')
        single.archive_pred_result(ensembled_pred_df)
        single.save()
        return single

    async def execute(self, stock_dfs_dict: Dict[str, pd.DataFrame]) -> Optional[SingleMLDataset]:
        if self.mode == 'none':
            return None
        necessary = self._get_necessary_dfs(stock_dfs_dict)
        ml_dataset1 = self._update_1st_model(necessary)
        ml_dataset2 = self._update_2nd_model(ml_dataset1, stock_dfs_dict, necessary)
        pred1 = ml_dataset1.get_pred_result()
        pred2 = ml_dataset2.get_pred_result()
        ensembled = self._ensemble(pred1, pred2)
        ml_dataset_ensembled = self._update_ensembled(ensembled)
        return ml_dataset_ensembled

