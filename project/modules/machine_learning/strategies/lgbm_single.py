import pandas as pd
from datetime import datetime
from machine_learning.factory import CollectionFactory
from machine_learning.params import LgbmParams

class LgbmSingle:
    
    @staticmethod
    def run(path: str,
            target_df: pd.DataFrame, features_df: pd.DataFrame, raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
            train_start_date: datetime, train_end_date: datetime, 
            test_start_date: datetime = None, test_end_date: datetime = None,
            sector_col_name: str = 'Sector', no_shift_features: list[str] | None = None, reuse_features_df:bool = False,
            train: bool = True, *args, **kwargs):
        
        if train:
            lgbm_collection = LgbmSingle._train_and_pred(path, target_df, features_df, raw_target_df, order_price_df,
                                                         train_start_date, train_end_date, test_start_date, test_end_date,
                                                         sector_col_name, no_shift_features, reuse_features_df, *args, **kwargs)
        else:
            lgbm_collection = LgbmSingle._pred(path, target_df, features_df, raw_target_df, order_price_df,
                                               train_start_date, train_end_date, test_start_date, test_end_date,
                                               sector_col_name, no_shift_features, reuse_features_df, *args, **kwargs)            
        return lgbm_collection

    @staticmethod
    def _train_and_pred(path: str,
                        target_df: pd.DataFrame, features_df: pd.DataFrame, raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
                        train_start_date: datetime, train_end_date: datetime, 
                        test_start_date: datetime = None, test_end_date: datetime = None,
                        sector_col_name: str = 'Sector', no_shift_features: list[str] | None = None, reuse_features_df:bool = False,
                        *args, **kwargs) -> pd.DataFrame:
        """
        リファクタリング後のコードを使用して学習と予測を行う
        
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            
        Returns:
            pd.DataFrame: 予測結果のデータフレーム
        """
        # コレクションの作成
        lgbm_collection = CollectionFactory.get_collection(path)
        lgbm_collection.generate_models([("LGBM", "lgbm", None)])
        lgbm_collection.set_train_test_data_all(target_df=target_df, features_df=features_df,
                                                train_start_date=train_start_date, train_end_date=train_end_date,
                                                test_start_date=test_start_date, test_end_date=test_end_date,
                                                outlier_threshold=3, no_shift_features=no_shift_features, reuse_features_df=reuse_features_df,
                                                separate_by_sector=False)
        # 全モデルの学習
        params = LgbmParams(categorical_features = ['Sector_cat'])
        lgbm_collection.set_params_all(params)
        lgbm_collection.train_all()
        
        # 全モデルの予測
        lgbm_collection.predict_all()
           
        # その他必要データの格納
        lgbm_collection.set_raw_target_for_all(raw_target_df)
        lgbm_collection.set_order_price_for_all(order_price_df)

        lgbm_collection.save(path)
        return lgbm_collection

    @staticmethod
    def _pred(path: str,
              target_df: pd.DataFrame, features_df: pd.DataFrame, raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
              train_start_date: datetime, train_end_date: datetime, 
              test_start_date: datetime = None, test_end_date: datetime = None,
              sector_col_name: str = 'Sector', no_shift_features: list[str] | None = None, reuse_features_df:bool = False,
              *args, **kwargs) -> pd.DataFrame:
        """
        リファクタリング後のコードを使用して学習と予測を行う
        
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            
        Returns:
            pd.DataFrame: 予測結果のデータフレーム
        """
        # コレクションの作成
        lgbm_collection = CollectionFactory.get_collection(path)
        lgbm_collection.generate_models([("LGBM", "lgbm", None)])
        lgbm_collection.set_train_test_data_all(target_df=target_df, features_df=features_df,
                                                train_start_date=train_start_date, train_end_date=train_end_date,
                                                test_start_date=test_start_date, test_end_date=test_end_date,
                                                outlier_threshold=3, no_shift_features=no_shift_features, reuse_features_df=reuse_features_df,
                                                separate_by_sector=False)

        # 全モデルの予測
        params = LgbmParams(categorical_features = ['Sector_cat'])
        lgbm_collection.set_params_all(params)
        lgbm_collection.predict_all()
           
        # その他必要データの格納
        lgbm_collection.set_raw_target_for_all(raw_target_df)
        lgbm_collection.set_order_price_for_all(order_price_df)

        lgbm_collection.save(path)
        return lgbm_collection