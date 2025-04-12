import pandas as pd
from datetime import datetime
from machine_learning.factory import CollectionFactory
from utils.jquants_api_utils.market_open_utils import get_next_open_date

class LassoBySector:
    
    @staticmethod
    def run(path: str,
            target_df: pd.DataFrame, features_df: pd.DataFrame, raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
            train_start_date: datetime, train_end_date: datetime, 
            test_start_date: datetime = None, test_end_date: datetime = None,
            sector_col_name: str = 'Sector', train: bool = True, *args, **kwargs):
        
        if train:
            lasso_collection = LassoBySector._train_and_pred(path, target_df, features_df, raw_target_df, order_price_df,
                                                             train_start_date, train_end_date, test_start_date, test_end_date,
                                                             sector_col_name, *args, **kwargs)
        else:
            lasso_collection = LassoBySector._pred(path, target_df, features_df, raw_target_df, order_price_df,
                                                   train_start_date, train_end_date, test_start_date, test_end_date,
                                                   sector_col_name, *args, **kwargs)            
        return lasso_collection

    @staticmethod
    def _train_and_pred(path: str,
                        target_df: pd.DataFrame, features_df: pd.DataFrame, raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
                        train_start_date: datetime, train_end_date: datetime, 
                        test_start_date: datetime = None, test_end_date: datetime = None,
                        sector_col_name: str = 'Sector', *args, **kwargs) -> pd.DataFrame:
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
        # セクターごとにモデルを生成、学習、予測
        sectors = target_df.index.get_level_values(sector_col_name).unique().tolist()

        # コレクションの作成
        lasso_collection = CollectionFactory.get_collection(path)
        lasso_collection.generate_models([(sector, "lasso", None) for sector in sectors])
        lasso_collection.set_train_test_data_all(target_df=target_df, features_df=features_df,
                                                 train_start_date=train_start_date, train_end_date=train_end_date,
                                                 test_start_date=test_start_date, test_end_date=test_end_date,
                                                 outlier_threshold= 3, get_next_open_date_func=get_next_open_date)
        
        # 全モデルの学習
        lasso_collection.train_all()
        
        # 全モデルの予測
        lasso_collection.predict_all()
        
        # その他必要データの格納
        lasso_collection.set_raw_target_for_all(raw_target_df)
        lasso_collection.set_order_price_for_all(order_price_df)
        
        # 保存
        lasso_collection.save(path)
        return lasso_collection

    @staticmethod
    def _pred(path: str,
              target_df: pd.DataFrame, features_df: pd.DataFrame, raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
              train_start_date: datetime, train_end_date: datetime, 
              test_start_date: datetime = None, test_end_date: datetime = None,
              sector_col_name: str = 'Sector', *args, **kwargs) -> pd.DataFrame:
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
        lasso_collection = CollectionFactory.get_collection(path)

        lasso_collection.set_train_test_data_all(target_df=target_df, features_df=features_df,
                                                 train_start_date=train_start_date, train_end_date=train_end_date,
                                                 test_start_date=test_start_date, test_end_date=test_end_date,
                                                 outlier_threshold= 3, get_next_open_date_func=get_next_open_date)
        lasso_collection.predict_all()
        
        # その他必要データの格納
        lasso_collection.set_raw_target_for_all(raw_target_df)
        lasso_collection.set_order_price_for_all(order_price_df)

        lasso_collection.save(path)
        return lasso_collection