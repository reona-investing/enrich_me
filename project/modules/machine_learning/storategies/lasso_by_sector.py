import pandas as pd
from datetime import datetime
import os

class LassoBySector:

    @staticmethod
    def train_and_predict_new_code(path: str,
                                   target_df: pd.DataFrame, features_df: pd.DataFrame, raw_target_df: pd.DataFrame, order_price_df: pd.DataFrame,
                                   train_start_date: datetime, train_end_date: datetime, 
                                   test_start_date: datetime, test_end_date: datetime,
                                   sector_col_name: str = 'Sector') -> pd.DataFrame:
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
        from machine_learning.factory import CollectionFactory

        # セクターごとにモデルを生成、学習、予測
        sectors = target_df.index.get_level_values(sector_col_name).unique().tolist()

        # コレクションの作成
        lasso_collection = CollectionFactory.get_collection(path)
        lasso_collection.generate_models([(sector, "lasso") for sector in sectors])
        lasso_collection.set_train_test_data_all(target_df=target_df, features_df=features_df,
                                                 train_start_date=train_start_date, train_end_date=train_end_date,
                                                 test_start_date=test_start_date, test_end_date=test_end_date,
                                                 outlier_threshold= 3)
        
        # 全モデルの学習
        lasso_collection.train_all()
        
        # 全モデルの予測
        lasso_collection.predict_all()
        
        # その他必要データの格納
        lasso_collection.set_raw_target_for_all(raw_target_df)
        lasso_collection.set_order_price_for_all(order_price_df)
        
        # 保存
        lasso_collection.save(path=path)