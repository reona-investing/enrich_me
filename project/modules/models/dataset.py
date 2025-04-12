from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Optional, List, Callable

from machine_learning.params import BaseParams
from machine_learning.utils import DataProcessor


class BaseModel(ABC):
    """機械学習モデルの基底クラス"""
    
    def __init__(self, name: str, params: Optional[BaseParams] = None):
        """
        Args:
            name: モデル名
            params: モデルパラメータ
        """
        self.name = name
        self.params = params
        self.model = None
        self.scaler = None
        self.target_train_df = None
        self.features_train_df = None
        self.target_test_df = None
        self.features_test_df = None
        self.train_start_date = None
        self.train_end_date = None
        self.test_start_date = None
        self.test_end_date = None
        self.pred_result_df = None
        
        # データ前処理関連の設定
        self.outlier_threshold = 0  # デフォルトでは外れ値除去なし
        self.no_shift_features = []  # デフォルトではすべての特徴量をシフト
        self.get_next_open_date_func = None  # 次の営業日を取得する関数（外部から設定）
        self.reuse_features_df = False  # デフォルトでは特徴量を再利用しない

        # 追加する新しいプロパティ (PostProcessingData に相当)
        self.raw_target_df = None  # 生の目的変数（処理前）
        self.order_price_df = None  # 発注価格情報

        
    def load_dataset(self, 
                    target_df: pd.DataFrame, 
                    features_df: pd.DataFrame,
                    train_start_date: datetime,
                    train_end_date: datetime,
                    test_start_date: Optional[datetime] = None,
                    test_end_date: Optional[datetime] = None,
                    outlier_threshold: float = 0,
                    no_shift_features: List[str] = None,
                    get_next_open_date_func: Optional[Callable] = None,
                    reuse_features_df: bool = False) -> None:
        """
        データセットを読み込み、前処理を適用し、学習用とテスト用に分割する
        
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_dateの翌日）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
            outlier_threshold: 外れ値除去の閾値（±何σ、0の場合は除去なし）
            no_shift_features: シフトしない特徴量のリスト
            get_next_open_date_func: 次の営業日を取得する関数（utils.jquants_api_utils.get_next_open_dateなど）
            reuse_features_df: 特徴量を他の業種から再利用するか
        """
        # 設定の保存
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date if test_start_date else train_end_date
        self.test_end_date = test_end_date if test_end_date else target_df.index.get_level_values('Date').max()
        self.outlier_threshold = outlier_threshold
        self.no_shift_features = no_shift_features or []
        self.get_next_open_date_func = get_next_open_date_func
        self.reuse_features_df = reuse_features_df
        
        # DataFrameをコピーして元のデータを変更しないようにする
        target_df = target_df.copy()
        features_df = features_df.copy()
        
        # データの前処理
        if get_next_open_date_func:
            # 次の営業日を追加
            target_df = DataProcessor.append_next_business_day_row(target_df, get_next_open_date_func)
            if not reuse_features_df:
                features_df = DataProcessor.append_next_business_day_row(features_df, get_next_open_date_func)
                
        # 特徴量のシフトと目的変数との整合性確保
        features_df = DataProcessor.shift_features(features_df, self.no_shift_features)
        features_df = DataProcessor.align_index(features_df, target_df)
        
        # 学習データとテストデータに分割
        self.target_train_df = DataProcessor.narrow_period(target_df, train_start_date, train_end_date)
        self.target_test_df = DataProcessor.narrow_period(target_df, self.test_start_date, self.test_end_date)
        self.features_train_df = DataProcessor.narrow_period(features_df, train_start_date, train_end_date)
        self.features_test_df = DataProcessor.narrow_period(features_df, self.test_start_date, self.test_end_date)
        
        # 外れ値除去（学習データのみ）
        if outlier_threshold > 0:
            self.target_train_df, self.features_train_df = DataProcessor.remove_outliers(
                self.target_train_df, self.features_train_df, outlier_threshold
            )

    def update_dataset(self, 
                      target_df: pd.DataFrame, 
                      features_df: pd.DataFrame,
                      test_start_date: Optional[datetime] = None,
                      test_end_date: Optional[datetime] = None,
                      get_next_open_date_func: Optional[Callable] = None) -> None:
        """
        学習済みモデルのテストデータセットを更新する
        
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            test_start_date: テストデータの開始日（省略時は既存設定を維持）
            test_end_date: テストデータの終了日（省略時は既存設定を維持）
            get_next_open_date_func: 次の営業日を取得する関数（指定時のみ更新）
        """
        # 設定の更新
        if test_start_date:
            self.test_start_date = test_start_date
        if test_end_date:
            self.test_end_date = test_end_date
        if get_next_open_date_func:
            self.get_next_open_date_func = get_next_open_date_func
            
        # DataFrameをコピーして元のデータを変更しないようにする
        target_df = target_df.copy()
        features_df = features_df.copy()
            
        # データの前処理
        if self.get_next_open_date_func:
            # 次の営業日を追加
            target_df = DataProcessor.append_next_business_day_row(target_df, self.get_next_open_date_func)
            if not self.reuse_features_df:
                features_df = DataProcessor.append_next_business_day_row(features_df, self.get_next_open_date_func)
                
        # 特徴量のシフトと目的変数との整合性確保
        features_df = DataProcessor.shift_features(features_df, self.no_shift_features)
        features_df = DataProcessor.align_index(features_df, target_df)
            
        # テストデータの更新
        self.target_test_df = DataProcessor.narrow_period(target_df, self.test_start_date, self.test_end_date)
        self.features_test_df = DataProcessor.narrow_period(features_df, self.test_start_date, self.test_end_date)
    
    # 新規メソッド：生の目的変数を設定
    def set_raw_target(self, raw_target_df: pd.DataFrame) -> None:
        """
        生の目的変数データフレームを設定する
        
        Args:
            raw_target_df: 生の目的変数のデータフレーム
        """
        self.raw_target_df = raw_target_df.copy()
    
    # 新規メソッド：発注価格情報を設定
    def set_order_price(self, order_price_df: pd.DataFrame) -> None:
        """
        発注価格情報を設定する
        
        Args:
            order_price_df: 発注価格のデータフレーム
        """
        self.order_price_df = order_price_df.copy()
    
    # 新規メソッド：生の目的変数を取得
    def get_raw_target(self) -> pd.DataFrame:
        """生の目的変数データフレームを取得する"""
        if self.raw_target_df is None:
            raise ValueError("生の目的変数がセットされていません。set_raw_target()を先に実行してください。")
        return self.raw_target_df
    
    # 新規メソッド：発注価格情報を取得
    def get_order_price(self) -> pd.DataFrame:
        """発注価格データフレームを取得する"""
        if self.order_price_df is None:
            raise ValueError("発注価格情報がセットされていません。set_order_price()を先に実行してください。")
        return self.order_price_df
    
    @abstractmethod
    def train(self) -> None:
        """モデルを学習する"""
        pass
    
    @abstractmethod
    def predict(self) -> pd.DataFrame:
        """予測を実行しpred_result_dfに格納する"""
        pass
    
    def get_pred_result(self) -> pd.DataFrame:
        """予測結果を取得する"""
        if self.pred_result_df is None:
            raise ValueError("予測が実行されていません。predict()を先に実行してください。")
        return self.pred_result_df