from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List

from machine_learning.data.dataset import Dataset
from machine_learning.core.collection import ModelCollection


class Strategy(ABC):
    """モデリング戦略の基底クラス"""
    
    def __init__(self, name: str, save_path: Optional[str] = None):
        """
        Args:
            name: 戦略名
            save_path: 保存先パス（省略可）
        """
        self.name = name
        self.save_path = save_path
        self.dataset = None
        self.collection = None
        self.trained = False
        
    def load_data(self, 
                target_df: pd.DataFrame, 
                features_df: pd.DataFrame,
                raw_target_df: Optional[pd.DataFrame] = None,
                order_price_df: Optional[pd.DataFrame] = None) -> None:
        """
        データを読み込む
        
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            raw_target_df: 生の目的変数のデータフレーム（省略可）
            order_price_df: 発注価格のデータフレーム（省略可）
        """
        self.dataset = Dataset(
            target_df=target_df,
            features_df=features_df,
            raw_target_df=raw_target_df,
            order_price_df=order_price_df
        )
    
    @abstractmethod
    def prepare_data(self, 
                    train_start_date: datetime,
                    train_end_date: datetime,
                    test_start_date: Optional[datetime] = None,
                    test_end_date: Optional[datetime] = None) -> None:
        """
        データの前処理を行う
        
        Args:
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_date）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
        """
        pass
    
    @abstractmethod
    def train(self) -> None:
        """モデルの学習を行う"""
        pass
    
    @abstractmethod
    def predict(self) -> pd.DataFrame:
        """予測を実行する"""
        pass
    
    def evaluate(self, metrics: List[str] = None) -> Dict[str, Any]:
        """
        評価を実行する
        
        Args:
            metrics: 計算する評価指標のリスト（省略時は全指標）
            
        Returns:
            評価結果の辞書
        """
        if not self.trained or self.collection is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        
        # 評価モジュールをインポート
        from machine_learning.evaluation.metrics import calculate_metrics
        
        # 予測結果を取得
        pred_result_df = self.collection.get_result_df()
        
        # 評価を実行
        return calculate_metrics(pred_result_df, metrics)
    
    def save(self, path: Optional[str] = None) -> None:
        """
        戦略を保存する
        
        Args:
            path: 保存先パス（省略時はインスタンス生成時のパスを使用）
        """
        save_path = path if path is not None else self.save_path
        if save_path is None:
            raise ValueError("保存先のパスが指定されていません。")
        
        if self.collection is not None:
            self.collection.save(save_path)
    
    @classmethod
    def load(cls, path: str) -> 'Strategy':
        """
        戦略を読み込む
        
        Args:
            path: 読み込むファイルパス
            
        Returns:
            読み込まれた戦略
        """
        # ModelCollectionを読み込む
        from machine_learning.core.collection import ModelCollection
        collection = ModelCollection.load(path)
        
        # 戦略インスタンスを作成
        strategy = cls(name=collection.name, save_path=path)
        strategy.collection = collection
        strategy.trained = True
        
        return strategy
    
    @classmethod
    @abstractmethod
    def run(cls, 
           path: str,
           target_df: pd.DataFrame, 
           features_df: pd.DataFrame,
           raw_target_df: pd.DataFrame, 
           order_price_df: pd.DataFrame,
           train_start_date: datetime, 
           train_end_date: datetime,
           test_start_date: Optional[datetime] = None, 
           test_end_date: Optional[datetime] = None,
           train: bool = True,
           **kwargs) -> ModelCollection:
        """
        戦略を実行する簡易メソッド
        
        Args:
            path: モデルの保存/読み込み先パス
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            raw_target_df: 生の目的変数のデータフレーム
            order_price_df: 発注価格のデータフレーム
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_date）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
            train: 学習を行うかどうか
            **kwargs: その他のパラメータ
            
        Returns:
            モデルコレクション
        """
        pass