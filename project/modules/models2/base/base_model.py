"""
ベースモデルインターフェース定義
すべてのモデルはこのインターフェースを実装する
"""
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    すべての単一モデルが実装すべき基本インターフェース
    """
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], **kwargs):
        """
        モデルを学習する
        
        Args:
            X: 特徴量DataFrame
            y: 目的変数（SeriesまたはDataFrame）
            **kwargs: 学習パラメータ
            
        Returns:
            self: メソッドチェーン用に自身を返す
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みモデルで予測を行う
        
        Args:
            X: 特徴量DataFrame
            
        Returns:
            np.ndarray: 予測値の配列
        """
        pass
    
    @property
    @abstractmethod
    def feature_importances(self) -> Optional[pd.DataFrame]:
        """
        特徴量重要度を取得する（該当する場合）
        
        Returns:
            Optional[pd.DataFrame]: 特徴量とその重要度を含むDataFrame、
                                    または特徴量重要度が利用できない場合はNone
        """
        pass