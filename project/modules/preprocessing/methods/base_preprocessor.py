"""
前処理クラスの基底クラス定義
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin, ABC):
    """
    前処理クラスの抽象基底クラス
    
    全ての前処理クラスが継承すべき共通インターフェースを定義。
    scikit-learn互換のTransformerとして動作。
    
    Parameters
    ----------
    copy : bool, default=True
        データをコピーするかどうか
    """
    
    def __init__(self, copy: bool = True):
        self.copy = copy
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'BasePreprocessor':
        """
        前処理のパラメータを学習
        
        Parameters
        ----------
        X : pd.DataFrame
            学習用データ
        y : Any, optional
            目的変数（使用しない場合はNone）
            
        Returns
        -------
        self : BasePreprocessor
            自身のインスタンス
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        学習したパラメータを使って変換を実行
        
        Parameters
        ----------
        X : pd.DataFrame
            変換対象データ
            
        Returns
        -------
        X_transformed : pd.DataFrame
            変換後のデータ
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        fit と transform を同時に実行
        
        Parameters
        ----------
        X : pd.DataFrame
            学習・変換用データ
        y : Any, optional
            目的変数
            
        Returns
        -------
        X_transformed : pd.DataFrame
            変換後のデータ
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """
        入力データの妥当性をチェック
        
        Parameters
        ----------
        X : pd.DataFrame
            チェック対象データ
            
        Raises
        ------
        ValueError
            データが不正な場合
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
    
    def _check_is_fitted(self) -> None:
        """
        fitが実行済みかチェック
        
        Raises
        ------
        ValueError
            fitが実行されていない場合
        """
        if not hasattr(self, 'feature_names_in_'):
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet.")
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        変換後の特徴量名を取得（sklearn互換）
        
        Parameters
        ----------
        input_features : List[str], optional
            入力特徴量名
            
        Returns
        -------
        feature_names : List[str]
            変換後の特徴量名
        """
        if hasattr(self, 'feature_names_in_'):
            return self.feature_names_in_
        return input_features if input_features is not None else []
    
    def _prepare_output(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        出力データの準備（コピーの処理など）
        
        Parameters
        ----------
        X : pd.DataFrame
            元データ
            
        Returns
        -------
        X_output : pd.DataFrame
            出力用データ
        """
        return X.copy() if self.copy else X