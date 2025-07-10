from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Set
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from utils.timeseries import Duration


class BasePreprocessor(BaseEstimator, TransformerMixin, ABC):
    """
    前処理クラスの抽象基底クラス
    
    全ての前処理クラスが継承すべき共通インターフェースを定義。
    scikit-learn互換のTransformerとして動作。
    時系列データに対応し、指定期間でfitして全期間でtransformすることが可能。
    
    Parameters
    ----------
    copy : bool, default=True
        データをコピーするかどうか
    fit_duration : Duration or None, default=None
        fitに使用する期間を表すDuration。Noneの場合は全期間を使用
    time_column : str
        時間列の名前
    """
    
    def __init__(self,
                 *,
                 copy: bool = True,
                 fit_duration: Optional[Duration] = None,
                 time_column: str):
        self.copy = copy
        self.fit_duration = fit_duration
        self.time_column = time_column
        self._is_fitted = False  # 内部的なfit状態管理
        self._fit_attributes: Set[str] = set()  # fit時に設定された属性を記録
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'BasePreprocessor':
        """前処理のパラメータを学習"""
        pass
    
    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """学習したパラメータを使って変換を実行"""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> Union[pd.DataFrame, np.ndarray]:
        """fit と transform を同時に実行"""
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """入力データの妥当性をチェック"""
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("入力されたDataFrameが空です")
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("入力されたnumpy配列が空です")
            if X.ndim < 2:
                raise ValueError("numpy配列は2次元以上である必要があります")
        else:
            raise ValueError("入力はpandas DataFrameまたはnumpy配列でなければなりません")
    
    def _mark_as_fitted(self, **kwargs) -> None:
        """
        fit完了をマークし、重要な属性を記録
        
        このメソッドを継承クラスのfit()の最後で呼び出すだけで
        バリデーションが機能するようになる
        
        Parameters
        ----------
        **kwargs : 任意のキーワード引数
            fit時に設定された重要な属性を渡す
            例: self._mark_as_fitted(pca_=self.pca_, n_components_=self.n_components)
        """
        self._is_fitted = True
        
        # 渡された属性を記録
        for key, value in kwargs.items():
            setattr(self, key, value)
            self._fit_attributes.add(key)
        
        # 共通の属性も自動で記録
        common_attrs = ['n_features_in_', 'feature_names_in_', 'is_fitted_']
        for attr in common_attrs:
            if hasattr(self, attr):
                self._fit_attributes.add(attr)
    
    def _check_is_fitted(self, additional_attributes: Optional[List[str]] = None) -> None:
        """
        fitが実行済みかチェック（簡素化版）
        
        Parameters
        ----------
        additional_attributes : List[str], optional
            追加でチェックしたい属性があれば指定
        """
        # 基本的なfit状態チェック
        if not self._is_fitted:
            raise ValueError(
                f"この {self.__class__.__name__} インスタンスはまだfitされていません。"
                "使用する前に適切な引数でfitを実行してください。"
            )
        
        # 追加属性のチェック（必要に応じて）
        if additional_attributes:
            missing_attrs = [attr for attr in additional_attributes if not hasattr(self, attr)]
            if missing_attrs:
                raise ValueError(
                    f"この {self.__class__.__name__} インスタンスは破損している可能性があります。"
                    f"欠落している属性: {missing_attrs}"
                )
    
    def _store_fit_metadata(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """fit時に共通メタデータを保存するヘルパーメソッド"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = len(self.feature_names_in_)
        elif isinstance(X, np.ndarray):
            self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
            self.feature_names_in_ = [f'feature_{i}' for i in range(self.n_features_in_)]
        
        self.is_fitted_ = True
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """変換後の特徴量名を取得（sklearn互換）"""
        if hasattr(self, 'feature_names_in_'):
            return self.feature_names_in_
        return input_features if input_features is not None else []
    
    def _prepare_output(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """出力データの準備（コピーの処理など）"""
        if self.copy:
            if isinstance(X, pd.DataFrame):
                return X.copy()
            elif isinstance(X, np.ndarray):
                return X.copy()
        return X
    
    def _get_feature_names_from_input(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """入力データから特徴量名を取得"""
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        elif isinstance(X, np.ndarray):
            n_features = X.shape[1] if X.ndim > 1 else 1
            return [f'feature_{i}' for i in range(n_features)]
        else:
            return []
    
    def get_fit_info(self) -> dict:
        """fit状態と設定された属性の情報を取得"""
        return {
            'is_fitted': self._is_fitted,
            'fit_attributes': list(self._fit_attributes),
            'n_features_in': getattr(self, 'n_features_in_', None),
            'feature_names_in': getattr(self, 'feature_names_in_', None),
            'fit_duration': self.fit_duration,
            'time_column': self.time_column
        }