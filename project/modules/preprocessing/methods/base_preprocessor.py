from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union, Set
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
    fit_start : str, pd.Timestamp, or None, default=None
        fitに使用する開始日時。Noneの場合は全期間を使用
    fit_end : str, pd.Timestamp, or None, default=None
        fitに使用する終了日時。Noneの場合は全期間を使用
    time_column : str or None, default=None
        時間列の名前。Noneの場合はindexを時間として使用
    """
    
    def __init__(self, 
                 copy: bool = True,
                 fit_start: Union[str, pd.Timestamp, None] = None,
                 fit_end: Union[str, pd.Timestamp, None] = None,
                 time_column: Optional[str] = None):
        self.copy = copy
        self.fit_start = fit_start
        self.fit_end = fit_end
        self.time_column = time_column
        self._is_fitted = False  # 内部的なfit状態管理
        self._fit_attributes: Set[str] = set()  # fit時に設定された属性を記録
    
    def _filter_data_by_time(self, X: Union[pd.DataFrame, np.ndarray], 
                           start: Union[str, pd.Timestamp, None] = None,
                           end: Union[str, pd.Timestamp, None] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        指定された期間でデータをフィルタリング
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            入力データ
        start : str, pd.Timestamp, or None
            開始日時
        end : str, pd.Timestamp, or None
            終了日時
            
        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            フィルタリングされたデータ
        """
        if isinstance(X, np.ndarray):
            # numpy配列の場合はフィルタリングできないのでそのまま返す
            return X
            
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Time filtering is only supported for pandas DataFrame")
        
        # 開始・終了日時が指定されていない場合はそのまま返す
        if start is None and end is None:
            return X
        
        # 時間列を特定
        if self.time_column is not None:
            if self.time_column not in X.columns:
                raise ValueError(f"Time column '{self.time_column}' not found in DataFrame")
            time_index = pd.to_datetime(X[self.time_column])
        else:
            # indexを時間として使用
            time_index = pd.to_datetime(X.index)
        
        # フィルタリング条件を作成
        mask = pd.Series(True, index=X.index)
        
        if start is not None:
            start_dt = pd.to_datetime(start)
            mask = mask & (time_index >= start_dt)
        
        if end is not None:
            end_dt = pd.to_datetime(end)
            mask = mask & (time_index <= end_dt)
        
        filtered_data = X[mask]
        
        if filtered_data.empty:
            raise ValueError(f"No data found in the specified time range: {start} to {end}")
        
        return filtered_data
    
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> 'BasePreprocessor':
        """前処理のパラメータを学習"""
        pass
    
    @abstractmethod
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """学習したパラメータを使って変換を実行"""
        pass
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> Union[pd.DataFrame, np.ndarray]:
        """fit と transform を同時に実行"""
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """入力データの妥当性をチェック"""
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("Input DataFrame is empty")
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Input numpy array is empty")
            if X.ndim < 2:
                raise ValueError("Input numpy array must be at least 2-dimensional")
        else:
            raise ValueError("Input must be a pandas DataFrame or numpy array")
    
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
                f"This {self.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator."
            )
        
        # 追加属性のチェック（必要に応じて）
        if additional_attributes:
            missing_attrs = [attr for attr in additional_attributes if not hasattr(self, attr)]
            if missing_attrs:
                raise ValueError(
                    f"This {self.__class__.__name__} instance appears to be corrupted. "
                    f"Missing expected attributes: {missing_attrs}"
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
            'fit_start': self.fit_start,
            'fit_end': self.fit_end,
            'time_column': self.time_column
        }