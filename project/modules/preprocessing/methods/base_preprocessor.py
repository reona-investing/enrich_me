from abc import ABC, abstractmethod
from typing import Any, Optional, List, Union
import pandas as pd
import numpy as np
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
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> 'BasePreprocessor':
        """
        前処理のパラメータを学習
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
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
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        学習したパラメータを使って変換を実行
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            変換対象データ
            
        Returns
        -------
        X_transformed : pd.DataFrame or np.ndarray
            変換後のデータ
        """
        pass
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        fit と transform を同時に実行
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            学習・変換用データ
        y : Any, optional
            目的変数
            
        Returns
        -------
        X_transformed : pd.DataFrame or np.ndarray
            変換後のデータ
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        入力データの妥当性をチェック
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            チェック対象データ
            
        Raises
        ------
        ValueError
            データが不正な場合
        """
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
    
    def _check_is_fitted(self, attributes: Optional[List[str]] = None) -> None:
        """
        fitが実行済みかチェック
        
        Parameters
        ----------
        attributes : List[str], optional
            チェックすべき属性名のリスト。Noneの場合は標準的な属性をチェック
            
        Raises
        ------
        ValueError
            fitが実行されていない場合
            
        Notes
        -----
        継承クラスでは以下のいずれかの方法でfitの確認を行う：
        1. attributes引数で明示的に指定
        2. デフォルトの属性チェック（推奨）
        3. クラス変数_FIT_REQUIRED_ATTRIBUTESで指定
        
        Examples
        --------
        # 方法1: 明示的指定
        self._check_is_fitted(['pca_', 'n_components_'])
        
        # 方法2: デフォルト（推奨）
        self._check_is_fitted()  # n_features_in_をチェック
        
        # 方法3: クラス変数で指定
        class MyProcessor(BasePreprocessor):
            _FIT_REQUIRED_ATTRIBUTES = ['model_', 'parameters_']
        """
        # 1. 明示的に指定された属性をチェック
        if attributes is not None:
            missing_attrs = [attr for attr in attributes if not hasattr(self, attr)]
            if missing_attrs:
                raise ValueError(
                    f"This {self.__class__.__name__} instance is not fitted yet. "
                    f"Missing attributes: {missing_attrs}"
                )
            return
        
        # 2. クラス変数で指定された属性をチェック
        if hasattr(self.__class__, '_FIT_REQUIRED_ATTRIBUTES'):
            required_attrs = self.__class__._FIT_REQUIRED_ATTRIBUTES
            missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
            if missing_attrs:
                raise ValueError(
                    f"This {self.__class__.__name__} instance is not fitted yet. "
                    f"Missing attributes: {missing_attrs}"
                )
            return
        
        # 3. デフォルトの属性チェック
        # 以下の順序で存在をチェックし、最初に見つかった属性で判定
        default_attrs = [
            'n_features_in_',      # 最も一般的
            'feature_names_in_',   # DataFrame用
            'is_fitted_',          # カスタム属性
        ]
        
        # 少なくとも1つの属性が存在するかチェック
        has_fit_attr = any(hasattr(self, attr) for attr in default_attrs)
        
        if not has_fit_attr:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator. "
                f"Expected one of: {default_attrs}"
            )
    
    def _store_fit_metadata(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        fit時に共通メタデータを保存するヘルパーメソッド
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            学習用データ
            
        Notes
        -----
        継承クラスのfit()メソッド内で呼び出すことを推奨。
        これにより、_check_is_fitted()が適切に動作する。
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = len(self.feature_names_in_)
        elif isinstance(X, np.ndarray):
            self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
            # numpy arrayの場合はgeneric feature names
            self.feature_names_in_ = [f'feature_{i}' for i in range(self.n_features_in_)]
        
        # カスタム属性（明示的なfit状態表示）
        self.is_fitted_ = True
    
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
    
    def _prepare_output(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        出力データの準備（コピーの処理など）
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            元データ
            
        Returns
        -------
        X_output : pd.DataFrame or np.ndarray
            出力用データ
        """
        if self.copy:
            if isinstance(X, pd.DataFrame):
                return X.copy()
            elif isinstance(X, np.ndarray):
                return X.copy()
        return X
    
    def _get_feature_names_from_input(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """
        入力データから特徴量名を取得
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            入力データ
            
        Returns
        -------
        feature_names : List[str]
            特徴量名のリスト
        """
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        elif isinstance(X, np.ndarray):
            n_features = X.shape[1] if X.ndim > 1 else 1
            return [f'feature_{i}' for i in range(n_features)]
        else:
            return []