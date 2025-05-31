import numpy as np
import pandas as pd
from typing import Union, List, Optional

from preprocessing.methods.base_preprocessor import BasePreprocessor


class FeatureNeutralizer(BasePreprocessor):
    """
    特徴量の直交化を行うTransformer
    
    BasePreprocessorを継承し、統一されたインターフェースを提供。
    
    Parameters
    ----------
    target_features : str, list of str, or None
        直交化対象の列名。Noneの場合は全列を互いに直交化
    neutralize_features : str, list of str, or None
        直交化に使用する列名。target_featuresがNoneでない場合に必須
    mode : str, default='mutual'
        'mutual': 全列を互いに直交化
        'specific': 指定列を指定列で直交化
    copy : bool, default=True
        データをコピーするかどうか
    fit_intercept : bool, default=False
        線形回帰で切片を含めるかどうか
    """
    def __init__(self, 
                 target_features: Union[str, List[str], None] = None,
                 neutralize_features: Union[str, List[str], None] = None,
                 mode: str = 'mutual',
                 copy: bool = True,
                 fit_intercept: bool = False):
        super().__init__(copy=copy)
        self.target_features = target_features
        self.neutralize_features = neutralize_features
        self.mode = mode
        self.fit_intercept = fit_intercept
        self._validate_params()
    
    def _validate_params(self) -> None:
        """パラメータの妥当性をチェック"""
        if self.mode not in ['specific', 'mutual']:
            raise ValueError("mode must be 'specific' or 'mutual'")
            
        if self.mode == 'specific':
            if self.target_features is None:
                raise ValueError("target_features must be specified when mode='specific'")
            if self.neutralize_features is None:
                raise ValueError("neutralize_features must be specified when mode='specific'")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[any] = None) -> 'FeatureNeutralizer':
        """直交化のパラメータを学習"""
        # 基本検証
        self._validate_input(X)
        
        # 共通メタデータを保存
        self._store_fit_metadata(X)
        
        # 実装固有の処理...
        feature_names = self._get_feature_names_from_input(X)
        
        if self.mode == 'specific':
            self.target_features_ = self._ensure_list(self.target_features)
            self.neutralize_features_ = self._ensure_list(self.neutralize_features)
        elif self.mode == 'mutual':
            if isinstance(X, pd.DataFrame):
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                self.target_features_ = numeric_cols
            else:
                self.target_features_ = feature_names
        
        # 重要: fit完了をマーク（これだけで_check_is_fitted()が動作する）
        self._mark_as_fitted(
            target_features_=self.target_features_,
            neutralize_features_=getattr(self, 'neutralize_features_', None),
            mode_=self.mode
        )
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """直交化を実行"""
        # シンプルなfit状態チェック
        self._check_is_fitted()
        
        # 基本検証
        self._validate_input(X)
        
        # 変換処理の実装...
        X_transformed = self._prepare_output(X)
        
        # 実際の変換ロジックはここに...
        return X_transformed
    
    def _ensure_list(self, features):
        """ヘルパーメソッド"""
        if isinstance(features, str):
            return [features]
        return features if features is not None else []