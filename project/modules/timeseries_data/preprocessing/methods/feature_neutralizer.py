import numpy as np
import pandas as pd
from typing import Union, List, Optional
from utils.timeseries import Duration
from sklearn.linear_model import LinearRegression

from .base_preprocessor import BasePreprocessor


class FeatureNeutralizer(BasePreprocessor):
    """
    特徴量の直交化を行うTransformer
    
    BasePreprocessorを継承し、統一されたインターフェースを提供。
    指定期間でfitし、全期間でtransformすることが可能。
    
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
    fit_duration : Duration or None, default=None
        fitに使用する期間
    time_column : str
        時間列の名前
    """
    def __init__(self,
                 *,
                 target_features: Union[str, List[str], None] = None,
                 neutralize_features: Union[str, List[str], None] = None,
                 mode: str = 'mutual',
                 copy: bool = True,
                 fit_intercept: bool = False,
                 fit_duration: Optional[Duration] = None,
                 time_column: str):
        super().__init__(copy=copy, fit_duration=fit_duration, time_column=time_column)
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
    
    def fit(self, X: pd.DataFrame, y: Optional[any] = None) -> 'FeatureNeutralizer':
        """直交化のパラメータを学習（指定期間のデータを使用）"""
        # 基本検証
        self._validate_input(X)
        
        # 指定期間でデータをフィルタリング
        if self.fit_duration is not None:
            X_fit = self.fit_duration.extract_from_df(X, self.time_column)
            if X_fit.empty:
                raise ValueError(
                    f"指定された期間({self.fit_duration.start}～{self.fit_duration.end})のデータが存在しません。"
                )
        else:
            X_fit = X
        
        # 共通メタデータを保存（元のデータで）
        self._store_fit_metadata(X)
        
        # 実装固有の処理...
        feature_names = self._get_feature_names_from_input(X)
        
        if self.mode == 'specific':
            self.target_features_ = self._ensure_list(self.target_features)
            self.neutralize_features_ = self._ensure_list(self.neutralize_features)
            
            # 指定期間のデータで回帰係数を学習
            self._fit_regression_coefficients(X_fit)
            
        elif self.mode == 'mutual':
            if isinstance(X, pd.DataFrame):
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                # 時間列は除外
                if self.time_column and self.time_column in numeric_cols:
                    numeric_cols.remove(self.time_column)
                self.target_features_ = numeric_cols
            else:
                self.target_features_ = feature_names
            
            # 相互直交化の場合の学習処理
            self._fit_mutual_neutralization(X_fit)
        
        # 重要: fit完了をマーク
        self._mark_as_fitted(
            target_features_=self.target_features_,
            neutralize_features_=getattr(self, 'neutralize_features_', None),
            mode_=self.mode
        )
        
        return self
    
    def _fit_regression_coefficients(self, X_fit: Union[pd.DataFrame, np.ndarray]) -> None:
        """特定の直交化のための回帰係数を学習"""
        self.regression_models_ = {}
        
        if isinstance(X_fit, pd.DataFrame):
            for target_col in self.target_features_:
                if target_col not in X_fit.columns:
                    continue
                    
                # 中和用特徴量を取得
                neutralize_cols = [col for col in self.neutralize_features_ if col in X_fit.columns and col != target_col]
                if not neutralize_cols:
                    continue
                
                # 回帰モデルを学習
                X_neutralize = X_fit[neutralize_cols].values
                y_target = X_fit[target_col].values
                
                # 欠損値を除去
                mask = ~(np.isnan(X_neutralize).any(axis=1) | np.isnan(y_target))
                if mask.sum() == 0:
                    continue
                
                reg = LinearRegression(fit_intercept=self.fit_intercept)
                reg.fit(X_neutralize[mask], y_target[mask])
                self.regression_models_[target_col] = {
                    'model': reg,
                    'neutralize_cols': neutralize_cols
                }
    
    def _fit_mutual_neutralization(self, X_fit: Union[pd.DataFrame, np.ndarray]) -> None:
        """相互直交化のためのパラメータを学習"""
        if isinstance(X_fit, pd.DataFrame):
            # 数値列のみで相関行列を計算
            numeric_data = X_fit[self.target_features_].select_dtypes(include=[np.number])
            self.correlation_matrix_ = numeric_data.corr()
            self.means_ = numeric_data.mean()
            self.stds_ = numeric_data.std()
        else:
            # numpy配列の場合
            self.correlation_matrix_ = np.corrcoef(X_fit.T)
            self.means_ = np.mean(X_fit, axis=0)
            self.stds_ = np.std(X_fit, axis=0)
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """直交化を実行（全期間のデータに適用）"""
        # シンプルなfit状態チェック
        self._check_is_fitted()
        
        # 基本検証
        self._validate_input(X)
        
        # 変換処理の実装...
        X_transformed = self._prepare_output(X)
        
        if self.mode == 'specific':
            X_transformed = self._apply_specific_neutralization(X_transformed)
        elif self.mode == 'mutual':
            X_transformed = self._apply_mutual_neutralization(X_transformed)
        
        return X_transformed
    
    def _apply_specific_neutralization(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """特定の直交化を適用"""
        if isinstance(X, pd.DataFrame):
            for target_col, reg_info in self.regression_models_.items():
                if target_col not in X.columns:
                    continue
                
                model = reg_info['model']
                neutralize_cols = reg_info['neutralize_cols']
                
                # 中和用特徴量が存在するかチェック
                available_cols = [col for col in neutralize_cols if col in X.columns]
                if not available_cols:
                    continue
                
                # 予測と残差計算
                X_neutralize = X[available_cols].values
                mask = ~np.isnan(X_neutralize).any(axis=1)
                
                if mask.sum() > 0:
                    predictions = model.predict(X_neutralize[mask])
                    X.loc[mask, target_col] = X.loc[mask, target_col] - predictions
        
        return X
    
    def _apply_mutual_neutralization(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """相互直交化を適用（簡単な実装例）"""
        if isinstance(X, pd.DataFrame):
            # グラム・シュミット過程の簡易版
            for i, col in enumerate(self.target_features_):
                if col not in X.columns:
                    continue
                
                for j, other_col in enumerate(self.target_features_[:i]):
                    if other_col not in X.columns:
                        continue
                    
                    # 射影を計算して除去
                    corr_coef = self.correlation_matrix_.loc[col, other_col]
                    if not np.isnan(corr_coef):
                        X[col] = X[col] - corr_coef * X[other_col]
        
        return X
    
    def _ensure_list(self, features):
        """ヘルパーメソッド"""
        if isinstance(features, str):
            return [features]
        return features if features is not None else []