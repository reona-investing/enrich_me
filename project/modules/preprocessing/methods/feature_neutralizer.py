import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Union, List, Optional
import warnings

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
    
    # fitの確認に必要な属性を明示的に定義
    _FIT_REQUIRED_ATTRIBUTES = ['target_features_', 'neutralization_params_']
    
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
        
        # パラメータ検証
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
        elif self.mode == 'mutual':
            if self.target_features is not None or self.neutralize_features is not None:
                warnings.warn("target_features and neutralize_features are ignored when mode='mutual'")
    
    def _ensure_list(self, features: Union[str, List[str]]) -> List[str]:
        """文字列またはリストを確実にリストに変換"""
        if isinstance(features, str):
            return [features]
        return features if features is not None else []
    
    def _orthogonalize_vector(self, target: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """
        targetをbasisに対して直交化
        
        Parameters
        ----------
        target : np.ndarray, shape (n_samples,)
            直交化対象のベクトル
        basis : np.ndarray, shape (n_samples, n_features)
            直交化に使用する特徴量行列
            
        Returns
        -------
        orthogonalized : np.ndarray, shape (n_samples,)
            直交化されたベクトル
        """
        # NaNチェック
        if np.any(np.isnan(target)) or np.any(np.isnan(basis)):
            warnings.warn("NaN values detected in orthogonalization. Results may be unreliable.")
        
        # 線形回帰でtargetをbasisで予測
        lr = LinearRegression(fit_intercept=self.fit_intercept)
        
        try:
            lr.fit(basis, target)
            # 予測値（射影成分）を計算
            projection = lr.predict(basis)
            # 元のベクトルから射影成分を引いて直交成分を取得
            orthogonalized = target - projection
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Linear algebra error in orthogonalization: {e}. Returning original target.")
            return target
        
        return orthogonalized
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[any] = None) -> 'FeatureNeutralizer':
        """
        直交化のパラメータを学習
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            入力データ
        y : ignored
            sklearn互換のため
            
        Returns
        -------
        self : FeatureNeutralizer
        """
        # 基本検証
        self._validate_input(X)
        
        # 共通メタデータを保存
        self._store_fit_metadata(X)
        
        # 特徴量名を取得
        feature_names = self._get_feature_names_from_input(X)
        
        if self.mode == 'specific':
            # 指定列モード
            self.target_features_ = self._ensure_list(self.target_features)
            self.neutralize_features_ = self._ensure_list(self.neutralize_features)
            
            # 列の存在チェック
            missing_target = set(self.target_features_) - set(feature_names)
            missing_neutralize = set(self.neutralize_features_) - set(feature_names)
            
            if missing_target:
                raise ValueError(f"Target features not found: {missing_target}")
            if missing_neutralize:
                raise ValueError(f"Neutralize features not found: {missing_neutralize}")
            
            # 重複チェック
            overlap = set(self.target_features_) & set(self.neutralize_features_)
            if overlap:
                warnings.warn(f"Features appear in both target and neutralize lists: {overlap}")
                
        elif self.mode == 'mutual':
            # 相互直交化モード
            if isinstance(X, pd.DataFrame):
                # DataFrameの場合は数値列のみを対象とする
                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) < len(X.columns):
                    non_numeric = set(X.columns) - set(numeric_cols)
                    warnings.warn(f"Non-numeric columns will be skipped: {non_numeric}")
                self.target_features_ = numeric_cols
            else:
                # numpy arrayの場合は全列を対象
                self.target_features_ = feature_names
        
        # 直交化パラメータを保存
        self.neutralization_params_ = {
            'mode': self.mode,
            'target_features': self.target_features_,
            'neutralize_features': getattr(self, 'neutralize_features_', None),
            'fit_intercept': self.fit_intercept
        }
        
        # データ品質チェック
        self._check_data_quality(X)
        
        return self
    
    def _check_data_quality(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """データ品質をチェック"""
        feature_names = self._get_feature_names_from_input(X)
        
        # 対象列のNaN率をチェック
        if self.mode == 'specific':
            check_cols = list(set(self.target_features_ + self.neutralize_features_))
        else:
            check_cols = self.target_features_
        
        if isinstance(X, pd.DataFrame):
            for col in check_cols:
                if col in X.columns:
                    nan_ratio = X[col].isnull().sum() / len(X)
                    if nan_ratio > 0.5:
                        warnings.warn(f"Column '{col}' has high NaN ratio: {nan_ratio:.2%}")
            
            # 定数列チェック
            for col in check_cols:
                if col in X.columns and X[col].nunique() <= 1:
                    warnings.warn(f"Column '{col}' appears to be constant")
        
        elif isinstance(X, np.ndarray):
            # numpy arrayの場合の品質チェック
            for i, col_name in enumerate(check_cols):
                if i < X.shape[1]:
                    col_data = X[:, i]
                    nan_ratio = np.isnan(col_data).sum() / len(col_data)
                    if nan_ratio > 0.5:
                        warnings.warn(f"Column {i} ('{col_name}') has high NaN ratio: {nan_ratio:.2%}")
                    
                    # 定数列チェック（NaNを除外）
                    valid_data = col_data[~np.isnan(col_data)]
                    if len(valid_data) > 0 and np.all(valid_data == valid_data[0]):
                        warnings.warn(f"Column {i} ('{col_name}') appears to be constant")
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        直交化を実行
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            入力データ
            
        Returns
        -------
        X_transformed : pd.DataFrame or np.ndarray
            直交化後のデータ（入力と同じ型）
        """
        # fit状態をチェック
        self._check_is_fitted()
        
        # 基本検証
        self._validate_input(X)
        
        # 特徴量の一致チェック
        input_feature_names = self._get_feature_names_from_input(X)
        expected_features = set(self.feature_names_in_)
        actual_features = set(input_feature_names)
        
        if not expected_features.issubset(actual_features):
            missing_cols = expected_features - actual_features
            raise ValueError(f"Missing features in input data: {missing_cols}")
        
        # データを準備
        X_transformed = self._prepare_output(X)
        
        if self.mode == 'specific':
            X_transformed = self._apply_specific_neutralization(X_transformed)
        elif self.mode == 'mutual':
            X_transformed = self._apply_mutual_neutralization(X_transformed)
        
        return X_transformed
    
    def _apply_specific_neutralization(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """指定列を指定列で直交化"""
        if isinstance(X, pd.DataFrame):
            neutralize_data = X[self.neutralize_features_].values
            
            for target_col in self.target_features_:
                target_data = X[target_col].values
                orthogonalized = self._orthogonalize_with_validation(target_data, neutralize_data, target_col)
                X[target_col] = orthogonalized
                
        elif isinstance(X, np.ndarray):
            # numpy arrayの場合はインデックスベースで処理
            target_indices = [self.feature_names_in_.index(col) for col in self.target_features_ 
                            if col in self.feature_names_in_]
            neutralize_indices = [self.feature_names_in_.index(col) for col in self.neutralize_features_ 
                                if col in self.feature_names_in_]
            
            neutralize_data = X[:, neutralize_indices]
            
            for target_idx in target_indices:
                target_data = X[:, target_idx]
                target_col_name = self.feature_names_in_[target_idx]
                orthogonalized = self._orthogonalize_with_validation(target_data, neutralize_data, target_col_name)
                X[:, target_idx] = orthogonalized
        
        return X
    
    def _apply_mutual_neutralization(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """全列を互いに直交化"""
        if isinstance(X, pd.DataFrame):
            for target_col in self.target_features_:
                if target_col in X.columns:
                    # 対象列以外を直交化用特徴量として使用
                    other_cols = [col for col in self.target_features_ if col != target_col and col in X.columns]
                    
                    if len(other_cols) > 0:
                        target_data = X[target_col].values
                        other_data = X[other_cols].values
                        orthogonalized = self._orthogonalize_with_validation(target_data, other_data, target_col)
                        X[target_col] = orthogonalized
                        
        elif isinstance(X, np.ndarray):
            target_indices = [self.feature_names_in_.index(col) for col in self.target_features_ 
                            if col in self.feature_names_in_]
            
            for target_idx in target_indices:
                # 対象列以外のインデックスを取得
                other_indices = [idx for idx in target_indices if idx != target_idx]
                
                if len(other_indices) > 0:
                    target_data = X[:, target_idx]
                    other_data = X[:, other_indices]
                    target_col_name = self.feature_names_in_[target_idx]
                    orthogonalized = self._orthogonalize_with_validation(target_data, other_data, target_col_name)
                    X[:, target_idx] = orthogonalized
        
        return X
    
    def _orthogonalize_with_validation(self, target_data: np.ndarray, 
                                     neutralize_data: np.ndarray, 
                                     target_name: str) -> np.ndarray:
        """バリデーション付きの直交化処理"""
        # NaNの処理
        if np.any(np.isnan(neutralize_data)):
            warnings.warn("NaN values in neutralize features. Using available data for orthogonalization.")
        
        # 有効なデータのマスクを作成
        valid_mask = ~(np.isnan(target_data) | np.any(np.isnan(neutralize_data), axis=1))
        
        if np.sum(valid_mask) < len(target_data) * 0.5:
            warnings.warn(f"Less than 50% valid data for orthogonalization of '{target_name}'")
        
        if np.sum(valid_mask) > 0:
            # 有効なデータのみで直交化
            orthogonalized_valid = self._orthogonalize_vector(
                target_data[valid_mask], 
                neutralize_data[valid_mask]
            )
            
            # 結果を元の配列に戻す
            orthogonalized = target_data.copy()
            orthogonalized[valid_mask] = orthogonalized_valid
            return orthogonalized
        else:
            warnings.warn(f"No valid data for orthogonalization of '{target_name}'. Column unchanged.")
            return target_data
    
    def get_neutralization_info(self) -> dict:
        """
        直交化の設定情報を取得
        
        Returns
        -------
        info : dict
            直交化の設定情報
        """
        self._check_is_fitted()
        
        info = {
            'mode': self.mode,
            'n_features_in': self.n_features_in_,
            'feature_names_in': self.feature_names_in_,
            'fit_intercept': self.fit_intercept,
            'neutralization_params': self.neutralization_params_
        }
        
        if self.mode == 'specific':
            info.update({
                'target_features': self.target_features_,
                'neutralize_features': self.neutralize_features_,
                'n_target_features': len(self.target_features_),
                'n_neutralize_features': len(self.neutralize_features_)
            })
        elif self.mode == 'mutual':
            info.update({
                'target_features': self.target_features_,
                'n_target_features': len(self.target_features_)
            })
        
        return info
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """変換後の特徴量名を取得（sklearn互換）"""
        if hasattr(self, 'feature_names_in_'):
            return self.feature_names_in_
        return input_features if input_features is not None else []


# 使用例とテスト用の関数
if __name__ == '__main__':
    def create_sample_data():
        """サンプルデータを作成"""
        np.random.seed(42)
        n_samples = 100
        
        # 相関のあるデータを作成
        X = np.random.randn(n_samples, 3)
        X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)  # X1はX0と相関
        X[:, 2] = X[:, 0] + X[:, 1] + 0.3 * np.random.randn(n_samples)  # X2はX0,X1と相関
        
        df = pd.DataFrame(X, columns=['A', 'B', 'C'])
        
        # 追加の特徴量
        df['D'] = 2 * df['A'] + np.random.randn(n_samples) * 0.1
        df['E'] = df['B'] + df['C'] + np.random.randn(n_samples) * 0.2
        
        return df

    def demonstrate_improved_usage():
        """FeatureNeutralizerの使用例"""
        print("=== FeatureNeutralizer使用例 ===\n")
        
        # サンプルデータ作成
        df = create_sample_data()
        print("元データの相関行列:")
        print(df.corr().round(6))
        print()
        
        # 1. DataFrame + 指定列モード
        print("1. DataFrame - 列Aを列B,Cで直交化:")
        neutralizer1 = FeatureNeutralizer(
            target_features='A',
            neutralize_features=['B', 'C'],
            mode='specific'
        )
        
        # fit前のエラーテスト
        try:
            neutralizer1.get_neutralization_info()
        except ValueError as e:
            print(f"Expected error before fit: {e}")
        
        df_neutralized1 = neutralizer1.fit_transform(df)
        print("処理後のA列と他列の相関:")
        print(df_neutralized1.corr().round(6))
        print("設定情報:", neutralizer1.get_neutralization_info())
        print()
        
        # 2. numpy array + 相互直交化モード
        print("2. Numpy array - 全列を互いに直交化:")
        X_array = df.values
        neutralizer2 = FeatureNeutralizer(mode='mutual')
        X_neutralized = neutralizer2.fit_transform(X_array)
        
        print(f"元データ形状: {X_array.shape}")
        print(f"処理後データ形状: {X_neutralized.shape}")
        print("Feature names:", neutralizer2.get_feature_names_out())
        print()
        
        # 3. 混合データ型のDataFrame
        print("3. 混合データ型のDataFrame:")
        df_mixed = df.copy()
        df_mixed['text_col'] = ['text'] * len(df)
        df_mixed['category'] = pd.Categorical(['A', 'B'] * (len(df)//2))
        
        neutralizer3 = FeatureNeutralizer(mode='mutual')
        df_mixed_result = neutralizer3.fit_transform(df_mixed)
        
        print(f"元データ列: {df_mixed.columns.tolist()}")
        print(f"処理対象列: {neutralizer3.target_features_}")
        print(f"結果データ形状: {df_mixed_result.shape}")

    demonstrate_improved_usage()