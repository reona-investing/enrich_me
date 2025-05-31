"""
特徴量直交化ハンドラー - BasePreprocessor継承版
"""
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
    
    def fit(self, X: pd.DataFrame, y: Optional[any] = None) -> 'FeatureNeutralizer':
        """
        直交化のパラメータを学習（主に列名の保存と検証）
        
        Parameters
        ----------
        X : pd.DataFrame
            入力データ
        y : ignored
            sklearn互換のため
            
        Returns
        -------
        self : FeatureNeutralizer
        """
        # 基本検証
        self._validate_input(X)
        
        # メタデータ保存
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        
        if self.mode == 'specific':
            # 指定列モード
            self.target_features_ = self._ensure_list(self.target_features)
            self.neutralize_features_ = self._ensure_list(self.neutralize_features)
            
            # 列の存在チェック
            missing_target = set(self.target_features_) - set(X.columns)
            missing_neutralize = set(self.neutralize_features_) - set(X.columns)
            
            if missing_target:
                raise ValueError(f"Target features not found in DataFrame: {missing_target}")
            if missing_neutralize:
                raise ValueError(f"Neutralize features not found in DataFrame: {missing_neutralize}")
            
            # 重複チェック
            overlap = set(self.target_features_) & set(self.neutralize_features_)
            if overlap:
                warnings.warn(f"Features appear in both target and neutralize lists: {overlap}")
                
        elif self.mode == 'mutual':
            # 相互直交化モード
            self.target_features_ = X.columns.tolist()
            
            # 数値列のみを対象とする
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < len(X.columns):
                non_numeric = set(X.columns) - set(numeric_cols)
                warnings.warn(f"Non-numeric columns will be skipped: {non_numeric}")
                self.target_features_ = numeric_cols
        
        # データ品質チェック
        self._check_data_quality(X)
        
        return self
    
    def _check_data_quality(self, X: pd.DataFrame) -> None:
        """データ品質をチェック"""
        # 対象列のNaN率をチェック
        if self.mode == 'specific':
            check_cols = list(set(self.target_features_ + self.neutralize_features_))
        else:
            check_cols = self.target_features_
        
        for col in check_cols:
            if col in X.columns:
                nan_ratio = X[col].isnull().sum() / len(X)
                if nan_ratio > 0.5:
                    warnings.warn(f"Column '{col}' has high NaN ratio: {nan_ratio:.2%}")
        
        # 定数列チェック
        for col in check_cols:
            if col in X.columns and X[col].nunique() <= 1:
                warnings.warn(f"Column '{col}' appears to be constant")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        直交化を実行
        
        Parameters
        ----------
        X : pd.DataFrame
            入力データ
            
        Returns
        -------
        X_transformed : pd.DataFrame
            直交化後のデータ
        """
        # 基本検証
        self._check_is_fitted()
        self._validate_input(X)
        
        # 列名の一致チェック
        if not set(self.feature_names_in_).issubset(set(X.columns)):
            missing_cols = set(self.feature_names_in_) - set(X.columns)
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # データを準備
        X_transformed = self._prepare_output(X)
        
        if self.mode == 'specific':
            X_transformed = self._apply_specific_neutralization(X_transformed)
        elif self.mode == 'mutual':
            X_transformed = self._apply_mutual_neutralization(X_transformed)
        
        return X_transformed
    
    def _apply_specific_neutralization(self, X: pd.DataFrame) -> pd.DataFrame:
        """指定列を指定列で直交化"""
        neutralize_data = X[self.neutralize_features_].values
        
        # NaNの処理
        if np.any(np.isnan(neutralize_data)):
            warnings.warn("NaN values in neutralize features. Using available data for orthogonalization.")
        
        for target_col in self.target_features_:
            target_data = X[target_col].values
            
            # 有効なデータのマスクを作成
            valid_mask = ~(np.isnan(target_data) | np.any(np.isnan(neutralize_data), axis=1))
            
            if np.sum(valid_mask) < len(target_data) * 0.5:
                warnings.warn(f"Less than 50% valid data for orthogonalization of '{target_col}'")
            
            if np.sum(valid_mask) > 0:
                # 有効なデータのみで直交化
                orthogonalized_valid = self._orthogonalize_vector(
                    target_data[valid_mask], 
                    neutralize_data[valid_mask]
                )
                
                # 結果を元の配列に戻す
                orthogonalized = target_data.copy()
                orthogonalized[valid_mask] = orthogonalized_valid
                X[target_col] = orthogonalized
            else:
                warnings.warn(f"No valid data for orthogonalization of '{target_col}'. Column unchanged.")
        
        return X
    
    def _apply_mutual_neutralization(self, X: pd.DataFrame) -> pd.DataFrame:
        """全列を互いに直交化"""
        for target_col in self.target_features_:
            # 対象列以外を直交化用特徴量として使用
            other_cols = [col for col in self.target_features_ if col != target_col]
            
            if len(other_cols) > 0:
                target_data = X[target_col].values
                other_data = X[other_cols].values
                
                # 有効なデータのマスクを作成
                valid_mask = ~(np.isnan(target_data) | np.any(np.isnan(other_data), axis=1))
                
                if np.sum(valid_mask) > 0:
                    orthogonalized_valid = self._orthogonalize_vector(
                        target_data[valid_mask], 
                        other_data[valid_mask]
                    )
                    
                    orthogonalized = target_data.copy()
                    orthogonalized[valid_mask] = orthogonalized_valid
                    X[target_col] = orthogonalized
        
        return X
    
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
            'fit_intercept': self.fit_intercept
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

    def demonstrate_usage():
        """使用例のデモンストレーション"""
        print("=== FeatureNeutralizer使用例（BasePreprocessor継承版） ===\n")
        
        # サンプルデータ作成
        df = create_sample_data()
        print("元データの相関行列:")
        print(df.corr().round(6))
        print()
        
        # 1. 指定列を指定列で直交化
        print("1. 列Aを列B,Cで直交化:")
        neutralizer1 = FeatureNeutralizer(
            target_features='A',
            neutralize_features=['B', 'C'],
            mode='specific'
        )
        df_neutralized1 = neutralizer1.fit_transform(df)
        print("処理後のA列と他列の相関:")
        print(df_neutralized1.corr().round(6))

        # 設定情報表示
        print("直交化設定情報:")
        print(neutralizer1.get_neutralization_info())
        print()
        
        # 2. 複数列を複数列で直交化
        print("2. 列A,Dを列B,Cで直交化:")
        neutralizer2 = FeatureNeutralizer(
            target_features=['A', 'D'],
            neutralize_features=['B', 'C'],
            mode='specific',
            fit_intercept=True  # 切片ありで試行
        )
        df_neutralized2 = neutralizer2.fit_transform(df)
        print("処理後の相関（A,Dと他列）:")
        print(df_neutralized2.corr().round(6))

        # 3. 全列を互いに直交化
        print("3. 全列を互いに直交化:")
        neutralizer3 = FeatureNeutralizer(mode='mutual')
        df_neutralized3 = neutralizer3.fit_transform(df)
        print("処理後の相関行列:")
        print(df_neutralized3.corr().round(6))
        print()
        
        # 4. エラーハンドリングのテスト
        print("4. エラーハンドリングテスト:")
        try:
            # 存在しない列を指定
            bad_neutralizer = FeatureNeutralizer(
                target_features='A',
                neutralize_features=['X', 'Y'],
                mode='specific'
            )
            bad_neutralizer.fit(df)
        except ValueError as e:
            print(f"予期されたエラー: {e}")

    demonstrate_usage()