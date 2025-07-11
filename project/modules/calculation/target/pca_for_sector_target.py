# project/modules/calculation/target/pca_for_sector_target.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Union
from utils.timeseries import Duration
import warnings

from timeseries_data.preprocessing.methods import PCAHandler


class PCAforMultiSectorTarget:
    """
    機械学習用途特化PCA前処理ファサード（簡略化版）
    
    特定用途（ML目的変数前処理）に特化したシンプルなファサードパターン。
    内部でPCAHandlerを使用し、ML特化の前後処理を提供。
    時系列対応により大幅に簡略化。
    
    Parameters
    ----------
    n_components : int
        抽出する主成分の数
    fit_duration : Duration or None, optional
        学習期間を表すDuration。Noneの場合は全期間を使用
    target_column : str, default='Target'
        対象となる列名
    mode : str, default='residuals'
        'residuals': 残差を抽出
        'components': 主成分を抽出
        'transform': PCA変換結果を直接取得
    copy : bool, default=True
        データをコピーするかどうか
    random_state : int, optional
        乱数シード
    """
    
    def __init__(self, 
                 n_components: int,
                 fit_duration: Optional[Duration] = None,
                 target_column: str = 'Target',
                 mode: str = 'residuals',
                 copy: bool = True,
                 random_state: Optional[int] = None):
        
        # ML特化パラメータ
        self.target_column = target_column
        self.copy = copy
        
        # PCAHandlerを初期化（時系列対応版）
        self.pca_handler = PCAHandler(
            n_components=n_components,
            mode=mode,
            copy=copy,
            random_state=random_state,
            fit_duration=fit_duration,
            time_column='Date'
        )
        
        # 内部状態管理
        self._original_sectors = None
        
        # パラメータ検証
        if fit_duration is not None:
            start_dt = pd.to_datetime(fit_duration.start)
            end_dt = pd.to_datetime(fit_duration.end)
            if start_dt > end_dt:
                raise ValueError("fit_duration.start は fit_duration.end よりも前でなければなりません")
    
    def apply_pca(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        ML用途特化のPCA前処理を実行
        
        初回実行時は学習も同時に行い、2回目以降は学習済みパラメータで変換のみ実行。
        BasePreprocessorの時系列機能により大幅に簡略化。
        
        Parameters
        ----------
        X : pd.DataFrame
            処理対象データ（二階層インデックス必須）
            
        Returns
        -------
        X_transformed : pd.DataFrame
            PCA処理後のデータ
        """
        # ML特化の前処理
        df_for_pca = self._prepare_data(X)
        
        if not self.pca_handler._is_fitted:
            # 初回実行：学習 + 変換
            # PCAHandlerが自動的に指定期間でfitして全期間でtransformする
            result_array = self.pca_handler.fit_transform(df_for_pca)
            
            # メタデータを保存
            self._original_sectors = df_for_pca.columns.tolist()
        else:
            # 2回目以降：変換のみ
            result_array = self.pca_handler.transform(df_for_pca)
        
        # ML特化の後処理（DataFrame復元）
        return self._restore_dataframe_format(result_array, df_for_pca)
    
    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """学習・変換用データフレームを準備（ML特化処理）"""
        # データをコピー（必要に応じて）
        if self.copy:
            X = X.copy()
        
        # Target列をアンスタック
        df_for_pca = X[self.target_column].unstack(-1)
        
        # NaNチェックと処理（警告付き）
        if df_for_pca.isnull().any().any():
            warnings.warn(
                "NaN values found in data. They will be forward-filled then back-filled.",
                UserWarning
            )
            df_for_pca = df_for_pca.ffill().bfill()
        
        return df_for_pca
    
    def _restore_dataframe_format(self, transformed_array: np.ndarray, 
                                 original_df: pd.DataFrame) -> pd.DataFrame:
        """変換後の配列をDataFrame形式に復元"""
        # 変換後の配列をDataFrameに変換
        if transformed_array.ndim == 2:
            columns = original_df.columns
        else:
            # 1次元の場合（単一特徴量出力）
            columns = original_df.columns[:1] if len(original_df.columns) > 0 else ['PC1']
            transformed_array = transformed_array.reshape(-1, 1)
        
        result_df = pd.DataFrame(
            transformed_array,
            index=original_df.index,
            columns=columns
        ).sort_index(ascending=True)
        
        # 元のマルチインデックス形式に戻す
        result = result_df.stack().to_frame(self.target_column)
        result.index.names = ['Date', 'Sector']  # インデックス名を設定
        
        return result
    
    # PCAHandlerへの委譲メソッド（簡略化）
    def get_explained_variance_ratio(self) -> np.ndarray:
        """各主成分の寄与率を取得"""
        if not self.pca_handler._is_fitted:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        return self.pca_handler.get_explained_variance_ratio()
    
    def get_cumulative_explained_variance_ratio(self) -> np.ndarray:
        """累積寄与率を取得"""
        if not self.pca_handler._is_fitted:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        return self.pca_handler.get_cumulative_explained_variance_ratio()
    
    def get_components(self) -> pd.DataFrame:
        """主成分ベクトルをDataFrame形式で取得"""
        if not self.pca_handler._is_fitted:
            raise ValueError("PCA has not been applied yet. Call apply_pca() first.")
        components_array = self.pca_handler.get_components()
        
        # ML特化：セクター名を使用してDataFrame化
        component_names = [f'PC_{i+1:02d}' for i in range(components_array.shape[0])]
        sector_names = self._original_sectors or [f'Sector_{i}' for i in range(components_array.shape[1])]
        
        return pd.DataFrame(
            components_array,
            index=component_names,
            columns=sector_names
        )
    
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """変換後の特徴量名を取得"""
        return [self.target_column]
    
    def get_fit_info(self) -> dict:
        """fit状態と設定情報を取得（PCAHandlerの情報も含む）"""
        # PCAHandlerの基本情報を取得
        pca_info = self.pca_handler.get_fit_info()
        
        # ML特化の情報を追加
        ml_info = {
            'target_column': self.target_column,
            'pca_mode': self.pca_handler.mode,
            'n_components': self.pca_handler.n_components,
        }
        
        if self.pca_handler._is_fitted:
            ml_info.update({
                'n_sectors': len(self._original_sectors) if self._original_sectors else 0,
                'original_sectors': self._original_sectors,
                'explained_variance_ratio': self.get_explained_variance_ratio().tolist(),
                'cumulative_variance_ratio': self.get_cumulative_explained_variance_ratio().tolist()
            })
        
        # 統合した情報を返す
        combined_info = {**pca_info, **ml_info}
        return combined_info
    
    @property
    def is_fitted(self) -> bool:
        """fit状態を確認"""
        return self.pca_handler._is_fitted
    
    @property
    def fit_start(self) -> Union[str, pd.Timestamp, None]:
        """fit開始日を取得"""
        if self.pca_handler.fit_duration is None:
            return None
        return self.pca_handler.fit_duration.start
    
    @property
    def fit_end(self) -> Union[str, pd.Timestamp, None]:
        """fit終了日を取得"""
        if self.pca_handler.fit_duration is None:
            return None
        return self.pca_handler.fit_duration.end
    
    # 新機能：PCAHandlerを直接操作したい場合のアクセサ
    @property
    def underlying_pca(self) -> PCAHandler:
        """内部のPCAHandlerインスタンスにアクセス（上級者向け）"""
        return self.pca_handler


# 使用例とテストコード
def create_test_data() -> pd.DataFrame:
    """テスト用のマルチセクターデータを作成"""
    # 日付範囲
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    sectors = ['Tech', 'Finance', 'Healthcare', 'Energy', 'Consumer']
    
    # ランダムデータ（相関のあるセクター間リターン）
    np.random.seed(42)
    n_dates = len(dates)
    n_sectors = len(sectors)
    
    # 共通因子を作成
    market_factor = np.random.normal(0, 0.02, n_dates)
    
    # セクター固有の要因
    sector_data = {}
    for i, sector in enumerate(sectors):
        sector_specific = np.random.normal(0, 0.01, n_dates)
        # セクター間の相関を作成
        correlation_with_market = 0.3 + 0.4 * (i / (n_sectors - 1))
        sector_returns = correlation_with_market * market_factor + sector_specific
        sector_data[sector] = np.cumsum(sector_returns)  # 累積リターン
    
    # マルチインデックスDataFrameを作成
    index_tuples = [(date, sector) for date in dates for sector in sectors]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['Date', 'Sector'])
    
    # データを平坦化
    target_values = []
    for date in dates:
        for sector in sectors:
            target_values.append(sector_data[sector][list(dates).index(date)])
    
    df = pd.DataFrame({
        'Target': target_values
    }, index=multi_index)
    
    return df


def test_pca_facade():
    """PCAファサードクラスのテスト"""
    print("=== PCAforMultiSectorTarget テスト ===\n")
    
    # テストデータ作成
    test_data = create_test_data()
    print(f"テストデータ形状: {test_data.shape}")
    print(f"期間: {test_data.index.get_level_values('Date').min()} to {test_data.index.get_level_values('Date').max()}")
    print(f"セクター: {test_data.index.get_level_values('Sector').unique().tolist()}\n")
    
    # PCAファサードを初期化（2020-2022年でfit）
    pca_facade = PCAforMultiSectorTarget(
        n_components=3,
        fit_duration=Duration(start='2020-01-01', end='2022-12-31'),
        mode='residuals',
        random_state=42
    )
    
    print("1. PCA適用前の状態:")
    print(f"is_fitted: {pca_facade.is_fitted}")
    print(f"fit期間: {pca_facade.fit_start} to {pca_facade.fit_end}\n")
    
    # PCA適用
    print("2. PCA適用中...")
    result = pca_facade.apply_pca(test_data)
    print(f"適用完了: {result.shape}")
    print(f"is_fitted: {pca_facade.is_fitted}\n")
    
    # 結果の確認
    print("3. PCA結果:")
    print(f"寄与率: {pca_facade.get_explained_variance_ratio()}")
    print(f"累積寄与率: {pca_facade.get_cumulative_explained_variance_ratio()}")
    print(f"第1主成分寄与率: {pca_facade.get_explained_variance_ratio()[0]:.3f}\n")
    
    # 主成分ベクトル
    components_df = pca_facade.get_components()
    print("4. 主成分ベクトル:")
    print(components_df)
    print()
    
    # fit情報
    fit_info = pca_facade.get_fit_info()
    print("5. fit情報:")
    for key, value in fit_info.items():
        if isinstance(value, (list, np.ndarray)) and len(value) > 3:
            print(f"  {key}: {type(value).__name__} (length: {len(value)})")
        else:
            print(f"  {key}: {value}")
    print()
    
    # 2回目の実行（変換のみ）
    print("6. 2回目の実行（変換のみ）:")
    result2 = pca_facade.apply_pca(test_data)
    print(f"結果が同じか: {np.allclose(result.values, result2.values)}")
    print()
    
    print("=== テスト完了 ===")


if __name__ == "__main__":
    test_pca_facade()