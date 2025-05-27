"""
前処理エンジンの基底クラス
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

from data_processing.core.contracts import (
    BaseInputContract, BaseOutputContract
)


class Preprocessor(ABC):
    """前処理の基底クラス"""
    
    def __init__(self, input_contract: Optional[BaseInputContract] = None,
                 output_contract: Optional[BaseOutputContract] = None):
        self.input_contract = input_contract
        self.output_contract = output_contract
        self.is_fitted = False
        self._fit_metadata = {}
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> 'Preprocessor':
        """前処理パラメータを学習"""
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """前処理を適用"""
        pass
    
    def fit_transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """学習と変換を同時実行"""
        return self.fit(df, **kwargs).transform(df)
    
    def validate_input(self, df: pd.DataFrame) -> None:
        """入力データの検証"""
        if self.input_contract:
            self.input_contract.validate(df)
    
    def validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """出力データの検証と整形"""
        if self.output_contract:
            self.output_contract.validate_output(df)
            return self.output_contract.format_output(df)
        return df
    
    def execute(self, df: pd.DataFrame, fit_params: Optional[Dict] = None) -> pd.DataFrame:
        """
        前処理の完全なパイプライン実行
        
        Args:
            df: 入力データフレーム
            fit_params: 学習パラメータ（新規学習する場合）
            
        Returns:
            前処理済みデータフレーム
        """
        self.validate_input(df)
        
        # 必要に応じて学習
        if fit_params is not None or not self.is_fitted:
            self.fit(df, **(fit_params or {}))
        
        # 変換実行
        result = self.transform(df)
        
        # 出力検証・整形
        return self.validate_output(result)
    
    def get_fit_metadata(self) -> Dict[str, Any]:
        """学習時のメタデータを取得"""
        return self._fit_metadata.copy()
    
    def save_model(self, filepath: str) -> None:
        """学習済みモデルを保存"""
        model_data = {
            'preprocessor_class': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'fit_metadata': self._fit_metadata,
            'model_state': self._get_model_state()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> 'Preprocessor':
        """学習済みモデルを読み込み"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        if model_data['preprocessor_class'] != self.__class__.__name__:
            raise ValueError(f"モデルクラスが一致しません: {model_data['preprocessor_class']} != {self.__class__.__name__}")
        
        self.is_fitted = model_data['is_fitted']
        self._fit_metadata = model_data['fit_metadata']
        self._set_model_state(model_data['model_state'])
        
        return self
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """モデルの状態を取得（サブクラスで実装）"""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """モデルの状態を設定（サブクラスで実装）"""
        pass


class NoPreprocessor(Preprocessor):
    """前処理なし（パススルー）"""
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'NoPreprocessor':
        """何もしない"""
        self.is_fitted = True
        self._fit_metadata = {
            'fit_time': datetime.now(),
            'input_shape': df.shape,
            'method': 'passthrough'
        }
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """データをそのまま返す"""
        return df.copy()
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {}
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        pass


class StatisticalPreprocessor(Preprocessor):
    """統計的前処理の基底クラス"""
    
    def __init__(self, target_columns: Optional[list] = None, **kwargs):
        super().__init__(**kwargs)
        self.target_columns = target_columns
        self.fitted_statistics = {}
    
    def _get_target_columns(self, df: pd.DataFrame) -> list:
        """対象カラムを取得"""
        if self.target_columns is None:
            # 数値カラムを自動選択
            return df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            return self.target_columns
    
    def _record_fit_statistics(self, df: pd.DataFrame) -> None:
        """学習時の統計情報を記録"""
        target_cols = self._get_target_columns(df)
        
        self.fitted_statistics = {}
        for col in target_cols:
            if col in df.columns:
                series = df[col].dropna()
                self.fitted_statistics[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75)
                }


class TimeSeriesPreprocessor(StatisticalPreprocessor):
    """時系列データ用前処理の基底クラス"""
    
    def __init__(self, group_column: str = 'Sector', **kwargs):
        super().__init__(**kwargs)
        self.group_column = group_column
    
    def _apply_grouped_preprocessing(self, df: pd.DataFrame, 
                                   preprocess_func, **kwargs) -> pd.DataFrame:
        """グループ別に前処理を適用"""
        if self.group_column in df.index.names:
            # マルチインデックスの場合
            return df.groupby(level=self.group_column).apply(
                lambda x: preprocess_func(x, **kwargs)
            )
        elif self.group_column in df.columns:
            # 通常のカラムの場合
            return df.groupby(self.group_column).apply(
                lambda x: preprocess_func(x, **kwargs)
            ).reset_index(level=0, drop=True)
        else:
            # グループ列がない場合は全体に適用
            return preprocess_func(df, **kwargs)


class WindowedPreprocessor(TimeSeriesPreprocessor):
    """ウィンドウベース前処理の基底クラス"""
    
    def __init__(self, window_size: int, min_periods: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.min_periods = min_periods or window_size
    
    def _apply_windowed_preprocessing(self, series: pd.Series, 
                                    preprocess_func, **kwargs) -> pd.Series:
        """ウィンドウベースで前処理を適用"""
        return series.rolling(
            window=self.window_size,
            min_periods=self.min_periods
        ).apply(preprocess_func, **kwargs)


class TrainTestSplitPreprocessor(Preprocessor):
    """訓練・テスト期間分割前処理の基底クラス"""
    
    def __init__(self, train_start: datetime, train_end: datetime, **kwargs):
        super().__init__(**kwargs)
        self.train_start = train_start
        self.train_end = train_end
        self.train_data = None
        self.test_data = None
    
    def _split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """訓練・テストデータに分割"""
        if 'Date' in df.index.names:
            date_index = df.index.get_level_values('Date')
        elif df.index.dtype == 'datetime64[ns]':
            date_index = df.index
        else:
            raise ValueError("Date情報が見つかりません")
        
        train_mask = (date_index >= self.train_start) & (date_index <= self.train_end)
        test_mask = date_index > self.train_end
        
        train_data = df[train_mask]
        test_data = df[test_mask]
        
        return train_data, test_data
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'TrainTestSplitPreprocessor':
        """訓練期間のデータで学習"""
        self.train_data, self.test_data = self._split_train_test(df)
        
        # サブクラスで実装される学習処理
        self._fit_on_train_data(self.train_data, **kwargs)
        
        self.is_fitted = True
        self._fit_metadata = {
            'fit_time': datetime.now(),
            'train_start': self.train_start,
            'train_end': self.train_end,
            'train_shape': self.train_data.shape,
            'test_shape': self.test_data.shape if self.test_data is not None else None
        }
        
        return self
    
    @abstractmethod
    def _fit_on_train_data(self, train_data: pd.DataFrame, **kwargs) -> None:
        """訓練データでの学習処理（サブクラスで実装）"""
        pass


class MultiStepPreprocessor(Preprocessor):
    """複数段階の前処理を組み合わせる基底クラス"""
    
    def __init__(self, preprocessors: list, **kwargs):
        super().__init__(**kwargs)
        self.preprocessors = preprocessors
        self.step_results = []
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'MultiStepPreprocessor':
        """各段階の前処理を順次学習"""
        current_df = df
        
        for i, preprocessor in enumerate(self.preprocessors):
            step_params = kwargs.get(f'step_{i}_params', {})
            preprocessor.fit(current_df, **step_params)
            current_df = preprocessor.transform(current_df)
        
        self.is_fitted = True
        self._fit_metadata = {
            'fit_time': datetime.now(),
            'num_steps': len(self.preprocessors),
            'step_classes': [p.__class__.__name__ for p in self.preprocessors]
        }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """各段階の前処理を順次適用"""
        if not self.is_fitted:
            raise ValueError("前処理が学習されていません。fit()を先に実行してください。")
        
        current_df = df
        self.step_results = []
        
        for preprocessor in self.preprocessors:
            current_df = preprocessor.transform(current_df)
            self.step_results.append(current_df.copy())
        
        return current_df
    
    def get_step_result(self, step_index: int) -> pd.DataFrame:
        """指定したステップの結果を取得"""
        if not self.step_results:
            raise ValueError("transform()が実行されていません")
        
        if step_index >= len(self.step_results):
            raise ValueError(f"ステップインデックスが範囲外です: {step_index}")
        
        return self.step_results[step_index]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """各前処理のモデル状態を保存"""
        states = {}
        for i, preprocessor in enumerate(self.preprocessors):
            states[f'step_{i}'] = preprocessor._get_model_state()
        return states
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """各前処理のモデル状態を復元"""
        for i, preprocessor in enumerate(self.preprocessors):
            if f'step_{i}' in state:
                preprocessor._set_model_state(state[f'step_{i}'])
                preprocessor.is_fitted = True


class ConditionalPreprocessor(Preprocessor):
    """条件付き前処理の基底クラス"""
    
    def __init__(self, condition_func, true_preprocessor: Preprocessor, 
                 false_preprocessor: Optional[Preprocessor] = None, **kwargs):
        super().__init__(**kwargs)
        self.condition_func = condition_func
        self.true_preprocessor = true_preprocessor
        self.false_preprocessor = false_preprocessor or NoPreprocessor()
    
    def fit(self, df: pd.DataFrame, **kwargs) -> 'ConditionalPreprocessor':
        """条件に応じて適切な前処理を学習"""
        if self.condition_func(df):
            self.active_preprocessor = self.true_preprocessor
        else:
            self.active_preprocessor = self.false_preprocessor
        
        self.active_preprocessor.fit(df, **kwargs)
        self.is_fitted = True
        
        self._fit_metadata = {
            'fit_time': datetime.now(),
            'condition_result': self.condition_func(df),
            'active_preprocessor': self.active_preprocessor.__class__.__name__
        }
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """アクティブな前処理を適用"""
        if not self.is_fitted:
            raise ValueError("前処理が学習されていません。fit()を先に実行してください。")
        
        return self.active_preprocessor.transform(df)
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'active_preprocessor_class': self.active_preprocessor.__class__.__name__,
            'active_preprocessor_state': self.active_preprocessor._get_model_state()
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        # 条件判定を再実行して適切な前処理を選択
        # 実際の実装では、より詳細な状態管理が必要
        if hasattr(self, 'active_preprocessor'):
            self.active_preprocessor._set_model_state(state['active_preprocessor_state'])
            self.active_preprocessor.is_fitted = True