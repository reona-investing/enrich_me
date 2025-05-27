"""
出力データの契約を定義するモジュール
"""
from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Tuple
import pandas as pd
import numpy as np


class OutputDataContract(Protocol):
    """出力データの契約を定義するプロトコル"""
    column_naming_convention: str
    index_structure: Dict[str, Any]
    value_ranges: Dict[str, Tuple[float, float]]
    
    def validate_output(self, df: pd.DataFrame) -> bool:
        """出力データが契約を満たすかを検証"""
        ...
    
    def format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """出力データを適切な形式に整形"""
        ...


class BaseOutputContract(ABC):
    """出力データ契約の基底クラス"""
    
    @property
    @abstractmethod
    def column_naming_convention(self) -> str:
        """カラム命名規則"""
        pass
    
    @property
    @abstractmethod
    def index_structure(self) -> Dict[str, Any]:
        """インデックス構造の要件"""
        pass
    
    @property
    @abstractmethod
    def value_ranges(self) -> Dict[str, Tuple[float, float]]:
        """各カラムの値の範囲"""
        pass
    
    def validate_output(self, df: pd.DataFrame) -> bool:
        """出力データが契約を満たすかを検証"""
        return (
            self._validate_index_structure(df) and
            self._validate_value_ranges(df) and
            self._validate_data_quality(df)
        )
    
    def format_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """出力データを適切な形式に整形"""
        formatted_df = df.copy()
        
        # インデックスをソート
        formatted_df = formatted_df.sort_index()
        
        # 数値精度を調整
        numeric_columns = formatted_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            formatted_df[col] = formatted_df[col].round(8)
        
        return formatted_df
    
    def _validate_index_structure(self, df: pd.DataFrame) -> bool:
        """インデックス構造を検証"""
        if 'levels' in self.index_structure:
            expected_levels = self.index_structure['levels']
            if len(expected_levels) != df.index.nlevels:
                raise ValueError(
                    f"インデックスレベル数が不正です。期待値: {len(expected_levels)}, "
                    f"実際: {df.index.nlevels}"
                )
        return True
    
    def _validate_value_ranges(self, df: pd.DataFrame) -> bool:
        """値の範囲を検証"""
        for column, (min_val, max_val) in self.value_ranges.items():
            if column in df.columns:
                col_min = df[column].min()
                col_max = df[column].max()
                
                if not pd.isna(col_min) and col_min < min_val:
                    print(f"警告: カラム '{column}' の最小値が範囲外です。"
                          f"最小値: {col_min}, 期待最小値: {min_val}")
                
                if not pd.isna(col_max) and col_max > max_val:
                    print(f"警告: カラム '{column}' の最大値が範囲外です。"
                          f"最大値: {col_max}, 期待最大値: {max_val}")
        return True
    
    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """データ品質を検証"""
        # 無限大値のチェック
        inf_columns = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_columns.append(col)
        
        if inf_columns:
            print(f"警告: 無限大値が含まれています: {inf_columns}")
        
        # 欠損値の比率チェック
        missing_ratios = df.isnull().mean()
        high_missing = missing_ratios[missing_ratios > 0.5]
        
        if not high_missing.empty:
            print(f"警告: 欠損値が50%を超えるカラムがあります: {high_missing.to_dict()}")
        
        return True


class TargetDataContract(BaseOutputContract):
    """目的変数出力の契約"""
    
    def __init__(self, target_column: str = 'Target'):
        self.target_column = target_column
    
    @property
    def column_naming_convention(self) -> str:
        return self.target_column
    
    @property
    def index_structure(self) -> Dict[str, Any]:
        return {
            'levels': ['Date', 'Sector'],
            'types': [pd.Timestamp, str]
        }
    
    @property
    def value_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            self.target_column: (-1.0, float('inf'))  # リターンは-100%以上
        }


class FeatureDataContract(BaseOutputContract):
    """特徴量出力の契約"""
    
    @property
    def column_naming_convention(self) -> str:
        return "{name}_{period}_{metric}"
    
    @property
    def index_structure(self) -> Dict[str, Any]:
        return {
            'levels': ['Date', 'Sector'],
            'types': [pd.Timestamp, str]
        }
    
    @property
    def value_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            # リターン系は-100%以上
            '_return': (-1.0, float('inf')),
            '_mom': (-1.0, float('inf')),
            
            # ボラティリティは0以上
            '_vola': (0.0, float('inf')),
            
            # ランキングは1以上
            '_rank': (1.0, float('inf')),
            
            # その他の特徴量
            'MarketCapAtClose': (0.0, float('inf')),
            'ForecastEPS': (-float('inf'), float('inf')),
            'Sector_cat': (0.0, float('inf'))
        }
    
    def _validate_value_ranges(self, df: pd.DataFrame) -> bool:
        """特徴量特有の値範囲検証"""
        for column in df.columns:
            # カラム名の suffix に基づいて検証
            for suffix, (min_val, max_val) in self.value_ranges.items():
                if column.endswith(suffix):
                    col_data = df[column].dropna()
                    if not col_data.empty:
                        col_min = col_data.min()
                        col_max = col_data.max()
                        
                        if col_min < min_val:
                            print(f"警告: カラム '{column}' の最小値が範囲外です。"
                                  f"最小値: {col_min}, 期待最小値: {min_val}")
                        
                        if max_val != float('inf') and col_max > max_val:
                            print(f"警告: カラム '{column}' の最大値が範囲外です。"
                                  f"最大値: {col_max}, 期待最大値: {max_val}")
                    break
        
        return True


class IndexDataContract(BaseOutputContract):
    """インデックスデータ出力の契約"""
    
    @property
    def column_naming_convention(self) -> str:
        return "{name}_price_data"
    
    @property
    def index_structure(self) -> Dict[str, Any]:
        return {
            'levels': ['Date', 'Sector'],
            'types': [pd.Timestamp, str]
        }
    
    @property
    def value_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'Open': (0.0, float('inf')),
            'High': (0.0, float('inf')),
            'Low': (0.0, float('inf')),
            'Close': (0.0, float('inf')),
            '1d_return': (-1.0, float('inf')),
            'MarketCapClose': (0.0, float('inf'))
        }


class PCADataContract(BaseOutputContract):
    """PCA処理結果の契約"""
    
    def __init__(self, n_components: int, output_type: str = 'residuals'):
        self.n_components = n_components
        self.output_type = output_type  # 'residuals' or 'components'
    
    @property
    def column_naming_convention(self) -> str:
        if self.output_type == 'components':
            return "PC_{component_index:02d}"
        else:  # residuals
            return "Target"  # 残差の場合は元のカラム名を保持
    
    @property
    def index_structure(self) -> Dict[str, Any]:
        return {
            'levels': ['Date', 'Sector'],
            'types': [pd.Timestamp, str]
        }
    
    @property
    def value_ranges(self) -> Dict[str, Tuple[float, float]]:
        if self.output_type == 'components':
            # 主成分は標準化されているので通常-3〜3程度
            return {f'PC_{i:02d}': (-5.0, 5.0) for i in range(self.n_components)}
        else:  # residuals
            # 残差は元データの範囲内
            return {'Target': (-1.0, float('inf'))}


class FactorDataContract(BaseOutputContract):
    """ファクターデータ出力の契約"""
    
    @property
    def column_naming_convention(self) -> str:
        return "{factor_name}_{metric}"
    
    @property
    def index_structure(self) -> Dict[str, Any]:
        return {
            'levels': ['Date', 'Factor'],
            'types': [pd.Timestamp, str]
        }
    
    @property
    def value_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            '_return': (-1.0, float('inf')),
            '_vol': (0.0, float('inf')),
            '_sharpe': (-float('inf'), float('inf'))
        }


# ファクトリー関数
def get_output_contract(data_type: str, **kwargs) -> BaseOutputContract:
    """データ型に応じた出力契約を取得"""
    contracts = {
        'target': TargetDataContract,
        'features': FeatureDataContract,
        'index': IndexDataContract,
        'pca': PCADataContract,
        'factor': FactorDataContract
    }
    
    if data_type not in contracts:
        raise ValueError(f"未対応のデータ型です: {data_type}")
    
    contract_class = contracts[data_type]
    
    if data_type == 'target':
        return contract_class(kwargs.get('target_column', 'Target'))
    elif data_type == 'pca':
        return contract_class(
            kwargs.get('n_components', 1),
            kwargs.get('output_type', 'residuals')
        )
    else:
        return contract_class()