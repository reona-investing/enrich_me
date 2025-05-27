"""
入力データの契約を定義するモジュール
"""
from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any
import pandas as pd


class InputDataContract(Protocol):
    """入力データの契約を定義するプロトコル"""
    required_columns: List[str]
    index_requirements: Dict[str, Any]
    data_types: Dict[str, type]
    
    def validate(self, df: pd.DataFrame) -> bool:
        """データが契約を満たすかを検証"""
        ...


class BaseInputContract(ABC):
    """入力データ契約の基底クラス"""
    
    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """必須カラム名のリスト"""
        pass
    
    @property
    @abstractmethod
    def index_requirements(self) -> Dict[str, Any]:
        """インデックスの要件"""
        pass
    
    @property
    @abstractmethod
    def data_types(self) -> Dict[str, type]:
        """データ型の要件"""
        pass
    
    def validate(self, df: pd.DataFrame) -> bool:
        """データが契約を満たすかを検証"""
        return (
            self._validate_columns(df) and
            self._validate_index(df) and
            self._validate_data_types(df)
        )
    
    def _validate_columns(self, df: pd.DataFrame) -> bool:
        """必須カラムの存在を確認"""
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"必須カラムが不足しています: {missing_columns}")
        return True
    
    def _validate_index(self, df: pd.DataFrame) -> bool:
        """インデックスの要件を確認"""
        if 'levels' in self.index_requirements:
            expected_levels = self.index_requirements['levels']
            if len(expected_levels) != df.index.nlevels:
                raise ValueError(
                    f"インデックスレベル数が不正です。期待値: {len(expected_levels)}, "
                    f"実際: {df.index.nlevels}"
                )
        return True
    
    def _validate_data_types(self, df: pd.DataFrame) -> bool:
        """データ型の要件を確認"""
        for column, expected_type in self.data_types.items():
            if column in df.columns:
                if not df[column].dtype.type == expected_type:
                    print(f"警告: カラム '{column}' のデータ型が期待値と異なります。"
                          f"期待値: {expected_type}, 実際: {df[column].dtype}")
        return True


class PriceDataContract(BaseInputContract):
    """価格データの契約"""
    
    @property
    def required_columns(self) -> List[str]:
        return ['Open', 'High', 'Low', 'Close']
    
    @property
    def index_requirements(self) -> Dict[str, Any]:
        return {
            'levels': ['Date'],
            'types': [pd.Timestamp]
        }
    
    @property
    def data_types(self) -> Dict[str, type]:
        return {
            'Open': float,
            'High': float,
            'Low': float,
            'Close': float,
            'Volume': float
        }


class SectorPriceDataContract(BaseInputContract):
    """セクター価格データの契約"""
    
    @property
    def required_columns(self) -> List[str]:
        return ['Open', 'High', 'Low', 'Close']
    
    @property
    def index_requirements(self) -> Dict[str, Any]:
        return {
            'levels': ['Date', 'Sector'],
            'types': [pd.Timestamp, str]
        }
    
    @property
    def data_types(self) -> Dict[str, type]:
        return {
            'Open': float,
            'High': float,
            'Low': float,
            'Close': float
        }


class IndexDataContract(BaseInputContract):
    """インデックスデータの契約"""
    
    @property
    def required_columns(self) -> List[str]:
        return ['Close']
    
    @property
    def index_requirements(self) -> Dict[str, Any]:
        return {
            'levels': ['Date'],
            'types': [pd.Timestamp]
        }
    
    @property
    def data_types(self) -> Dict[str, type]:
        return {
            'Close': float
        }


class FinancialDataContract(BaseInputContract):
    """財務データの契約"""
    
    @property
    def required_columns(self) -> List[str]:
        return ['Code', 'Date', 'ForecastEPS']
    
    @property
    def index_requirements(self) -> Dict[str, Any]:
        return {
            'levels': [],  # インデックス要件なし
            'types': []
        }
    
    @property
    def data_types(self) -> Dict[str, type]:
        return {
            'Code': str,
            'Date': pd.Timestamp,
            'ForecastEPS': float
        }


class ReturnDataContract(BaseInputContract):
    """リターンデータの契約（目的変数や特徴量用）"""
    
    def __init__(self, target_column: str = 'Target'):
        self.target_column = target_column
    
    @property
    def required_columns(self) -> List[str]:
        return [self.target_column]
    
    @property
    def index_requirements(self) -> Dict[str, Any]:
        return {
            'levels': ['Date', 'Sector'],
            'types': [pd.Timestamp, str]
        }
    
    @property
    def data_types(self) -> Dict[str, type]:
        return {
            self.target_column: float
        }


class MultiIndexReturnDataContract(BaseInputContract):
    """マルチインデックスリターンデータの契約"""
    
    @property
    def required_columns(self) -> List[str]:
        return []  # カラムは動的に決定される
    
    @property
    def index_requirements(self) -> Dict[str, Any]:
        return {
            'levels': ['Date', 'Sector'],
            'types': [pd.Timestamp, str]
        }
    
    @property
    def data_types(self) -> Dict[str, type]:
        return {}  # データ型は動的に決定される
    
    def validate(self, df: pd.DataFrame) -> bool:
        """マルチインデックスの特別な検証"""
        if df.index.nlevels != 2:
            raise ValueError("インデックスは2階層である必要があります")
        
        # 数値データであることを確認
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"カラム '{column}' は数値型である必要があります")
        
        return True


# ファクトリー関数
def get_input_contract(data_type: str, **kwargs) -> BaseInputContract:
    """データ型に応じた入力契約を取得"""
    contracts = {
        'price': PriceDataContract,
        'sector_price': SectorPriceDataContract,
        'index': IndexDataContract,
        'financial': FinancialDataContract,
        'return': ReturnDataContract,
        'multi_return': MultiIndexReturnDataContract
    }
    
    if data_type not in contracts:
        raise ValueError(f"未対応のデータ型です: {data_type}")
    
    contract_class = contracts[data_type]
    if data_type == 'return':
        return contract_class(kwargs.get('target_column', 'Target'))
    else:
        return contract_class()