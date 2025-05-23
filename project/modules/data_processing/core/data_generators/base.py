"""
データ生成器の基底クラス
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime


class DataGenerator(ABC):
    """データ生成の基底クラス"""
    
    def __init__(self):
        self._generation_metadata = {}
    
    @abstractmethod
    def generate(self, input_data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """
        データを生成する
        
        Args:
            input_data: 生成に必要な入力データ
            **kwargs: 追加パラメータ
            
        Returns:
            生成されたデータフレーム
        """
        pass
    
    @property
    @abstractmethod
    def output_data_type(self) -> str:
        """出力データの型"""
        pass
    
    @property
    @abstractmethod
    def required_input_data(self) -> List[str]:
        """必要な入力データのキー一覧"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """入力データの検証"""
        missing_keys = set(self.required_input_data) - set(input_data.keys())
        if missing_keys:
            raise ValueError(f"必要な入力データが不足しています: {missing_keys}")
    
    def get_generation_metadata(self) -> Dict[str, Any]:
        """生成メタデータを取得"""
        return self._generation_metadata.copy()
    
    def _record_generation_metadata(self, input_data: Dict[str, Any], 
                                  output_df: pd.DataFrame, **kwargs) -> None:
        """生成メタデータを記録"""
        self._generation_metadata = {
            'generation_time': datetime.now(),
            'output_shape': output_df.shape,
            'output_data_type': self.output_data_type,
            'input_keys': list(input_data.keys()),
            'parameters': kwargs
        }