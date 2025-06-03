import os
import pickle
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseDataComponent(ABC):
    """データコンポーネントの抽象基底クラス"""
    
    # サブクラスで定義する必要がある
    instance_vars: Dict[str, str] = {}
    
    def __init__(self, folder_path: str | os.PathLike[str], init_load: bool = True):
        self.folder_path = folder_path
        if init_load:
            self._load_all_data()
    
    def _load_all_data(self):
        """全データをロード"""
        for attr_name, ext in self.instance_vars.items():
            file_path = f"{self.folder_path}/{attr_name}{ext}"
            hided_attr_name = f"_{attr_name}"
            setattr(self, hided_attr_name, self._load_file(file_path))
    
    def _load_file(self, file_path: str) -> Optional[Any]:
        """単一ファイルをロード"""
        try:
            if not file_path or not os.path.exists(file_path):
                return None
            if file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"{file_path} の読み込みに失敗しました。: {e}")
            return None
    
    def save_instance(self):
        """インスタンスを保存"""
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        
        for attr_name, ext in self.instance_vars.items():
            hided_attr_name = f"_{attr_name}"
            attr = getattr(self, hided_attr_name, None)
            if attr is not None:
                file_path = f"{self.folder_path}/{attr_name}{ext}"
                self._save_file(file_path, attr)
    
    def _save_file(self, file_path: str, data: Any):
        """単一ファイルを保存"""
        if file_path.endswith('.parquet'):
            data.to_parquet(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
    
    @abstractmethod
    def getter(self) -> Any:
        """データクラスとして返却（サブクラスで実装）"""
        pass