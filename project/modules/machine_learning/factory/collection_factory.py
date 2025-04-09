from typing import Union, List, Dict, Optional, Any, TypeVar
import pandas as pd

from machine_learning.collection.model_collection import ModelCollection
from machine_learning.params.hyperparams import HyperParams
from machine_learning.factory.model_factory import ModelFactory

class ModelCollectionFactory:
    """モデルコレクションのファクトリークラス"""
    
    @staticmethod
    def create_collection(model_type: str = 'lasso') -> ModelCollection:
        """新しいモデルコレクションを作成する"""
        return ModelCollection(model_type=model_type)
    
    @staticmethod
    def load_collection(filepath: str) -> ModelCollection:
        """ファイルからモデルコレクションを読み込む"""
        return ModelCollection.load(filepath)
    
    @staticmethod
    def create_from_data(data: Dict[str, Dict[str, Union[pd.DataFrame, pd.Series]]],
                        model_type: str = 'lasso',
                        params: HyperParams = None) -> ModelCollection:
        """データからモデルコレクションを作成して学習する"""
        collection = ModelCollection(model_type=model_type)
        
        # 各セクターごとにモデルを作成・学習
        for sector_name in data.keys():
            model = ModelFactory.create_model(model_type)
            collection.add_model(sector_name, model)
        
        # 全モデルを学習
        if params:
            collection.train_all(data, params)
            
        return collection