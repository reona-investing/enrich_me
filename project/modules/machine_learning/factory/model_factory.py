from typing import Optional
from machine_learning.collection import ModelCollection


class ModelFactory:
    """個別モデルのファクトリークラス"""
    
    @staticmethod
    def create_model(name: str, type: str, collection: Optional[ModelCollection] = None) -> ModelCollection:
        """
        指定されたタイプのモデルを作成する
        
        Args:
            name: モデル名
            type: モデルタイプ ('lasso' or 'lgbm')
            collection: モデルを追加するコレクション（省略時は新規作成）
            
        Returns:
            モデルを含むコレクション
        """
        if collection is None:
            collection = ModelCollection()
        
        collection.generate_model(name, type)
        return collection