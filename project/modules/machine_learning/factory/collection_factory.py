import os
from machine_learning.collection import ModelCollection


class CollectionFactory:
    """モデルコレクションのファクトリークラス"""
    
    @staticmethod
    def get_collection(path: str = None) -> ModelCollection:
        """
        pathが指定され、ファイルが存在する場合は既存のコレクションを読み込み、
        そうでなければ新しいコレクションを作成する
        
        Args:
            path: 読み込むファイルパス（省略時または存在しない場合は新規作成）
            
        Returns:
            ModelCollection: 読み込まれたまたは新規作成されたモデルコレクション
        """
        if path is not None and os.path.exists(path):
            return CollectionFactory._load_collection(path)
        else:
            return CollectionFactory._create_collection(path)
    
    @staticmethod
    def _create_collection(path=str) -> ModelCollection:
        """
        新しいモデルコレクションを作成する（内部メソッド）
        
        Returns:
            空のモデルコレクション
        """
        return ModelCollection(path)
    
    @staticmethod
    def _load_collection(path: str) -> ModelCollection:
        """
        ファイルからモデルコレクションを読み込む（内部メソッド）
        
        Args:
            path: 読み込むファイルパス
            
        Returns:
            読み込まれたモデルコレクション
        """
        return ModelCollection.load(path)