from machine_learning.collection import ModelCollection


class CollectionFactory:
    """モデルコレクションのファクトリークラス"""
    
    def create_collection(self) -> ModelCollection:
        """
        新しいモデルコレクションを作成する
        
        Returns:
            空のモデルコレクション
        """
        return ModelCollection()
    
    def load_collection(self, path: str) -> ModelCollection:
        """
        ファイルからモデルコレクションを読み込む
        
        Args:
            path: 読み込むファイルパス
            
        Returns:
            読み込まれたモデルコレクション
        """
        return ModelCollection.load(path)