import os
import pickle
from typing import Optional

from machine_learning.collection import BaseCollection
from machine_learning.models import BaseModel, LassoModel, LgbmModel
from machine_learning.params import BaseParams, LassoParams, LgbmParams


class ModelCollection(BaseCollection):
    """機械学習モデルのコレクション"""
    
    def __init__(self):
        """モデルコレクションを初期化"""
        super().__init__()
        self.models = {}  # Dict[str, BaseModel]
    
    def generate_model(self, name: str, type: str, params: Optional[BaseParams] = None) -> BaseModel:
        """
        新しいモデルを生成してコレクションに追加する
        
        Args:
            name: モデル名
            type: モデルタイプ ('lasso' or 'lgbm')
            params: モデルパラメータ（省略時はデフォルト）
            
        Returns:
            生成されたモデルのインスタンス
        
        Raises:
            ValueError: 不明なモデルタイプの場合
        """
        if type.lower() == 'lasso':
            model = LassoModel(name, params or LassoParams())
        elif type.lower() == 'lgbm':
            model = LgbmModel(name, params or LgbmParams())
        else:
            raise ValueError(f"不明なモデルタイプです: {type}. 'lasso'または'lgbm'を指定してください。")
        
        self.models[name] = model
        return model
    
    def save(self, path: str) -> None:
        """
        コレクションをファイルに保存する
        
        Args:
            path: 保存先のファイルパス
        """
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存用の辞書を作成
        collection_data = {
            'models': self.models,
        }
        
        # ファイルに保存
        with open(path, 'wb') as file:
            pickle.dump(collection_data, file)
        
        print(f"モデルコレクションを {path} に保存しました。")
    
    @classmethod
    def load(cls, path: str) -> 'ModelCollection':
        """
        ファイルからコレクションを読み込む
        
        Args:
            path: 読み込むファイルパス
            
        Returns:
            読み込まれたモデルコレクション
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: ファイルが正しくない形式の場合
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")
        
        # ファイルから読み込み
        with open(path, 'rb') as file:
            collection_data = pickle.load(file)
        
        # コレクションの作成
        collection = cls()
        
        # モデルの設定
        if 'models' in collection_data:
            collection.models = collection_data['models']
        else:
            raise ValueError(f"ファイル {path} が正しくない形式です。")
        
        print(f"モデルコレクションを {path} から読み込みました。")
        return collection