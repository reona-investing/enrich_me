import os
import pickle

from machine_learning.core.model_base import ModelBase


def save_model(model: ModelBase, path: str) -> None:
    """
    モデルをシリアライズして保存する
    
    Args:
        model: 保存するモデル
        path: 保存先のファイルパス
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str) -> ModelBase:
    """
    シリアライズされたモデルを読み込む
    
    Args:
        path: 読み込むファイルパス
        
    Returns:
        読み込まれたモデル
        
    Raises:
        FileNotFoundError: 指定されたパスにファイルが存在しない場合
        ValueError: 読み込まれたオブジェクトがModelBaseを継承していない場合
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    if not isinstance(model, ModelBase):
        raise ValueError(f"読み込まれたオブジェクトはModelBaseを継承していません: {type(model)}")
    
    return model


def save_collection(collection, path: str) -> None:
    """
    モデルコレクションを保存する
    
    Args:
        collection: 保存するモデルコレクション
        path: 保存先のファイルパス
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # コレクションデータの準備
    collection_data = {
        'name': collection.name,
        'models': collection.models,
    }
    
    # 保存
    with open(path, 'wb') as f:
        pickle.dump(collection_data, f)


def load_collection(path: str):
    """
    シリアライズされたモデルコレクションを読み込む
    
    Args:
        path: 読み込むファイルパス
        
    Returns:
        読み込まれたモデルコレクション
        
    Raises:
        FileNotFoundError: 指定されたパスにファイルが存在しない場合
        KeyError: 必要なキーが存在しない場合
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    
    with open(path, 'rb') as f:
        collection_data = pickle.load(f)
    
    # モジュールのインポート
    from machine_learning.core.collection import ModelCollection
    
    # コレクションの作成
    collection = ModelCollection(
        name=collection_data.get('name', 'loaded_collection'),
        path=path
    )
    
    # モデルの設定
    if 'models' in collection_data:
        collection.models = collection_data['models']
    else:
        raise KeyError("読み込まれたデータに 'models' キーが存在しません。")
    
    return collection