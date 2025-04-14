from typing import Dict, Type, Optional, Any
from machine_learning.core.model_base import ModelBase


class ModelRegistry:
    """モデルの登録と管理を行うレジストリクラス"""
    
    _registry: Dict[str, Type[ModelBase]] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[ModelBase]) -> None:
        """
        モデルクラスをレジストリに登録する
        
        Args:
            model_type: モデルタイプ名（例: 'lasso', 'lightgbm'）
            model_class: モデルクラス
        """
        cls._registry[model_type.lower()] = model_class
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Type[ModelBase]:
        """
        モデルタイプからモデルクラスを取得する
        
        Args:
            model_type: モデルタイプ名
            
        Returns:
            モデルクラス
            
        Raises:
            ValueError: 未登録のモデルタイプが指定された場合
        """
        model_type = model_type.lower()
        if model_type not in cls._registry:
            raise ValueError(f"未登録のモデルタイプです: {model_type}")
        return cls._registry[model_type]
    
    @classmethod
    def create_model(cls, model_type: str, model_name: str, **kwargs) -> ModelBase:
        """
        モデルを作成する
        
        Args:
            model_type: モデルタイプ名
            model_name: モデル名
            **kwargs: モデル初期化パラメータ
            
        Returns:
            作成されたモデルインスタンス
        """
        model_class = cls.get_model_class(model_type)
        return model_class(name=model_name, **kwargs)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Type[ModelBase]]:
        """
        利用可能なモデルタイプの一覧を取得する
        
        Returns:
            利用可能なモデルタイプとクラスの辞書
        """
        return cls._registry.copy()


# 登録初期化（他のモジュールからインポート時に実行される）
def _initialize_registry():
    """レジストリの初期化（モデルの登録）"""
    # 循環インポートを避けるため、関数内でインポート
    from machine_learning.estimators.lasso import LassoModel
    from machine_learning.estimators.lightgbm import LightGBMModel
    
    # モデルを登録
    ModelRegistry.register('lasso', LassoModel)
    ModelRegistry.register('lightgbm', LightGBMModel)


# 初期化の実行
_initialize_registry()