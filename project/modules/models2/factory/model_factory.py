"""
さまざまな種類のモデルを作成するファクトリークラス
"""
from typing import Dict, List

from models2.base.base_model import BaseModel
from models2.models3.lasso_model import LassoModel
from models2.models3.lgbm_model import LgbmModel
from models2.containers.model_container import ModelContainer


class ModelFactory:
    """
    さまざまな種類のモデルを作成するファクトリークラス
    """
    
    @staticmethod
    def create_lasso() -> LassoModel:
        """単一セクター用のLassoモデルを作成"""
        return LassoModel()
    
    @staticmethod
    def create_lgbm() -> LgbmModel:
        """LightGBMモデルを作成"""
        return LgbmModel()
    
    @staticmethod
    def create_lasso_container(sectors: List[str]) -> ModelContainer:
        """
        複数セクター用のLassoモデルコンテナを作成
        
        Args:
            sectors: セクター名のリスト
            
        Returns:
            ModelContainer: 各セクターのLassoモデルを含むコンテナ
        """
        container = ModelContainer(name="LassoMultiSector")
        for sector in sectors:
            container.add_model(sector, LassoModel())
        return container
    
    @staticmethod
    def create_from_existing(models_dict: Dict[str, BaseModel], name: str = "") -> ModelContainer:
        """
        既存のモデル辞書からコンテナを作成
        
        Args:
            models_dict: キーとモデルのペアを含む辞書
            name: コンテナの名前
            
        Returns:
            ModelContainer: 指定されたモデルを含むコンテナ
        """
        container = ModelContainer(name=name)
        for key, model in models_dict.items():
            container.add_model(key, model)
        return container