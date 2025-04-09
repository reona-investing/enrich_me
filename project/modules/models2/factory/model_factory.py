"""
さまざまな種類のモデルを作成するファクトリークラス
"""
from typing import Dict, List

from models2.base.base_model import BaseModel
from models2.models3.lasso_model import LassoModel
from models2.models3.lgbm_model import LgbmModel
from models2.containers.model_container import ModelContainer
from models2.containers.ensemble import EnsembleModel
from models2.containers.period_model import PeriodSwitchingModel


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

    @staticmethod
    def create_ensemble_model(name: str = "Ensemble") -> EnsembleModel:
        """
        複数のモデルの予測結果をアンサンブルするためのモデルを作成
        
        Args:
            name: アンサンブルの名前
            
        Returns:
            EnsembleModel: 複数モデルのアンサンブルを管理するコンテナ
        """
        return EnsembleModel(name=name)

    @staticmethod
    def create_period_switching_model(name: str = "PeriodSwitchingModel") -> PeriodSwitchingModel:
        """
        期間ごとに異なるモデルを切り替えるためのコンテナを作成
        
        Args:
            name: モデルの名前
            
        Returns:
            PeriodSwitchingModel: 期間ごとにモデルを切り替えるコンテナ
        """
        return PeriodSwitchingModel(name=name)