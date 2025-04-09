from machine_learning.models.ml_model_base import MachineLearningModelBase
from machine_learning.models.lasso_model import LassoModel
from machine_learning.models.lgbm_model import LgbmModel

class ModelFactory:
    """個別モデルのファクトリークラス"""
    @staticmethod
    def create_model(model_type: str = 'lasso') -> MachineLearningModelBase:
        """新しいモデルインスタンスを作成する"""
        if model_type.lower() == 'lasso':
            return LassoModel()
        elif model_type.lower() == 'lgbm':
            return LgbmModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")