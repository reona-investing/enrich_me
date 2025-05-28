from models.machine_learning.ml_dataset.components.base_data_component import BaseDataComponent
from models.machine_learning.outputs import TrainerOutputs

class MLObjects(BaseDataComponent):
    """機械学習オブジェクトの管理"""
    
    instance_vars = {
        'models': '.pkl',
        'scalers': '.pkl',
    }

    def archive_ml_objects(self, models: list[object], scalers: list[object]):
        """機械学習のモデルとスケーラーを格納"""
        self._models = models
        self._scalers = scalers

    def getter(self) -> TrainerOutputs:
        """データクラスとして返却"""
        return TrainerOutputs(models=self._models, scalers=self._scalers)