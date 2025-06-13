from typing import Optional
from models.machine_learning.ml_dataset.components.base_data_component import BaseDataComponent
from models.machine_learning.outputs import TrainerOutputs

class MLObjects(BaseDataComponent):
    """機械学習オブジェクトの管理"""
    
    instance_vars = {
        'model': '.pkl',
        'scaler': '.pkl',
    }

    def archive_ml_objects(self, model: object, scaler: Optional[object] = None):
        """機械学習のモデルとスケーラーを格納"""
        self._model = model
        self._scaler = scaler

    def getter(self) -> TrainerOutputs:
        """データクラスとして返却

        ``_model`` や ``_scaler`` が存在しない場合は ``None`` を返す。
        予測のみを行う際に ``archive_ml_objects`` が呼ばれていない状況でも
        エラーとならないようにしている。
        """
        model = getattr(self, "_model", None)
        scaler = getattr(self, "_scaler", None)
        return TrainerOutputs(model=model, scaler=scaler)
