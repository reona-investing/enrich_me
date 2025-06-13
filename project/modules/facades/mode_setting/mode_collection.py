from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, model_validator

class OrderExecutionMode(str, Enum):
    """注文実行に関するモード"""
    NEW = 'new'
    NONE = 'none'

class MachineLearningMode(str, Enum):
    """機械学習に関するモード"""
    NONE = 'none'
    TRAIN_AND_PREDICT = 'train_and_predict'
    PREDICT_ONLY = 'predict_only'
    LOAD_ONLY = 'load_only'

class DataUpdateMode(str, Enum):
    """データ更新に関するモード"""
    NONE = 'none'
    LOAD_ONLY = 'load_only'
    UPDATE = 'update'

class ModeCollection(BaseModel):
    """フラグ管理用のモードコレクション"""
    order_execution_mode: OrderExecutionMode = Field(default=OrderExecutionMode.NONE)
    machine_learning_mode: MachineLearningMode = Field(default=MachineLearningMode.NONE)
    data_update_mode: DataUpdateMode = Field(default=DataUpdateMode.NONE)

    @model_validator(mode='after')
    def adjust_modes(cls, values: 'ModeCollection') -> 'ModeCollection':
        messages = []
        if (
            values.order_execution_mode is OrderExecutionMode.NEW
            and values.machine_learning_mode is MachineLearningMode.NONE
        ):
            values.machine_learning_mode = MachineLearningMode.LOAD_ONLY
            messages.append(
                "order_execution_modeが'new'のためmachine_learning_modeを'load_only'に変更しました。"
            )
        if (
            values.machine_learning_mode in {MachineLearningMode.TRAIN_AND_PREDICT, MachineLearningMode.PREDICT_ONLY}
            and values.data_update_mode is DataUpdateMode.NONE
        ):
            values.data_update_mode = DataUpdateMode.LOAD_ONLY
            messages.append(
                "machine_learning_modeの設定に合わせるためdata_update_modeを'load_only'に変更しました。"
            )
        if messages:
            for msg in messages:
                print(msg)
        return values
