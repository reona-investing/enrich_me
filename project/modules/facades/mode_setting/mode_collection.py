from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Any

class DataUpdateMode(str, Enum):
    """データ更新に関するモード"""
    UPDATE = 'update'
    LOAD_ONLY = 'load_only'
    NONE = 'none'

class MachineLearningMode(str, Enum):
    """機械学習に関するモード"""
    TRAIN_AND_PREDICT = 'train_and_predict'
    PREDICT_ONLY = 'predict_only'
    LOAD_ONLY = 'load_only'
    NONE = 'none'

class OrderExecutionMode(str, Enum):
    """注文実行に関するモード"""
    NEW = 'new'
    ADDITIONAL = 'additional'
    SETTLE = 'settle'
    NONE = 'none'

class TradeDataFetchMode(str, Enum):
    """データ更新に関するモード"""
    FETCH = 'fetch'
    NONE = 'none'

class ModeCollection(BaseModel):
    """フラグ管理用クラス"""
    model_config = ConfigDict(validate_assignment=True)
    order_execution_mode: OrderExecutionMode = Field(default=OrderExecutionMode.NONE)
    machine_learning_mode: MachineLearningMode = Field(default=MachineLearningMode.NONE)
    data_update_mode: DataUpdateMode = Field(default=DataUpdateMode.NONE)
    trade_data_fetch_mode: TradeDataFetchMode = Field(default=TradeDataFetchMode.NONE)

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
                f"machine_learning_modeが{values.machine_learning_mode}のためdata_update_modeを'load_only'に変更しました。"
            )
        if messages:
            for msg in messages:
                print(msg)
        return values

    def model_copy(
        self,
        *,
        update: Optional[dict[str, Any]] = None,
        deep: bool = False,
    ) -> "ModeCollection":
        """
        モデルをコピーし、更新後にバリデーションを実行する。
        
        Params:
            update (dict[str, Any]): 更新したいパラメータを指定。
            
            パラメータ:
            data_update_mode: ['update', 'load_only', 'none'], 
            machine_learning_mode: ['train_and_predict', 'predict_only', 'load_only', 'none'], 
            order_execution_mode: ['new', 'additional', 'settle', 'none'], 
            trade_data_fetch_mode: ['fetch', 'none']
        """
        copied = super().model_copy(update=update, deep=deep)
        return self.__class__.model_validate(copied.model_dump())




class DataUpdateMode(str, Enum):
    """データ更新に関するモード"""
    UPDATE = 'update'
    LOAD_ONLY = 'load_only'
    NONE = 'none'

class MachineLearningMode(str, Enum):
    """機械学習に関するモード"""
    TRAIN_AND_PREDICT = 'train_and_predict'
    PREDICT_ONLY = 'predict_only'
    LOAD_ONLY = 'load_only'
    NONE = 'none'

class OrderExecutionMode(str, Enum):
    """注文実行に関するモード"""
    NEW = 'new'
    ADDITIONAL = 'additional'
    SETTLE = 'settle'
    NONE = 'none'

class TradeDataFetchMode(str, Enum):
    """データ更新に関するモード"""
    FETCH = 'fetch'
    NONE = 'none'