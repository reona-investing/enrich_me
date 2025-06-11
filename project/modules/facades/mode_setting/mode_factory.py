from .mode_collection import ModeCollection
from typing import Literal

class ModeFactory:

    @staticmethod
    def create_mode_collection(pattern: Literal['take_new_positions',
                                                'take_additional_positions',
                                                'settle_positions',
                                                'fetch_trade_data',
                                                'learn_models',
                                                'none']):
        
        if pattern == 'take_new_positions':
            return ModeCollection(data_update_mode='update_and_load',
                                  machine_learning_mode='predict_only',
                                  order_execution_mode='new',
                                  trade_data_fetch_mode='none')
        if pattern == 'take_additional_positions':
            return ModeCollection(data_update_mode='none',
                                  machine_learning_mode='none',
                                  order_execution_mode='additional',
                                  trade_data_fetch_mode='none')
        if pattern == 'settle_positions':
            return ModeCollection(data_update_mode='none',
                                  machine_learning_mode='none',
                                  order_execution_mode='settle',
                                  trade_data_fetch_mode='none')
        if pattern == 'fetch_trade_data':
            return ModeCollection(data_update_mode='none',
                                  machine_learning_mode='none',
                                  order_execution_mode='none',
                                  trade_data_fetch_mode='fetch')
        if pattern == 'learn_models':
            return ModeCollection(data_update_mode='load_only',
                                  machine_learning_mode='train_and_predict',
                                  order_execution_mode='none',
                                  trade_data_fetch_mode='none')
        if pattern == 'none':
            return ModeCollection(data_update_mode='none',
                                  machine_learning_mode='none',
                                  order_execution_mode='none',
                                  trade_data_fetch_mode='none')