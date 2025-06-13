from dataclasses import dataclass
from typing import Literal

@dataclass
class ModeCollection:
    data_update_mode: Literal['update_and_load', 'load_only', 'none']
    machine_learning_mode: Literal['train_and_predict', 'predict_only', 'load_only', 'none']
    order_execution_mode: Literal['new', 'additional', 'settle', 'none']
    trade_data_fetch_mode: Literal['fetch', 'none']