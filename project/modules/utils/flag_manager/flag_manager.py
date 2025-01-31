from datetime import datetime, time
from utils.jquants_api_utils import is_market_open
from utils.singleton import SingletonMeta
from enum import Enum
from typing import List

class Flags(Enum):
    FETCH_DATA = "fetch_data"
    PREDICT = "predict"
    TAKE_NEW_POSITIONS = "take_new_positions"
    TAKE_ADDITIONAL_POSITIONS = "take_additional_positions"
    SETTLE_POSITIONS = "settle_positions"
    FETCH_RESULT = "fetch_result"
    LEARN = "learn"
    PROCESS_STOCK_PRICE = "process_stock_price"
    UPDATE_DATASET = "update_dataset"
    UPDATE_MODELS = "update_models"    

class FlagManager(metaclass=SingletonMeta):
    """フラグ管理クラス（シングルトン）"""
    
    def __init__(self):
        self.flags = {flag: False for flag in Flags}  # 列挙型を使用してフラグを初期化
        self.update_flags()
    
    def update_flags(self):
        """現在時刻に基づいてフラグを更新する"""
        current_time = datetime.now().time()
        if self._is_market_open_day():
            self.flags[Flags.FETCH_DATA] = self._is_between(current_time, time(7, 0), time(8, 59))
            self.flags[Flags.PREDICT] = self._is_between(current_time, time(7, 0), time(8, 59))
            self.flags[Flags.TAKE_NEW_POSITIONS] = self._is_between(current_time, time(7, 0), time(8, 59))
            self.flags[Flags.TAKE_ADDITIONAL_POSITIONS] = self._is_between(current_time, time(9, 0), time(9, 29))
            self.flags[Flags.SETTLE_POSITIONS] = self._is_between(current_time, time(11, 30), time(15, 19))
            self.flags[Flags.FETCH_RESULT] = self._is_between(current_time, time(15, 30), time(23, 59))
        self._update_dependent_flags()

    def set_flag(self, flag: Flags, value: bool):
        """特定のフラグを手動で設定する"""
        if flag not in self.flags:
            raise ValueError(f"Invalid flag name: {flag}")
        if flag in [Flags.UPDATE_DATASET, Flags.UPDATE_MODELS]:
            raise ValueError(f"Cannot set the flag manually: {flag}")
        self.flags[flag] = value
        self._update_dependent_flags()  # フラグを設定した後に自動更新

    def set_flags(self, turn_true: List[Flags] = [], turn_false: List[Flags] = []):
        """
        複数のフラグをまとめて設定する。
        turn_trueのフラグをTrueに、turn_falseのフラグをFalseに設定します。
        """
        for flag in turn_true:
            if not isinstance(flag, Flags):
                raise ValueError(f"Invalid flag type: {type(flag)}. Must be an instance of Flag.")
            self.set_flag(flag, True)
            self._update_dependent_flags()

        for flag in turn_false:
            if not isinstance(flag, Flags):
                raise ValueError(f"Invalid flag type: {type(flag)}. Must be an instance of Flag.")
            self.set_flag(flag, False)
            self._update_dependent_flags()

    def _is_between(self, current_time, start_time, end_time):
        """現在時刻が指定した時間範囲内にあるかを判定"""
        return start_time <= current_time <= end_time
    
    def _is_market_open_day(self) -> bool:
        # 今日が営業日かどうかの判定
        return is_market_open(datetime.today())

    def _update_dependent_flags(self):
        # update_datasetとupdate_modelsのフラグを他のフラグに基づいて更新
        self.flags[Flags.UPDATE_DATASET] = self.flags[Flags.LEARN] or self.flags[Flags.FETCH_DATA]
        self.flags[Flags.UPDATE_MODELS] = \
            any(self.flags[flag] for flag in [Flags.LEARN, Flags.PREDICT, Flags.TAKE_NEW_POSITIONS])
  
    def get_flags(self):
        """現在のフラグ状態を取得"""
        return self.flags

flag_manager = FlagManager()

# 使用例
if __name__ == "__main__":
    # シングルトンオブジェクトの取得
    flag_manager_1 = FlagManager()
    print(flag_manager_1.get_flags())
    # フラグの更新
    flag_manager_1.set_flag(Flags.PREDICT, True)
    print(flag_manager_1.get_flags())
    # シングルトンオブジェクトの再取得
    flag_manager_1 = FlagManager()
    print(flag_manager_1.get_flags())
# TODO update_dataとupdate_modelの制約条件を実装する。

