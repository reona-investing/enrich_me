from datetime import datetime, time
import pandas as pd
import os
import jquants_api_fetcher as fetcher
import paths
from utils import SingletonMeta

class FlagManager(metaclass=SingletonMeta):
    """フラグ管理クラス（シングルトン）"""
    
    def __init__(self):
        self.flags = {
            "fetch_data": False,
            "predict": False,
            "take_new_positions": False,
            "take_additional_positions": False,
            "settle_positions": False,
            "fetch_result": False,
            "learn": False,
            "process_stock_price": False,
            "update_dataset": False,
            "update_models": False,
        }
        self.update_flags()
    
    def update_flags(self):
        """現在時刻に基づいてフラグを更新する"""
        current_time = datetime.now().time()
        if self._is_market_open_day():
            self.flags["fetch_data"] = self._is_between(current_time, time(7, 0), time(8, 59))
            self.flags["predict"] = self._is_between(current_time, time(7, 0), time(8, 59))
            self.flags["take_new_positions"] = self._is_between(current_time, time(7, 0), time(8, 59))
            self.flags["take_additional_positions"] = self._is_between(current_time, time(9, 0), time(9, 29))
            self.flags["settle_positions"] = self._is_between(current_time, time(11, 30), time(15, 19))
            self.flags["fetch_result"] = self._is_between(current_time, time(15, 30), time(23, 59))
        self._update_dependent_flags()

    def set_flag(self, flag_name: str, value: bool):
        """特定のフラグを手動で設定する"""
        if flag_name not in self.flags:
            raise ValueError(f"Invalid flag name: {flag_name}")
        if flag_name in ["update_dataset", "update_models"]:
            raise ValueError(f"Connot set the flag manually: {flag_name}")
        self.flags[flag_name] = value
        self._update_dependent_flags()  # フラグを設定した後に自動更新
            

    def set_flags(self, **kwargs):
        """
        複数のフラグをまとめて設定する。
        例: set_flags(training=True, prediction=False)
        """
        for flag_name, value in kwargs.items():
            if flag_name not in self.flags:
                raise ValueError(f"Invalid flag name: {flag_name}")
            if flag_name in ["update_dataset", "update_models"]:
                raise ValueError(f"Connot set the flag manually: {flag_name}")
            if value is not None:
                self.set_flag(flag_name, value)
                self._update_dependent_flags()

    def _is_between(self, current_time, start_time, end_time):
        """現在時刻が指定した時間範囲内にあるかを判定"""
        return start_time <= current_time <= end_time
    
    def _is_market_open_day(self) -> bool:
        # 今日が営業日かどうかの判定
        current_date = datetime.now().date()
        market_open_day_df = fetcher.cli.get_markets_trading_calendar(
                from_yyyymmdd=(current_date).strftime('%Y%m%d'),
                to_yyyymmdd=(current_date).strftime('%Y%m%d')
            )
        is_market_open_day =  market_open_day_df['HolidayDivision'].iat[0] == '1' #平日：1、土日：0、祝日：3
        return is_market_open_day

    def _update_dependent_flags(self):
        # update_datasetとupdate_modelsのフラグを他のフラグに基づいて更新
        self.flags["update_dataset"] = self.flags["learn"] or self.flags["fetch_data"]
        self.flags["update_models"] = any(self.flags[flag] for flag in ["learn", "predict", "take_new_positions"])
  
    def get_flags(self):
        """現在のフラグ状態を取得"""
        return self.flags


# 使用例
if __name__ == "__main__":
    # シングルトンオブジェクトの取得
    flag_manager_1 = FlagManager()
    print(flag_manager_1.get_flags())
    # フラグの更新
    flag_manager_1.set_flag('predict', True)
    print(flag_manager_1.get_flags())
    # シングルトンオブジェクトの再取得
    flag_manager_1 = FlagManager()
    print(flag_manager_1.get_flags())
# TODO update_dataとupdate_modelの制約条件を実装する。
# TODO 既存のフラグマネージャ―をこれに置き換える。