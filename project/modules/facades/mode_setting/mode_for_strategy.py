from datetime import datetime, time
from utils.jquants_api_utils import is_market_open
from .mode_factory import ModeFactory


class ModeForStrategy:

    @staticmethod
    def generate_mode():
        """現在時刻に基づいたModeCollectionを返す。"""
        mode_pattern = ModeForStrategy._select_mode()
        return ModeFactory.create_mode_collection(pattern=mode_pattern)

    @staticmethod
    def _select_mode():
        """現在時刻に基づいてモードを決定する。"""
        current_time = datetime.now().time()
        if not is_market_open(datetime.today()):
            return 'none'
        if ModeForStrategy._is_between(current_time, time(7, 0), time(8, 59)):
            return 'take_new_positions'
        if ModeForStrategy._is_between(current_time, time(9, 0), time(9, 29)):
            return 'take_additional_positions'
        if ModeForStrategy._is_between(current_time, time(11, 30), time(15, 19)):
            return 'settle_positions'
        if ModeForStrategy._is_between(current_time, time(15, 30), time(23, 59)):
            return 'fetch_trade_data'
        return 'none'

    @staticmethod
    def _is_between(current_time, start_time, end_time):
        """現在時刻が指定した時間範囲内にあるかを判定"""
        return start_time <= current_time <= end_time