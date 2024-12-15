from dataclasses import dataclass
from typing import Optional
import os
import json
from datetime import datetime
from typing import List, Dict
from typing import Literal
import paths

@dataclass
class TradeParameters:
    '''新規発注時のパラメータを規定。symbol_codeのみは必須入力。'''
    symbol_code: str = None
    trade_type: Literal["現物買", "現物売", "信用新規買", "信用新規売"] = "信用新規買"            # ex: "信用新規買"
    unit: int = 100
    order_type: Literal["指値", "成行", "逆指値"] =  "成行"
    order_type_value: Literal["寄指", "引指", "不成", "IOC指",
                              "寄成", "引成", "IOC成"] = "寄成"
    limit_order_price: Optional[float] = None
    stop_order_trigger_price: Optional[float] = None
    stop_order_type: Literal["指値", "成行"] = "成行"
    stop_order_price: Optional[float] = None
    period_type: Literal["当日中", "今週中", "期間指定"] = "当日中"
    period_value: Optional[str] = None
    period_index: Optional[int] = None
    trade_section: Literal["特定預り", "一般預り", "NISA預り", "旧NISA預り"] = "特定預り"
    margin_trade_section: Literal["制度", "一般", "日計り"] = "制度"

class OrderStatus:
    WAITING = "発注待ち"
    ORDERED = "発注済"
    EXECUTED = "新規約定済"
    CLOSING_ORDERED = "決済発注済"
    CLOSED = "決済完了"

class SBIOrderManager:
    def __init__(self):
        self.base_dir = paths.ORDERS_FOLDER
        os.makedirs(self.base_dir, exist_ok=True)
        self.today = datetime.now().strftime("%Y%m%d")
        self.file_path = os.path.join(self.base_dir, f"orders_{self.today}.json")

        self.orders: List[Dict] = []

        self._load_data()

    def _load_data(self):
        """当日ファイルの読み込み。存在しなければ新規ファイル作成"""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.orders = json.load(f)
        else:
            # 当日初回実行なので空ファイルとして初期化
            self.orders = []
            self._save_data()

    def _save_data(self):
        """当日のファイルへ書き込み"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.orders, f, ensure_ascii=False, indent=2)

    def add_new_order(self, params: TradeParameters) -> int:
        """新規に発注待ちポジションを追加し、その注文のリスト上での順番を返す"""
        if params.symbol_code is None:
            raise ValueError("証券コードを設定してください。")
        self.orders.append({
            "order_id": 0,
            "status": OrderStatus.WAITING,
            "params": vars(params)  # dataclassをdictへ変換
        })
        self._save_data()

        return len(self.orders) - 1
    
    def update_order_id(self, order_num: int, order_id: str):
        '''発注時に確定したオーダーIDを'''
        self.orders[order_num]['order_id'] = order_id
        self._save_data()

    def remove_waiting_order(self, order_id: str):
        """'指定した銘柄コードのデータを削除"""
        # '発注待ち'以外のデータのみを保持
        for order in self.orders:
            if order['order_id'] == order_id:
                if order['status'] != OrderStatus.WAITING:
                    raise ValueError(f'削除できるのは発注待ちの注文のみです。注文ID{order_id}の注文は、{order["status"]}です。')
                self.orders = [order for order in self.orders if order["order_id"] != order_id]
                self._save_data()
                return
        raise ValueError(f'銘柄コード{order_id}の注文データが存在しません。')
    

    def remove_waiting_orders(self):
        """'発注待ち'ステータスのデータを全削除"""
        # '発注待ち'以外のデータのみを保持
        self.orders = [order for order in self.orders if order["status"] != OrderStatus.WAITING]
        self._save_data()

    def update_status(self, order_id: str, new_status: str):
        """ステータスの更新"""
        for order in self.orders:
            if order['order_id'] == order_id:
                order["status"] = new_status
                self._save_data()
                return

    def get_order_info(self, order_id: str):
        """指定銘柄の注文情報取得"""
        for order in self.orders:
            if order['order_id'] == order_id:
                return order

    def get_all_orders(self):
        """当日管理中の全注文情報取得"""
        return self.orders
