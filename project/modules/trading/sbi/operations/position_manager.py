import os
import json
from typing import List, Dict
from datetime import datetime
from utils.paths import Paths
from trading.sbi.operations.trade_parameters import TradeParameters

class PositionManager:
    # ステータス定数
    STATUS_UNORDERED = "未発注"
    STATUS_ORDERED = "発注済"

    def __init__(self):
        """ポジション管理クラス"""
        self.base_dir = Paths.POSITIONS_FOLDER
        os.makedirs(self.base_dir, exist_ok=True)
        self.today = datetime.now().strftime("%Y%m%d")
        self.file_path = os.path.join(self.base_dir, f"positions_{self.today}.json")

        self.positions: List[Dict] = []
        self._load_data()

    def _load_data(self):
        """ポジションデータをJSONファイルから読み込む"""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.positions = json.load(f)
        else:
            self.positions = []
            self._save_data()

    def _save_data(self):
        """ポジションデータをJSONファイルに保存"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.positions, f, ensure_ascii=False, indent=2, default=lambda o: o.dict())

    def add_position(self, order_params: TradeParameters):
        """新しいポジションを追加"""
        position = {
            "order_id": None,  # 発注時ID（未設定）
            "order_status": self.STATUS_UNORDERED,
            "order_params": order_params,  # TradeParametersオブジェクト
            "settlement_id": None,  # 決済時ID（未設定）
            "settlement_status": self.STATUS_UNORDERED,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.positions.append(position)
        self._save_data()

    def update_status(self, order_id: int, status_type: str, new_status: str):
        """
        特定のポジションのステータスを更新
        - status_type: 'order_status' または 'settlement_status'
        """
        for position in self.positions:
            if position["order_id"] == order_id:
                position[status_type] = new_status
                position["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._save_data()
                return True
        return False

    def find_unordered_position_by_params(self, trade_params: TradeParameters) -> int:
        """
        指定したパラメータのポジションのリスト内での位置を取得
        比較対象は'symbol_code', 'trade_type', 'unit'の3つ
        """
        compare_keys = ['symbol_code', 'trade_type', 'unit']
        input_condition = [getattr(trade_params, key) for key in compare_keys]
        for i, position in enumerate(self.positions):
            if type(position['order_params']) == TradeParameters:
                control_condition = [getattr(position['order_params'], key) for key in compare_keys]
            else:
                control_condition = [position['order_params'][key] for key in compare_keys]
            if (position['order_status'] == PositionManager.STATUS_UNORDERED) & (input_condition == control_condition):
                return i
        return None

    def update_order_id(self, index: int, order_id: int):
        """発注時IDを更新"""
        if 0 <= index < len(self.positions):
            self.positions[index]["order_id"] = order_id
            self._save_data()
            return True
        return False

    def update_by_symbol(self, symbol_code: str, update_key: str, update_value: str | int | float) -> bool:
        """
        指定された銘柄コードに一致するレコードのorder_idを更新します。
        複数のレコードが一致した場合は、そのすべてに対して更新を行います。

        Args:
            symbol_code (str): 更新対象レコードを選択するためのシンボルコード
            update_key (str): 更新対象とするデータのキー
            update_value (str | int | float): 更新対象とするデータの更新後の値

        Returns:
            bool: 1件以上のレコードが更新された場合はTrue、更新対象がない場合はFalse
        """
        updated = False

        for position in self.positions:
            # "order_params"が存在するかどうかのチェックを行い、"symbol_code"の値を比較
            if position.get("order_params", {}).get("symbol_code") == symbol_code:
                position[update_key] = update_value
                updated = True

        if updated:
            self._save_data()

        return updated

    def get_all_positions(self) -> List[Dict]:
        """全ポジション情報を取得"""
        return self.positions

    def get_pending_positions(self) -> List[Dict]:
        """
        発注待ちのポジションを取得
        - order_status: '未発注'
        - settlement_status: '未発注'
        """
        pending_positions = [
            pos for pos in self.positions
            if pos["order_status"] == self.STATUS_UNORDERED and pos["settlement_status"] == self.STATUS_UNORDERED
        ]
        return pending_positions

    def get_open_positions(self) -> List[Dict]:
        """
        決済発注待ちのポジションを取得
        - order_status: '発注済'
        - settlement_status: '未発注'
        """
        pending_positions = [
            pos for pos in self.positions
            if pos["order_status"] == self.STATUS_ORDERED and pos["settlement_status"] == self.STATUS_UNORDERED
        ]
        return pending_positions

    def remove_waiting_order(self, order_id: str) -> None:
        """'指定した銘柄コードのデータを削除"""
        # '発注待ち'以外のデータのみを保持
        for order in self.positions:
            if order['order_id'] == order_id:
                if order['order_status'] != PositionManager.STATUS_UNORDERED:
                    raise ValueError(f'削除できるのは発注待ちの注文のみです。注文ID{order_id}の注文は、発注待ちではありません。')
                self.positions = [order for order in self.positions if order["order_id"] != order_id]
                self._save_data()
                return
        raise ValueError(f'銘柄コード{order_id}の注文データが存在しません。')