import os
import csv
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from utils.paths import Paths
from trading.sbi.operations.trade_parameters import TradeParameters

class PositionManager:
    # ステータス定数
    STATUS_UNORDERED = "未発注"
    STATUS_ORDERED = "発注済"
    
    # CSVファイルのヘッダー
    CSV_HEADERS = [
        "order_id", "order_status", "settlement_id", "settlement_status",
        "symbol_code", "trade_type", "unit", "updated_at"
    ]

    def __init__(self):
        """ポジション管理クラス"""
        self.base_dir = Paths.POSITIONS_FOLDER
        os.makedirs(self.base_dir, exist_ok=True)
        self.today = datetime.now().strftime("%Y%m%d")
        self.file_path = os.path.join(self.base_dir, f"positions_{self.today}.csv")

        self.positions: List[Dict] = []
        self._load_data()

    def _load_data(self):
        """ポジションデータをCSVファイルから読み込む"""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.positions = list(reader)
                
                # 数値型のフィールドを変換
                for position in self.positions:
                    # 数値フィールドの型変換
                    if position["order_id"] and position["order_id"] != "None":
                        position["order_id"] = int(position["order_id"])
                    else:
                        position["order_id"] = None
                        
                    if position["settlement_id"] and position["settlement_id"] != "None":
                        position["settlement_id"] = int(position["settlement_id"])
                    else:
                        position["settlement_id"] = None
                        
                    if position["unit"] and position["unit"] != "None":
                        position["unit"] = int(position["unit"])
                    
        else:
            self.positions = []
            self._save_data()

    def _save_data(self):
        """ポジションデータをCSVファイルに保存"""
        with open(self.file_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writeheader()
            writer.writerows(self._prepare_rows_for_csv())

    def _prepare_rows_for_csv(self) -> List[Dict[str, Any]]:
        """CSVに書き込むための行データを準備"""
        rows = []
        for position in self.positions:
            row = position.copy()
            
            # TradeParametersオブジェクトがあれば展開する
            if "order_params" in row:
                order_params = row.pop("order_params")
                if isinstance(order_params, TradeParameters):
                    row["symbol_code"] = order_params.symbol_code
                    row["trade_type"] = order_params.trade_type
                    row["unit"] = order_params.unit
                elif isinstance(order_params, dict):
                    row["symbol_code"] = order_params.get("symbol_code")
                    row["trade_type"] = order_params.get("trade_type")
                    row["unit"] = order_params.get("unit")
            
            # 存在しないキーに対してはNoneを設定
            for header in self.CSV_HEADERS:
                if header not in row:
                    row[header] = None
            
            rows.append(row)
        return rows

    def _trade_params_to_dict(self, trade_params: TradeParameters) -> Dict[str, Any]:
        """TradeParametersオブジェクトを辞書に変換"""
        return {
            "symbol_code": trade_params.symbol_code,
            "trade_type": trade_params.trade_type,
            "unit": trade_params.unit,
        }

    def add_position(self, order_params: TradeParameters) -> int:
        """
        新しいポジションを追加し、追加したポジションのインデックスを返す
        
        Args:
            order_params (TradeParameters): 注文パラメータ
            
        Returns:
            int: 追加したポジションのインデックス
        """
        position = {
            "order_id": None,
            "order_status": self.STATUS_UNORDERED,
            "settlement_id": None,
            "settlement_status": self.STATUS_UNORDERED,
            "symbol_code": order_params.symbol_code,
            "trade_type": order_params.trade_type,
            "unit": order_params.unit,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.positions.append(position)
        self._save_data()
        return len(self.positions) - 1  # 追加したポジションのインデックスを返す

    def update_status(self, order_id: int, status_type: str, new_status: str) -> bool:
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

    def find_unordered_position_by_params(self, trade_params: TradeParameters) -> Optional[int]:
        """
        指定したパラメータのポジションのリスト内での位置を取得
        比較対象は'symbol_code', 'trade_type', 'unit'の3つ
        """
        compare_keys = ['symbol_code', 'trade_type', 'unit']
        input_condition = [getattr(trade_params, key) for key in compare_keys]
        
        for i, position in enumerate(self.positions):
            control_condition = [position.get(key) for key in compare_keys]
            
            # ユニットが文字列の場合は数値に変換して比較
            if isinstance(control_condition[2], str) and control_condition[2].isdigit():
                control_condition[2] = int(control_condition[2])
                
            if (position['order_status'] == self.STATUS_UNORDERED and 
                input_condition == control_condition):
                return i
        return None

    def update_order_id(self, index: int, order_id: int) -> bool:
        """発注時IDを更新"""
        if 0 <= index < len(self.positions):
            self.positions[index]["order_id"] = order_id
            self._save_data()
            return True
        return False

    def update_by_symbol(self, symbol_code: str, update_key: str, update_value: Union[str, int, float]) -> bool:
        """
        指定された銘柄コードに一致するレコードの特定フィールドを更新します。
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
            if position.get("symbol_code") == symbol_code:
                position[update_key] = update_value
                position["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        open_positions = [
            pos for pos in self.positions
            if pos["order_status"] == self.STATUS_ORDERED and pos["settlement_status"] == self.STATUS_UNORDERED
        ]
        return open_positions

    def remove_waiting_order(self, order_id: Union[str, int]) -> None:
        """指定した注文IDのデータを削除"""
        # 文字列の場合は数値に変換
        if isinstance(order_id, str) and order_id.isdigit():
            order_id = int(order_id)
            
        # 該当する注文を探す
        for order in self.positions:
            current_order_id = order['order_id']
            # 文字列の場合は数値に変換して比較
            if isinstance(current_order_id, str) and current_order_id.isdigit():
                current_order_id = int(current_order_id)
                
            if current_order_id == order_id:
                if order['order_status'] != self.STATUS_UNORDERED:
                    raise ValueError(f'削除できるのは発注待ちの注文のみです。注文ID{order_id}の注文は、発注待ちではありません。')
                self.positions = [pos for pos in self.positions if pos["order_id"] != order_id]
                self._save_data()
                return
                
        raise ValueError(f'注文ID{order_id}の注文データが存在しません。')