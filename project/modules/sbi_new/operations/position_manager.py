import os
import json
from typing import List, Dict, Optional
from datetime import datetime
import paths
from .trade_parameters import TradeParameters

class PositionManager:
    def __init__(self):
        """ポジション管理クラス"""
        self.base_dir = paths.ORDERS_FOLDER
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
            "order_status": "未発注",
            "order_params": order_params,  # TradeParametersオブジェクト
            "settlement_id": None,  # 決済時ID（未設定）
            "settlement_status": "未発注",
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

    def find_positions_by_status(self, status_type: str, status: str) -> List[Dict]:
        """指定したステータスのポジションを取得"""
        return [pos for pos in self.positions if pos[status_type] == status]

    def update_order_id(self, index: int, order_id: int):
        """発注時IDを更新"""
        if 0 <= index < len(self.positions):
            self.positions[index]["order_id"] = order_id
            self._save_data()
            return True
        return False

    def get_all_positions(self) -> List[Dict]:
        """全ポジション情報を取得"""
        return self.positions
