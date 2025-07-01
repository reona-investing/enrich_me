from __future__ import annotations

from dataclasses import dataclass
from typing import List

from machine_learning.ml_dataset.core import MLDataset

@dataclass
class ModelOrderConfig:
    """設定オブジェクト。margin_weightの合計は normalize_margin_weights() で調整する"""
    ml_dataset: MLDataset
    sector_csv: str
    trading_sector_num: int
    candidate_sector_num: int
    top_slope: float
    margin_weight: float = 0.5


def normalize_margin_weights(configs: List[ModelOrderConfig]) -> None:
    """margin_weight の合計が 1 になるよう正規化する"""
    if not configs:
        return
    total = sum(cfg.margin_weight for cfg in configs)
    if total <= 0:
        equal = 1 / len(configs)
        for cfg in configs:
            cfg.margin_weight = equal
    elif total != 1:
        for cfg in configs:
            cfg.margin_weight = cfg.margin_weight / total
