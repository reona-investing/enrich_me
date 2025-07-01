from dataclasses import dataclass, field
from typing import List

@dataclass
class PreprocessingConfig:
    """前処理設定を管理するデータクラス"""
    outlier_threshold: float = 0.0
    no_shift_features: List[str] = field(default_factory=list)
    remove_missing_data: bool = True
