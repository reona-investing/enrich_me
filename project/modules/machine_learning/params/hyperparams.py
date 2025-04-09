from abc import ABC
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class HyperParams(ABC):
    """ハイパーパラメータの基底データクラス"""
    pass

@dataclass
class LassoParams(HyperParams):
    """Lassoモデルのハイパーパラメータ"""
    alpha: float = 0.01
    max_iter: int = 1000
    tol: float = 0.0001
    max_features: int = 5
    min_features: int = 3

    # その他のパラメータは**kwargsで渡せるようにする
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LgbmParams:
    """LightGBMモデルのハイパーパラメータ"""
    
    # 基本パラメータ
    objective: str = 'regression'
    metric: str = 'rmse'
    boosting_type: str = 'gbdt'
    learning_rate: float = 0.001
    num_leaves: int = 7
    random_seed: int = 42
    lambda_l1: float = 0.5
    
    # 学習制御パラメータ
    num_boost_round: int = 100000
    early_stopping_rounds: Optional[int] = None
    
    # 特徴量関連パラメータ
    categorical_features: Optional[List[str]] = None
    
    # その他のパラメータは**kwargsで渡せるようにする
    additional_params: Dict[str, Any] = field(default_factory=dict)