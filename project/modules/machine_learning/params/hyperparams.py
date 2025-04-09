from abc import ABC
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class HyperParams(ABC):
    """ハイパーパラメータの基底データクラス"""
    def get_model_params(self) -> Dict[str, Any]:
        """データクラスのすべてのインスタンス変数を{変数名：変数の中身}の形の辞書で返す"""
        return asdict(self)

@dataclass
class LassoParams(HyperParams):
    """Lassoモデルのハイパーパラメータ"""
    alpha: float = 0.01
    max_iter: int = 1000
    tol: float = 0.0001
    max_features: int = 5
    min_features: int = 3

@dataclass
class LgbmParams(HyperParams):  # HyperParamsを継承するように変更
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