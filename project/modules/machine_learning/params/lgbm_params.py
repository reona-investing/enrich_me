from dataclasses import dataclass, field
from typing import Optional, List
from machine_learning.params import BaseParams


@dataclass
class LgbmParams(BaseParams):
    """LightGBM回帰モデルのパラメータ"""
    
    # LightGBMモデル固有のパラメータ
    objective: str = 'regression'  # 目的関数
    metric: str = 'rmse'  # 評価指標
    boosting_type: str = 'gbdt'  # ブースティング方法
    learning_rate: float = 0.001  # 学習率
    num_leaves: int = 7  # 葉の数
    verbose: int = -1  # ログ出力レベル
    lambda_l1: float = 0.5  # L1正則化
    
    # その他のパラメータ
    num_boost_round: int = 100000  # ブースト回数
    early_stopping_rounds: Optional[int] = None  # 早期終了のラウンド数
    
    # カテゴリ特徴量
    categorical_features: List[str] = field(default_factory=list)