from dataclasses import dataclass
from machine_learning.params import BaseParams


@dataclass
class LassoParams(BaseParams):
    """LASSO回帰モデルのパラメータ"""
    
    # LASSOモデル固有のパラメータ
    alpha: float = 0.001  # 正則化の強さ
    max_iter: int = 100000  # 最大反復回数
    tol: float = 0.00001  # 収束条件の閾値
    
    # 特徴量選択関連
    max_features: int = 5  # 採用する特徴量の最大数
    min_features: int = 3  # 採用する特徴量の最小数
    
    # アルファ探索の範囲
    min_alpha: float = 0.000005
    max_alpha: float = 0.005
    alpha_search_iterations: int = 3  # ランダム探索の試行回数