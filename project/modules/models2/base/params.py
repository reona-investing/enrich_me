"""
モデル固有のハイパーパラメータを管理するデータクラス
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class LassoParams:
    """Lassoモデルのハイパーパラメータ"""
    
    # モデル探索用パラメータ
    max_features: int = 5
    min_features: int = 3
    
    # Lassoモデル自体のパラメータ
    alpha: Optional[float] = None  # 自動探索する場合はNone
    max_iter: int = 100000
    tol: float = 0.00001
    
    # その他のパラメータは**kwargsで渡せるようにする
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def get_model_params(self) -> Dict[str, Any]:
        """sklearn.Lassoに渡すパラメータを取得"""
        params = {
            "max_iter": self.max_iter,
            "tol": self.tol
        }
        
        # alphaが指定されていれば追加
        if self.alpha is not None:
            params["alpha"] = self.alpha
            
        # 追加パラメータを統合
        params.update(self.additional_params)
        
        return params


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
    
    def get_model_params(self) -> Dict[str, Any]:
        """lightgbm.trainに渡すパラメータを取得"""
        params = {
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "random_seed": self.random_seed,
            "lambda_l1": self.lambda_l1,
            "verbose": -1  # 出力を抑制
        }
        
        # 追加パラメータを統合
        params.update(self.additional_params)
        
        return params