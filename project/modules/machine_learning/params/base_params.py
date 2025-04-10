from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class BaseParams:
    """機械学習モデルのパラメータの基底クラス"""
    
    # 共通パラメータ
    random_seed: int = 42
    
    # その他のパラメータを辞書として保持
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """パラメータを辞書として取得"""
        result = {k: v for k, v in self.__dict__.items() if k != 'extra_params'}
        result.update(self.extra_params)
        return result
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'BaseParams':
        """辞書からパラメータオブジェクトを生成"""
        # クラスの定義済みフィールドを取得
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        
        # 定義済みフィールドとそれ以外を分離
        known_params = {k: v for k, v in params_dict.items() if k in field_names}
        extra_params = {k: v for k, v in params_dict.items() if k not in field_names}
        
        # インスタンス生成と余分なパラメータの設定
        instance = cls(**known_params)
        instance.extra_params = extra_params
        return instance