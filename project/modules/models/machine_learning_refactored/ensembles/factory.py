from typing import List, Dict
from models.machine_learning.ensembles.base_ensemble_method import BaseEnsembleMethod
from models.machine_learning.ensembles.by_rank import ByRankMethod
from models.machine_learning.ensembles.by_predict_value import ByPredictValueMethod
from models.machine_learning.ensembles.by_voting import ByVotingMethod

class EnsembleMethodFactory:
    """
    アンサンブル手法のインスタンスを生成・管理するためのファクトリークラス。

    事前に登録されたアンサンブル手法を名前で取得したり、
    新たな手法を登録するための機能を提供する。

    デフォルトで利用可能なアンサンブル手法は以下の通り:
        - 'by_rank': 予測値のランクに基づく手法
        - 'by_predict_value': 予測値の大小に基づく手法
        - 'by_voting': クラスごとの投票に基づく手法
    """
    
    _methods: Dict[str, BaseEnsembleMethod] = {
        'by_rank': ByRankMethod(),
        'by_predict_value': ByPredictValueMethod(),
        'by_voting': ByVotingMethod(),
    }
    
    @classmethod
    def create_method(cls, method_name: str, **kwargs) -> BaseEnsembleMethod:
        """
        指定された名前のアンサンブル手法を作成
        
        Args:
            method_name (str): アンサンブル手法の名前
            **kwargs: アンサンブル手法固有のパラメータ
            
        Returns:
            BaseEnsembleMethod: アンサンブル手法のインスタンス
            
        Raises:
            ValueError: 存在しない手法名が指定された場合
        """
        if method_name not in cls._methods:
            available_methods = list(cls._methods.keys())
            raise ValueError(f"未知のアンサンブル手法: {method_name}. "
                           f"利用可能な手法: {available_methods}")
        
        # パラメータが指定されている場合は新しいインスタンスを作成
        if kwargs:
            if method_name == 'by_voting':
                return ByVotingMethod(**kwargs)
            # 他の手法でもパラメータ対応が必要な場合はここに追加
        
        return cls._methods[method_name]
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """利用可能なアンサンブル手法の一覧を取得"""
        return list(cls._methods.keys())
    
    @classmethod
    def register_method(cls, name: str, method: BaseEnsembleMethod) -> None:
        """新しいアンサンブル手法を登録"""
        cls._methods[name] = method