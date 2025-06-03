import pandas as pd
from typing import Dict, List, Optional
from models.machine_learning_refactored.ml_dataset.single_ml_dataset import SingleMLDataset

class MLDatasets:
    """複数のSingleMLDatasetを統合管理するクラス"""
    
    def __init__(self):
        self.named_models: Dict[str, SingleMLDataset] = {}
    
    def append_model(self, single_ml_dataset: SingleMLDataset):
        """SingleMLDatasetを追加"""
        if not isinstance(single_ml_dataset, SingleMLDataset):
            raise TypeError("single_ml_datasetにはSingleMLDatasetインスタンスを指定してください。")
        
        model_name = single_ml_dataset.get_name()
        self.named_models[model_name] = single_ml_dataset
    
    def remove_model(self, model_name: str):
        """指定された名前のモデルを削除"""
        if model_name in self.named_models:
            del self.named_models[model_name]
        else:
            raise KeyError(f"モデル '{model_name}' が見つかりません。")
    
    def replace_model(self, single_ml_dataset: SingleMLDataset):
        """既存のモデルを差し替え（同じ名前のモデルが存在する場合）"""
        model_name = single_ml_dataset.get_name()
        if model_name in self.named_models:
            self.named_models[model_name] = single_ml_dataset
        else:
            raise KeyError(f"差し替え対象のモデル '{model_name}' が見つかりません。")
    
    def get_model(self, model_name: str) -> SingleMLDataset:
        """指定された名前のモデルを取得"""
        if model_name in self.named_models:
            return self.named_models[model_name]
        else:
            raise KeyError(f"モデル '{model_name}' が見つかりません。")
    
    def get_model_names(self) -> List[str]:
        """登録されているモデル名の一覧を取得"""
        return list(self.named_models.keys())
    
    def get_pred_result(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        各SingleMLDatasetの予測結果dfをconcatして日付順に並べたものを出力
        
        Args:
            model_names (Optional[List[str]]): 対象とするモデル名のリスト。
                                              Noneの場合は全モデルを対象とする。
        
        Returns:
            pd.DataFrame: 統合された予測結果データフレーム
        """
        if not self.named_models:
            raise ValueError("登録されているモデルがありません。")
        
        # 対象モデルの決定
        target_models = model_names if model_names is not None else list(self.named_models.keys())
        
        # 存在チェック
        for model_name in target_models:
            if model_name not in self.named_models:
                raise KeyError(f"モデル '{model_name}' が見つかりません。")
        
        pred_results = []
        for model_name in target_models:
            model = self.named_models[model_name]
            try:
                # 予測結果データフレームを取得
                pred_result_df = model.evaluation_materials.pred_result_df.copy()
                # モデル名の列を追加
                pred_result_df['Model'] = model_name
                pred_results.append(pred_result_df)
            except AttributeError:
                print(f"警告: モデル '{model_name}' に予測結果データが見つかりません。スキップします。")
                continue
        
        if not pred_results:
            raise ValueError("有効な予測結果データが見つかりませんでした。")
        
        # 結合して日付順にソート
        combined_df = pd.concat(pred_results, axis=0)
        
        # インデックスがDateを含む場合は日付でソート
        if 'Date' in combined_df.index.names:
            combined_df = combined_df.sort_index()
        elif hasattr(combined_df.index, 'get_level_values') and 'Date' in combined_df.index.get_level_values(0):
            combined_df = combined_df.sort_index()
        
        return combined_df
    
    def save_all(self):
        """全てのモデルを保存"""
        for model in self.named_models.values():
            model.save()
    
    def __len__(self) -> int:
        """登録されているモデル数を返す"""
        return len(self.named_models)
    
    def __contains__(self, model_name: str) -> bool:
        """指定されたモデル名が登録されているかチェック"""
        return model_name in self.named_models
    
    def __iter__(self):
        """イテレータ（モデル名でイテレート）"""
        return iter(self.named_models.keys())
    
    def items(self):
        """(モデル名, SingleMLDataset)のペアを返す"""
        return self.named_models.items()