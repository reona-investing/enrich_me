import pandas as pd
from typing import Dict, List, Optional, Callable
from models.machine_learning.ml_dataset.single_ml_dataset import SingleMLDataset

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

    def _merge_dfs(
        self,
        getter: Callable[[SingleMLDataset], pd.DataFrame],
        model_names: Optional[List[str]],
        data_name: str
    ) -> pd.DataFrame:
        """各モデルからDataFrameを取得して結合するヘルパー"""
        if not self.named_models:
            raise ValueError("登録されているモデルがありません。")

        target_models = model_names if model_names is not None else list(self.named_models.keys())

        for model_name in target_models:
            if model_name not in self.named_models:
                raise KeyError(f"モデル '{model_name}' が見つかりません。")

        results = []
        for model_name in target_models:
            model = self.named_models[model_name]
            try:
                df = getter(model).copy()
                results.append(df)
            except AttributeError:
                print(f"警告: モデル '{model_name}' に{data_name}データが見つかりません。スキップします。")
                continue

        if not results:
            raise ValueError(f"有効な{data_name}データが見つかりませんでした。")

        combined_df = pd.concat(results, axis=0)

        if 'Date' in combined_df.index.names:
            combined_df = combined_df.sort_index()
        elif hasattr(combined_df.index, 'get_level_values') and 'Date' in combined_df.index.get_level_values(0):
            combined_df = combined_df.sort_index()

        return combined_df
    
    def get_pred_result(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        各SingleMLDatasetの予測結果dfをconcatして日付順に並べたものを出力

        Args:
            model_names (Optional[List[str]]): 対象とするモデル名のリスト。
                                              Noneの場合は全モデルを対象とする。

        Returns:
            pd.DataFrame: 統合された予測結果データフレーム
        """
        return self._merge_dfs(
            getter=lambda m: m.evaluation_materials.pred_result_df,
            model_names=model_names,
            data_name="予測結果"
        )

    def get_raw_target(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """raw_target_dfをマージして返却"""
        return self._merge_dfs(
            getter=lambda m: m.evaluation_materials.raw_target_df,
            model_names=model_names,
            data_name="raw_target"
        )

    def get_order_price(self, model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """order_price_dfをマージして返却"""
        return self._merge_dfs(
            getter=lambda m: m.stock_selection_materials.order_price_df,
            model_names=model_names,
            data_name="order_price"
        )
    
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