import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

from machine_learning.core.collection import ModelCollection
from machine_learning.core.model_base import ModelBase
from machine_learning.ensemble.methods import by_rank, weighted_average


# グローバルスコープでEnsembleModelを定義
class EnsembleModel(ModelBase):
    """アンサンブル結果を格納するためのモデルクラス"""
    
    def train(self) -> None:
        """このモデルは学習不要（アンサンブル結果のみ保持）"""
        self.trained = True
    
    def predict(self) -> pd.DataFrame:
        """予測結果を返す（既に計算済み）"""
        return self.pred_result_df


class EnsembleUtility:
    """
    複数の予測結果をアンサンブル（合成）するためのユーティリティ
    """
    
    @staticmethod
    def ensemble_collections(collections: List[Tuple[ModelCollection, float]], 
                            method: str = "rank", 
                            output_path: Optional[str] = None,
                            model_name: str = "EnsembleModel") -> ModelCollection:
        """
        複数のモデルコレクションの予測結果をアンサンブルする
        
        Args:
            collections: (モデルコレクション, 重み)のタプルのリスト
            method: アンサンブル手法（"rank"または"average"）
            output_path: 結果の保存先パス（省略可）
            model_name: アンサンブルモデルの名前
            
        Returns:
            アンサンブル結果を格納したモデルコレクション
        """
        # コレクションが指定されているか確認
        if not collections:
            raise ValueError("アンサンブル対象のコレクションが指定されていません。")
        
        # 各コレクションの予測結果を取得
        pred_dfs = []
        for collection, weight in collections:
            pred_df = collection.get_result_df()
            pred_dfs.append((pred_df, weight))
        
        # アンサンブル結果を計算
        ensembled_result = EnsembleUtility.ensemble_dataframes(pred_dfs, method)
        
        # 新しいモデルコレクションを作成
        output_collection = ModelCollection(name="ensemble", path=output_path)
        
        # 単一のアンサンブルモデルを生成（重複を避けるため）
        ensemble_model = EnsembleUtility._create_ensemble_model(
            model_name=model_name,
            ensembled_result=ensembled_result,
            original_collection=collections[0][0]  # 元データ構造のリファレンス
        )
        
        # コレクションにモデルを追加
        output_collection.add_model(ensemble_model)
        
        # 元のコレクションから生の目的変数と発注価格を取得して設定
        # raw_target_dfの設定
        if hasattr(collections[0][0].models[next(iter(collections[0][0].models))], 'raw_target_df'):
            output_collection.set_raw_target_for_all(
                collections[0][0].get_raw_targets(), 
                separate_by_sector=False  # アンサンブルモデルは単一のため分割しない
            )
        
        # order_price_dfの設定
        if hasattr(collections[0][0].models[next(iter(collections[0][0].models))], 'order_price_df'):
            output_collection.set_order_price_for_all(
                collections[0][0].get_order_prices()
            )
        
        # 保存パスが指定されていれば保存
        if output_path:
            output_collection.save()
        
        return output_collection
    
    @staticmethod
    def ensemble_dataframes(pred_dfs: List[Tuple[pd.DataFrame, float]], 
                           method: str = "rank") -> pd.DataFrame:
        """
        複数の予測結果のデータフレームをアンサンブルする
        
        Args:
            pred_dfs: (予測結果データフレーム, 重み)のタプルのリスト
            method: アンサンブル手法（"rank"または"average"）
            
        Returns:
            アンサンブル後の予測結果のデータフレーム
        """
        # データフレームが指定されているか確認
        if not pred_dfs:
            raise ValueError("アンサンブル対象のデータフレームが指定されていません。")
        
        # アンサンブル処理
        if method == "rank":
            ensembled_pred_df = by_rank(pred_dfs)
        elif method == "average":
            ensembled_pred_df = weighted_average(pred_dfs)
        else:
            raise ValueError(f"不明なアンサンブル手法です: {method}")
        
        # 生の目的変数と予測値を含むデータフレームを作成
        # 最初のデータフレームから目的変数を取得（存在する場合）
        if 'Target' in pred_dfs[0][0].columns:
            result_df = pred_dfs[0][0][['Target']].copy()
            result_df['Pred'] = ensembled_pred_df['Pred']
            return result_df
        else:
            return ensembled_pred_df
    
    @staticmethod
    def load_and_ensemble(collection_paths: List[str], 
                         weights: Optional[List[float]] = None,
                         method: str = "rank",
                         output_path: Optional[str] = None,
                         model_name: str = "EnsembleModel") -> ModelCollection:
        """
        複数のモデルコレクションをパスから読み込んでアンサンブルする
        
        Args:
            collection_paths: アンサンブル対象のコレクションのパスのリスト
            weights: 各コレクションの重み（省略時は均等配分）
            method: アンサンブル手法（"rank"または"average"）
            output_path: 結果の保存先パス（省略可）
            model_name: アンサンブルモデルの名前
            
        Returns:
            アンサンブル結果を格納したモデルコレクション
        """
        # コレクションパスのチェック
        if not collection_paths:
            raise ValueError("アンサンブル対象のコレクションパスが指定されていません。")
        
        # 重みのチェック
        if weights is None:
            weights = [1.0] * len(collection_paths)
        elif len(weights) != len(collection_paths):
            weights = [1.0] * len(collection_paths)
        
        # コレクションを読み込み
        collections = []
        for collection_path, weight in zip(collection_paths, weights):
            try:
                collection = ModelCollection.load(collection_path)
                collections.append((collection, weight))
            except FileNotFoundError:
                raise ValueError(f"コレクションファイルが見つかりません: {collection_path}")
        
        # アンサンブル処理
        return EnsembleUtility.ensemble_collections(
            collections=collections,
            method=method,
            output_path=output_path,
            model_name=model_name
        )
    
    @staticmethod
    def _create_ensemble_model(model_name: str, 
                              ensembled_result: pd.DataFrame, 
                              original_collection: ModelCollection) -> EnsembleModel:
        """
        アンサンブル結果を格納するモデルインスタンスを生成する
        
        Args:
            model_name: 生成するモデルの名前
            ensembled_result: アンサンブル後の予測結果データフレーム
            original_collection: 元のコレクション（データ構造参照用）
            
        Returns:
            アンサンブル結果を格納したモデルインスタンス
        """
        # モデルインスタンスを生成
        ensemble_model = EnsembleModel(name=model_name)
        ensemble_model.pred_result_df = ensembled_result
        ensemble_model.trained = True
        
        # 元の生データと発注価格を参照（存在する場合）
        # 最初のモデルから参照
        original_model = next(iter(original_collection.models.values()))
        
        if hasattr(original_model, 'raw_target_df') and original_model.raw_target_df is not None:
            ensemble_model.raw_target_df = original_model.raw_target_df
            
        if hasattr(original_model, 'order_price_df') and original_model.order_price_df is not None:
            ensemble_model.order_price_df = original_model.order_price_df
            
        return ensemble_model