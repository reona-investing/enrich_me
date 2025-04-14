import pandas as pd
from datetime import datetime
from typing import Optional, List

from machine_learning.strategies.base_strategy import Strategy
from machine_learning.core.collection import ModelCollection
from machine_learning.ensemble.methods import by_rank, weighted_average


class EnsembleStrategy(Strategy):
    """複数の戦略/モデルをアンサンブルする戦略"""
    
    def __init__(self, name: str = "ensemble", save_path: Optional[str] = None):
        """
        Args:
            name: 戦略名
            save_path: 保存先パス（省略可）
        """
        super().__init__(name, save_path)
        self.sub_collections = []
        self.weights = []
        self.ensemble_method = "rank"  # "rank" or "average"
    
    def add_collection(self, collection: ModelCollection, weight: float = 1.0) -> None:
        """
        アンサンブル対象のコレクションを追加する
        
        Args:
            collection: モデルコレクション
            weight: アンサンブル時の重み
        """
        self.sub_collections.append(collection)
        self.weights.append(weight)
    
    def prepare_data(self, 
                   train_start_date: datetime = None,
                   train_end_date: datetime = None,
                   test_start_date: Optional[datetime] = None,
                   test_end_date: Optional[datetime] = None,
                   **kwargs) -> None:
        """
        データの前処理を行う
        
        Args:
            train_start_date: 学習データの開始日（アンサンブル戦略では不要）
            train_end_date: 学習データの終了日（アンサンブル戦略では不要）
            test_start_date: テストデータの開始日（アンサンブル戦略では不要）
            test_end_date: テストデータの終了日（アンサンブル戦略では不要）
            **kwargs: その他のパラメータ
                - ensemble_method: アンサンブル手法（"rank"または"average"）
        """
        # アンサンブル戦略の場合、各サブコレクションが既にデータを準備済みであることを前提とする
        if not self.sub_collections:
            raise ValueError("アンサンブル対象のコレクションが追加されていません。add_collection()を先に実行してください。")
        
        # アンサンブル手法の設定
        self.ensemble_method = kwargs.get("ensemble_method", "rank")
        
        # モデルコレクションの初期化
        self.collection = ModelCollection(name=self.name, path=self.save_path)
    
    def train(self) -> None:
        """モデルの学習を行う"""
        # アンサンブル戦略の場合、各サブコレクションが既に学習済みであることを前提とする
        if not self.sub_collections:
            raise ValueError("アンサンブル対象のコレクションが追加されていません。add_collection()を先に実行してください。")
        
        # 各サブコレクションが学習されていることを確認
        for collection in self.sub_collections:
            for model_name, model in collection.get_models():
                if not hasattr(model, 'trained') or not model.trained:
                    raise ValueError(f"モデル {model_name} が学習されていません。")
        
        self.trained = True
    
    def predict(self) -> pd.DataFrame:
        """予測を実行する"""
        if not self.trained:
            raise ValueError("学習が完了していません。train()を先に実行してください。")
        
        # 各サブコレクションの予測結果を取得
        pred_dfs = []
        for collection, weight in zip(self.sub_collections, self.weights):
            collection.predict_all()
            pred_df = collection.get_result_df()
            pred_dfs.append((pred_df, weight))
        
        # アンサンブル手法に応じて予測結果を統合
        if self.ensemble_method == "rank":
            ensembled_pred_df = by_rank(pred_dfs)
        elif self.ensemble_method == "average":
            ensembled_pred_df = weighted_average(pred_dfs)
        else:
            raise ValueError(f"不明なアンサンブル手法です: {self.ensemble_method}")
        
        # 生の目的変数と予測値を含むデータフレームを作成
        result_df = pred_dfs[0][0][['Target']].copy()
        result_df['Pred'] = ensembled_pred_df['Pred']
        
        # 結果を保存
        for model_name, model in self.sub_collections[0].get_models():
            model_copy = model.__class__(name=f"ensemble_{model_name}")
            model_copy.pred_result_df = result_df
            model_copy.raw_target_df = model.raw_target_df
            model_copy.order_price_df = model.order_price_df
            self.collection.add_model(model_copy)
        
        return result_df
    
    @classmethod
    def create_ensemble(cls, 
                       path: str,
                       collection_paths: List[str], 
                       weights: Optional[List[float]] = None,
                       ensemble_method: str = "rank") -> 'EnsembleStrategy':
        """
        アンサンブル戦略を生成する専用ファクトリーメソッド
        
        Args:
            path: 保存先パス
            collection_paths: アンサンブル対象のコレクションのパスのリスト
            weights: 各コレクションの重み（省略時は均等配分）
            ensemble_method: アンサンブル手法（"rank"または"average"）
            
        Returns:
            アンサンブル戦略
        """
        # コレクションパスのチェック
        if not collection_paths:
            raise ValueError("アンサンブル対象のコレクションパスが指定されていません。")
        
        # 重みのチェック
        if weights is None:
            weights = [1.0] * len(collection_paths)
        elif len(weights) != len(collection_paths):
            weights = [1.0] * len(collection_paths)
        
        # 戦略インスタンスを作成
        strategy = cls(name="ensemble", save_path=path)
        
        # 各サブコレクションを読み込み
        for collection_path, weight in zip(collection_paths, weights):
            try:
                collection = ModelCollection.load(collection_path)
                strategy.add_collection(collection, weight)
            except FileNotFoundError:
                raise ValueError(f"コレクションファイルが見つかりません: {collection_path}")
        
        # データを準備
        strategy.prepare_data(ensemble_method=ensemble_method)
        
        # モデルを学習（サブコレクションは既に学習済みと想定）
        strategy.train()
        
        # 予測を実行
        strategy.predict()
        
        # 保存
        strategy.save()
        
        return strategy
    
    @classmethod
    def run(cls, 
           path: str,
           target_df: pd.DataFrame, 
           features_df: pd.DataFrame,
           raw_target_df: pd.DataFrame, 
           order_price_df: pd.DataFrame,
           train_start_date: datetime, 
           train_end_date: datetime,
           test_start_date: Optional[datetime] = None, 
           test_end_date: Optional[datetime] = None,
           train: bool = True,
           **kwargs) -> ModelCollection:
        """
        アンサンブル戦略を実行する
        
        本メソッドは戦略クラスの一貫性のために維持していますが、
        アンサンブル戦略ではcreate_ensemble()メソッドの使用を推奨します。
        
        Args:
            path: モデルの保存/読み込み先パス
            target_df: 目的変数のデータフレーム（アンサンブル戦略では不要）
            features_df: 特徴量のデータフレーム（アンサンブル戦略では不要）
            raw_target_df: 生の目的変数のデータフレーム（アンサンブル戦略では不要）
            order_price_df: 発注価格のデータフレーム（アンサンブル戦略では不要）
            train_start_date: 学習データの開始日（アンサンブル戦略では不要）
            train_end_date: 学習データの終了日（アンサンブル戦略では不要）
            test_start_date: テストデータの開始日（アンサンブル戦略では不要）
            test_end_date: テストデータの終了日（アンサンブル戦略では不要）
            train: 学習を行うかどうか（アンサンブル戦略では常にTrue）
            **kwargs: その他のパラメータ
                - collection_paths: アンサンブル対象のコレクションのパスのリスト（必須）
                - weights: 各コレクションの重み
                - ensemble_method: アンサンブル手法（"rank"または"average"）
            
        Returns:
            モデルコレクション
        """
        collection_paths = kwargs.get('collection_paths', [])
        weights = kwargs.get('weights', [1.0] * len(collection_paths))
        ensemble_method = kwargs.get('ensemble_method', 'rank')
        
        # create_ensemble()メソッドを利用
        strategy = cls.create_ensemble(
            path=path, 
            collection_paths=collection_paths,
            weights=weights,
            ensemble_method=ensemble_method
        )
        
        return strategy.collection