import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from machine_learning.core.model_base import ModelBase
from machine_learning.utils.serialization import save_collection, load_collection


class ModelCollection:
    """複数モデルを管理するコレクションクラス"""
    
    def __init__(self, name: str = "model_collection", path: Optional[str] = None):
        """
        Args:
            name: コレクション名
            path: 保存先パス（省略可）
        """
        self.name = name
        self.path = path
        self.models: Dict[str, ModelBase] = {}
        
    def add_model(self, model: ModelBase) -> None:
        """モデルをコレクションに追加する"""
        self.models[model.name] = model
        
    def get_model(self, name: str) -> ModelBase:
        """名前を指定してモデルを取得する"""
        if name not in self.models:
            raise KeyError(f"モデル '{name}' はコレクションに存在しません。")
        return self.models[name]
    
    def get_models(self) -> List[Tuple[str, ModelBase]]:
        """全モデルの名前とインスタンスのタプルのリストを取得する"""
        return [(name, model) for name, model in self.models.items()]
    
    def set_train_test_data_all(self, 
                               target_df: pd.DataFrame, 
                               features_df: pd.DataFrame,
                               train_start_date: datetime,
                               train_end_date: datetime,
                               test_start_date: Optional[datetime] = None,
                               test_end_date: Optional[datetime] = None,
                               outlier_threshold: float = 0,
                               no_shift_features: List[str] = None,
                               reuse_features_df: bool = False,
                               separate_by_sector: bool = True) -> None:
        """
        すべてのモデルに対してデータセットを設定する
        
        Args:
            target_df: 目的変数のデータフレーム
            features_df: 特徴量のデータフレーム
            train_start_date: 学習データの開始日
            train_end_date: 学習データの終了日
            test_start_date: テストデータの開始日（省略時はtrain_end_date）
            test_end_date: テストデータの終了日（省略時はデータの最終日）
            outlier_threshold: 外れ値除去の閾値（±何σ、0の場合は除去なし）
            no_shift_features: シフトしない特徴量のリスト
            reuse_features_df: 特徴量を他の業種から再利用するか
            separate_by_sector: セクターごとにデータを分割するか
        """
        for model_name, model in self.models.items():
            if target_df.index.nlevels > 1 and "Sector" in target_df.index.names and separate_by_sector:
                # セクターでフィルタリング（モデル名をセクター名と解釈）
                sector_target = target_df[target_df.index.get_level_values('Sector') == model_name]
                sector_features = features_df[features_df.index.get_level_values('Sector') == model_name]
                
                model.load_dataset(
                    target_df=sector_target,
                    features_df=sector_features,
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    test_start_date=test_start_date,
                    test_end_date=test_end_date,
                    outlier_threshold=outlier_threshold,
                    no_shift_features=no_shift_features,
                    reuse_features_df=reuse_features_df
                )
            else:
                # セクター分割なし（全体データを使用）
                model.load_dataset(
                    target_df=target_df,
                    features_df=features_df,
                    train_start_date=train_start_date,
                    train_end_date=train_end_date,
                    test_start_date=test_start_date,
                    test_end_date=test_end_date,
                    outlier_threshold=outlier_threshold,
                    no_shift_features=no_shift_features,
                    reuse_features_df=reuse_features_df
                )
    
    def train_all(self) -> None:
        """すべてのモデルを学習する"""
        for model_name, model in self.models.items():
            print(f"Training model: {model_name}")
            model.train()
    
    def predict_all(self) -> None:
        """すべてのモデルで予測を実行する"""
        for model_name, model in self.models.items():
            print(f"Predicting with model: {model_name}")
            model.predict()
    
    def get_result_df(self) -> pd.DataFrame:
        """すべてのモデルの予測結果を結合する"""
        result_dfs = []
        for model in self.models.values():
            if model.pred_result_df is not None:
                result_dfs.append(model.pred_result_df)
        
        if not result_dfs:
            raise ValueError("予測結果が存在しません。predict_all()を先に実行してください。")
        
        return pd.concat(result_dfs)
        
    def get_raw_targets(self) -> pd.DataFrame:
        """すべてのモデルの生の目的変数を結合する"""
        raw_target_dfs = []
        for model in self.models.values():
            raw_target_df = model.get_raw_target()
            if raw_target_df is not None:
                raw_target_dfs.append(raw_target_df)
        
        if not raw_target_dfs:
            raise ValueError("生の目的変数が存在しません。set_raw_target_for_all()を先に実行してください。")
        
        return pd.concat(raw_target_dfs)
    
    def get_order_prices(self) -> pd.DataFrame:
        """すべてのモデルの発注価格情報を結合する"""
        order_price_dfs = []
        for model in self.models.values():
            order_price_df = model.get_order_price()
            if order_price_df is not None:
                order_price_dfs.append(order_price_df)
        
        if not order_price_dfs:
            raise ValueError("発注価格情報が存在しません。set_order_price_for_all()を先に実行してください。")
        
        return pd.concat(order_price_dfs)
    
    def set_raw_target_for_all(self, raw_target_df: pd.DataFrame, separate_by_sector: bool = True) -> None:
        """すべてのモデルに生の目的変数を設定する"""
        for model_name, model in self.models.items():
            if raw_target_df.index.nlevels > 1 and "Sector" in raw_target_df.index.names and separate_by_sector:
                # セクターでフィルタリング
                sector_data = raw_target_df[raw_target_df.index.get_level_values('Sector') == model_name]
                model.set_raw_target(sector_data)
            else:
                # セクター分割なし
                model.set_raw_target(raw_target_df)
    
    def set_order_price_for_all(self, order_price_df: pd.DataFrame) -> None:
        """すべてのモデルに発注価格情報を設定する"""
        for model in self.models.values():
            model.set_order_price(order_price_df)
    
    def save(self, path: Optional[str] = None) -> None:
        """コレクションを保存する"""
        save_path = path if path is not None else self.path
        if save_path is None:
            raise ValueError("保存先のパスが指定されていません。")
        
        save_collection(self, save_path)
        print(f"Model collection saved to {save_path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelCollection':
        """コレクションを読み込む"""
        return load_collection(path)