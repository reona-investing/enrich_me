import os
import pickle
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Callable

from machine_learning.models import BaseModel, LassoModel, LgbmModel
from machine_learning.params import BaseParams, LassoParams, LgbmParams


class ModelCollection:
    """機械学習モデルのコレクション"""
    
    def __init__(self, path: Optional[str] = None):
        """
        モデルコレクションを初期化
        
        Args:
            path: コレクションの保存先パス（省略可）
        """
        self.models: Dict[str, BaseModel] = {}
        self.path = path
        
    def get_model(self, name: str) -> BaseModel:
        """
        名前を指定してモデルを取得する
        
        Args:
            name: モデル名
            
        Returns:
            指定された名前のモデルインスタンス
            
        Raises:
            KeyError: 指定された名前のモデルが存在しない場合
        """
        if name not in self.models:
            raise KeyError(f"モデル '{name}' はコレクションに存在しません。")
        return self.models[name]
    
    def set_model(self, model: BaseModel) -> None:
        """
        モデルをコレクションに追加/更新する
        
        Args:
            model: 追加/更新するモデルインスタンス
        """
        self.models[model.name] = model
    
    def get_models(self) -> List[Tuple[str, BaseModel]]:
        """
        全モデルの名前とインスタンスのリストを取得する
        
        Returns:
            (モデル名, モデルインスタンス)のタプルのリスト
        """
        return [(name, model) for name, model in self.models.items()]
    
    def set_models(self, models: List[BaseModel]) -> None:
        """
        モデルをコレクションに追加/更新する
        
        Args:
            model: 追加/更新するモデルインスタンス
        """
        for model in models:
            self.models[model.name] = model

    def generate_model(self, name: str, type: str, params: Optional[BaseParams] = None) -> BaseModel:
        """
        新しいモデルを生成してコレクションに追加する。既に同名のモデルが存在する場合は、
        既存のモデルを返す。
        
        Args:
            name: モデル名
            type: モデルタイプ ('lasso' or 'lgbm')
            params: モデルパラメータ（省略時はデフォルト）
            
        Returns:
            生成されたモデルのインスタンス、または既存のモデルのインスタンス
        
        Raises:
            ValueError: 不明なモデルタイプの場合
        """
        # 既存のモデルがあれば、それを返す
        if name in self.models:
            return self.models[name]
        
        # 新しいモデルを作成
        if type.lower() == 'lasso':
            model = LassoModel(name, params or LassoParams())
        elif type.lower() == 'lgbm':
            model = LgbmModel(name, params or LgbmParams())
        else:
            raise ValueError(f"不明なモデルタイプです: {type}. 'lasso'または'lgbm'を指定してください。")
        
        self.models[name] = model
        return model
 
    def generate_models(self, list_to_generate:list[Tuple[str, str, Optional[BaseParams]]] = None) -> list[BaseModel]:
        """
        新しいモデルを生成してコレクションに追加する
        
        Args:
            generates: [モデル名, モデルタイプ, モデルパラメータ（省略時はデフォルト）]
            
        Returns:
            生成されたモデルのインスタンスをまとめたリスト
        
        Raises:
            ValueError: 不明なモデルタイプの場合
        """
        models = []
        for generate_param in list_to_generate:
            models.append(self.generate_model(generate_param[0], generate_param[1], generate_param[2]))
        
        return models
       
    def train_all(self) -> None:
        """全てのモデルを学習する"""
        for model in self.models.values():
            model.train()
    
    def predict_all(self) -> None:
        """全てのモデルで予測を実行する"""
        for model in self.models.values():
            model.predict()
    
    def get_result_df(self) -> pd.DataFrame:
        """
        全てのモデルの予測結果を結合する
        
        Returns:
            結合された予測結果のデータフレーム
            
        Raises:
            ValueError: 予測結果が存在しない場合
        """
        result_dfs = []
        for model in self.models.values():
            if model.pred_result_df is not None:
                result_dfs.append(model.pred_result_df)
        
        if not result_dfs:
            raise ValueError("予測結果が存在しません。predict_all()を先に実行してください。")
        
        return pd.concat(result_dfs)
    
    def set_result_df(self, result_df: pd.DataFrame, separate_by_sector: bool = True):
        """
        全てのモデルの予測結果を結合する
        
        Args:
            結合された予測結果のデータフレーム
            
        Raises:
            ValueError: 予測結果が存在しない場合
        """
        for sector, model in self.models.items():
            # セクターでフィルタリングして設定
            if model.pred_result_df.index.nlevels > 1 and \
                "Sector" in model.pred_result_df.index.names and \
                    separate_by_sector:
                model.pred_result_df = result_df[result_df.index.get_level_values('Sector') == sector]
            else:
                model.pred_result_df = result_df        
    
    def set_train_test_data_all(self, 
                                target_df: pd.DataFrame, 
                                features_df: pd.DataFrame,
                                train_start_date: datetime,
                                train_end_date: datetime,
                                test_start_date: Optional[datetime] = None,
                                test_end_date: Optional[datetime] = None,
                                outlier_threshold: float = 0,
                                no_shift_features: List[str] = None,
                                get_next_open_date_func: Optional[Callable] = None,
                                reuse_features_df: bool = False,
                                separate_by_sector: bool =True) -> None:
        

        for sector, model in self.models.items():
            # セクターでフィルタリングして設定
            if target_df.index.nlevels > 1 and "Sector" in target_df.index.names and separate_by_sector:
                sector_target = target_df[target_df.index.get_level_values('Sector') == sector]
            else:
                sector_target = target_df

            if features_df.index.nlevels > 1 and "Sector" in features_df.index.names and separate_by_sector:
                sector_features = features_df[features_df.index.get_level_values('Sector') == sector]
            else:
                sector_features = features_df
            
            model.load_dataset(sector_target, sector_features,
                               train_start_date, train_end_date, test_start_date, test_end_date,
                               outlier_threshold, no_shift_features, get_next_open_date_func, reuse_features_df)

    # 新規メソッド: 生の目的変数を全モデルに設定
    def set_raw_target_for_all(self, raw_target_df: pd.DataFrame) -> None:
        """
        すべてのモデルに生の目的変数を設定する
        
        Args:
            raw_target_df: 生の目的変数のデータフレーム
        """
        for model in self.models.values():
            sector = model.name
            # セクターでフィルタリングして設定
            if raw_target_df.index.nlevels > 1 and "Sector" in raw_target_df.index.names:
                sector_data = raw_target_df[raw_target_df.index.get_level_values('Sector') == sector]
                model.set_raw_target(sector_data)
            else:
                # セクター情報がない場合はそのまま設定
                model.set_raw_target(raw_target_df)
    
    # 新規メソッド: 発注価格情報を全モデルに設定
    def set_order_price_for_all(self, order_price_df: pd.DataFrame) -> None:
        """
        すべてのモデルに発注価格情報を設定する
        
        Args:
            order_price_df: 発注価格のデータフレーム
        """
        for model in self.models.values():
            sector = model.name
            # セクターでフィルタリングして設定
            if order_price_df.index.nlevels > 1 and "Sector" in order_price_df.index.names:
                sector_data = order_price_df[order_price_df.index.get_level_values('Sector') == sector]
                model.set_order_price(sector_data)
            else:
                # セクター情報がない場合はそのまま設定
                model.set_order_price(order_price_df)
    
    # 新規メソッド: 全モデルの生の目的変数を取得
    def get_raw_target_df(self) -> pd.DataFrame:
        """
        全てのモデルの生の目的変数を結合する
        
        Returns:
            結合された生の目的変数データフレーム
        
        Raises:
            ValueError: 生の目的変数が存在しない場合
        """
        result_dfs = []
        for model in self.models.values():
            if model.raw_target_df is not None:
                result_dfs.append(model.raw_target_df)
        
        if not result_dfs:
            raise ValueError("生の目的変数が存在しません。set_raw_target_for_all()を先に実行してください。")
        
        return pd.concat(result_dfs)
    
    # 新規メソッド: 全モデルの発注価格情報を取得
    def get_order_price_df(self) -> pd.DataFrame:
        """
        全てのモデルの発注価格情報を結合する
        
        Returns:
            結合された発注価格データフレーム
        
        Raises:
            ValueError: 発注価格情報が存在しない場合
        """
        result_dfs = []
        for model in self.models.values():
            if model.order_price_df is not None:
                result_dfs.append(model.order_price_df)
        
        if not result_dfs:
            raise ValueError("発注価格情報が存在しません。set_order_price_for_all()を先に実行してください。")
        
        return pd.concat(result_dfs)

    def set_params_all(self, params: BaseParams):
        for model in self.models.values():
            model.params = params

    def get_params_all(self) -> list[BaseParams]:
        params_list = []
        for model in self.models.values():
            params_list.append(model.params)
        return params_list

    def save(self, path: Optional[str] = None) -> None:
        """
        コレクションをファイルに保存する
        
        Args:
            path: 保存先のファイルパス（省略時はインスタンス作成時のパスを使用）
        """
        # 保存先パスの決定
        save_path = path if path is not None else self.path
        if save_path is None:
            raise ValueError("保存先のパスが指定されていません。save(path)を呼び出すか、初期化時にパスを指定してください。")
            
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存用の辞書を作成
        collection_data = {
            'models': self.models,
            # 生の目的変数と発注価格情報も保存
            'raw_target_data': {name: model.raw_target_df for name, model in self.models.items() if hasattr(model, 'raw_target_df') and model.raw_target_df is not None},
            'order_price_data': {name: model.order_price_df for name, model in self.models.items() if hasattr(model, 'order_price_df') and model.order_price_df is not None},
        }
        
        # ファイルに保存
        with open(save_path, 'wb') as file:
            pickle.dump(collection_data, file)
        
        print(f"モデルコレクションを {save_path} に保存しました。")
    
    @classmethod
    def load(cls, path: str) -> 'ModelCollection':
        """
        ファイルからコレクションを読み込む
        
        Args:
            path: 読み込むファイルパス
            
        Returns:
            読み込まれたモデルコレクション
            
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            ValueError: ファイルが正しくない形式の場合
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")
        
        # ファイルから読み込み
        with open(path, 'rb') as file:
            collection_data = pickle.load(file)
        
        # コレクションの作成
        collection = cls(path)
        
        # モデルの設定
        if 'models' in collection_data:
            collection.models = collection_data['models']
        else:
            raise ValueError(f"ファイル {path} が正しくない形式です。")
        
        # 生の目的変数の復元（存在する場合）
        if 'raw_target_data' in collection_data:
            raw_target_data: Dict[str, pd.DataFrame] = collection_data['raw_target_data']
            for name, raw_target_df in raw_target_data.items():
                if name in collection.models:
                    collection.models[name].raw_target_df = raw_target_df
                    
        # 発注価格データの復元（存在する場合）
        if 'order_price_data' in collection_data:
            order_price_data: Dict[str, pd.DataFrame] = collection_data['order_price_data']
            for name, order_price_df in order_price_data.items():
                if name in collection.models:
                    collection.models[name].order_price_df = order_price_df
        
        print(f"モデルコレクションを {path} から読み込みました。")
        
        return collection