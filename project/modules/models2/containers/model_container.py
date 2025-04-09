"""
複数のモデルを管理するコンテナクラス
"""
from typing import Dict, List, Union, Optional, Any, Tuple
import pandas as pd
import os
import pickle

from models2.base.base_model import BaseModel
from models2.base.base_container import BaseContainer


class ModelContainer(BaseContainer):
    """
    複数のモデルを管理するコンテナクラス
    """
    
    def __init__(self, name: str = ""):
        """
        初期化
        
        Args:
            name: コンテナの名前（任意）
        """
        super().__init__(name=name)
        self.models: Dict[str, BaseModel] = {}
        self._feature_importances: Dict[str, pd.DataFrame] = {}
    
    def add_model(self, key: str, model: BaseModel):
        """
        モデルをコンテナに追加
        
        Args:
            key: モデルの識別子（セクター名など）
            model: 追加するモデルインスタンス
        """
        self.models[key] = model
    
    def get_model(self, key: str) -> Optional[BaseModel]:
        """
        特定のキーに対応するモデルを取得
        
        Args:
            key: モデルの識別子
            
        Returns:
            BaseModel: 対応するモデルのインスタンス（存在しない場合はNone）
        """
        return self.models.get(key)
    
    def get_keys(self) -> List[str]:
        """登録されているモデルのキー一覧を取得"""
        return list(self.models.keys())
    
    def get_feature_importances(self, key: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        特徴量重要度を取得
        
        Args:
            key: 特定のモデルの識別子（指定がなければ全モデルの特徴量重要度を辞書で返す）
            
        Returns:
            pd.DataFrame または Dict[str, pd.DataFrame]: 特徴量重要度
        """
        if key is not None:
            model = self.get_model(key)
            if model is None:
                raise KeyError(f"キー '{key}' に対応するモデルが見つかりません")
            return model.feature_importances
        
        # 全モデルの特徴量重要度を収集
        result = {}
        for key, model in self.models.items():
            if model.feature_importances is not None:
                result[key] = model.feature_importances
        
        return result
    
    def train(self, X: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
              y: Union[pd.Series, pd.DataFrame, Dict[str, Union[pd.Series, pd.DataFrame]]], 
              params: Optional[Dict[str, Any]] = None,
              **kwargs):
        """
        コンテナ内の全モデルを学習する
        
        Args:
            X: 特徴量データ。キー別のDataFrameの辞書またはマルチインデックスDataFrame
            y: 目的変数データ。キー別のDataFrameの辞書またはマルチインデックスDataFrame
            params: モデル共通のパラメータ辞書（オプション）
            **kwargs: その他の学習パラメータ
            
        Returns:
            self: メソッドチェーン用
        """
        # パラメータの準備
        if params is None:
            params = {}
        
        # Xが辞書の場合
        if isinstance(X, dict):
            # 各キーに対応するデータで個別にモデルを学習
            for key, model in self.models.items():
                if key in X and key in y:
                    model.train(X[key], y[key], **{**params, **kwargs})
        
        # Xがマルチインデックスのpd.DataFrameの場合
        elif isinstance(X, pd.DataFrame) and X.index.nlevels > 1:
            # 想定されるキーの位置（通常はレベル1がセクター）
            level = kwargs.pop('level', 1)
            key_name = X.index.names[level]
            
            # 登録されている各モデルに対して
            for key, model in self.models.items():
                # キーに対応するデータを抽出
                if key in X.index.get_level_values(level):
                    # キーに対応する行を取得
                    X_subset = X.xs(key, level=level, drop_level=False)
                    
                    # yもDataFrameの場合は同様に処理
                    if isinstance(y, pd.DataFrame) and y.index.nlevels > 1:
                        y_subset = y.xs(key, level=level, drop_level=False)
                    else:
                        # yがシリーズやシングルインデックスの場合は警告
                        print(f"Warning: y is not multi-indexed but X is. Using full y for model {key}.")
                        y_subset = y
                    
                    # モデルを学習
                    model.train(X_subset, y_subset, **{**params, **kwargs})
        
        # その他の場合（単一のDataFrame/Series）
        else:
            # 単一のデータで各モデルを学習
            # 注意: 通常はマルチセクターでこのパターンは使用しない
            for key, model in self.models.items():
                model.train(X, y, **{**params, **kwargs})
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        コンテナ内のモデルで予測を行い、結果を結合して返す
        
        Args:
            X: 特徴量データ。キー別のDataFrameの辞書またはマルチインデックスDataFrame
            
        Returns:
            pd.DataFrame: 全モデルの予測結果を結合したDataFrame
        """
        predictions = {}
        
        # Xが辞書の場合
        if isinstance(X, dict):
            for key, model in self.models.items():
                if key in X:
                    # キーに対応するデータで予測
                    X_subset = X[key]
                    predictions[key] = pd.DataFrame({
                        'Pred': model.predict(X_subset)
                    }, index=X_subset.index)
        
        # Xがマルチインデックスのpd.DataFrameの場合
        elif isinstance(X, pd.DataFrame) and X.index.nlevels > 1:
            # デフォルトでレベル1をキーとして使用（通常はセクター）
            level = 1
            key_name = X.index.names[level]
            
            all_predictions = []
            
            for key, model in self.models.items():
                # キーに対応するデータを抽出
                if key in X.index.get_level_values(level):
                    X_subset = X.xs(key, level=level, drop_level=False)
                    
                    # 予測実行
                    preds = model.predict(X_subset)
                    
                    # 結果をデータフレームに格納
                    pred_df = pd.DataFrame({
                        'Pred': preds
                    }, index=X_subset.index)
                    
                    all_predictions.append(pred_df)
            
            # すべての予測を連結
            if all_predictions:
                return pd.concat(all_predictions)
            else:
                return pd.DataFrame()
        
        # その他の場合（単一のDataFrame）
        else:
            # これは例外的なケース - 全モデルで同じ入力を使用
            all_predictions = {}
            
            for key, model in self.models.items():
                preds = model.predict(X)
                all_predictions[key] = preds
            
            # 予測結果を列として結合
            return pd.DataFrame(all_predictions, index=X.index)
        
        # 予測結果をデータフレームに変換して返す
        if predictions:
            # 辞書からのケース
            result_df = pd.concat(predictions.values())
            return result_df
        else:
            return pd.DataFrame()  # 空のケース
    
    def _get_state_dict(self) -> Dict[str, Any]:
        """コンテナの状態を辞書形式で取得する"""
        state = super()._get_state_dict()
        
        # モデルキーの保存
        state['model_keys'] = list(self.models.keys())
        
        return state
    
    def _set_state_dict(self, state_dict: Dict[str, Any]):
        """辞書からコンテナの状態を復元する"""
        super()._set_state_dict(state_dict)
        
        # モデルは_import_modelsで復元されるため、ここでは何もしない
        pass
    
    def _export_models(self, models_dir: str):
        """コンテナ内のモデルをエクスポートする"""
        for key, model in self.models.items():
            # モデル固有のディレクトリを作成
            model_dir = os.path.join(models_dir, key)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # モデルをシリアライズして保存
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
    
    def _import_models(self, models_dir: str):
        """保存されたモデルをインポートする"""
        # ディレクトリ内の各モデルをロード
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir, "model.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    self.models[model_dir] = model