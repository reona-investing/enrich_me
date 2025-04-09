"""
期間ごとに異なるモデルの予測結果を結合するクラス
"""
from typing import Dict, List, Union, Tuple, Optional, Any
import pandas as pd
from datetime import datetime
import os
import pickle

from models2.base.base_model import BaseModel
from models2.containers.model_container import ModelContainer
from models2.base.base_container import BaseContainer


class PeriodSwitchingModel(BaseContainer):
    """
    期間ごとに異なるモデルの予測結果を結合するクラス。
    ある日付範囲ではモデルA、別の日付範囲ではモデルBというように、
    時間軸に沿ってモデルを切り替えて予測結果を生成します。
    """
    
    def __init__(self, name: str = "PeriodSwitchingModel"):
        """
        初期化
        
        Args:
            name: モデルの名前
        """
        super().__init__(name=name)
        self.model_periods = []  # (開始日, 終了日, モデル名, モデル) のタプルリスト
        self.models = {}  # モデル名 -> モデルのマッピング
    
    def add_model_period(self, 
                         model: Union[BaseModel, ModelContainer, BaseContainer], 
                         start_date: datetime,
                         end_date: Optional[datetime] = None,
                         model_name: Optional[str] = None):
        """
        特定の期間を担当するモデルを追加
        
        Args:
            model: 予測に使用するモデルまたはコンテナ
            start_date: この期間の開始日
            end_date: この期間の終了日（Noneの場合は無期限）
            model_name: モデルの名前（Noneの場合は自動生成）
        """
        if model_name is None:
            model_name = f"model_{len(self.models) + 1}"
            
        if model_name in self.models:
            # 既存のモデル名なら上書き
            self.models[model_name] = model
        else:
            # 新規モデル名の場合は追加
            self.models[model_name] = model
            
        # period_tupleを追加
        period_tuple = (start_date, end_date, model_name)
        self.model_periods.append(period_tuple)
        
        # 開始日でソート
        self.model_periods.sort(key=lambda x: x[0])
    
    def predict(self, X: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        各期間ごとのモデルを使って予測を行い、結果を結合
        
        Args:
            X: 特徴量データ。キー別のDataFrameの辞書またはマルチインデックスDataFrame
            
        Returns:
            pd.DataFrame: 全期間の予測結果を結合したDataFrame
        """
        if isinstance(X, dict):
            # 辞書型の入力は現在サポートしていない
            raise ValueError("辞書型の入力はサポートされていません。DataFrameを使用してください。")
            
        # DataFrameのインデックスから日付列を特定
        if not isinstance(X.index, pd.DatetimeIndex) and X.index.nlevels > 1:
            # マルチインデックスの場合、最初のレベルが日付と仮定
            date_index = X.index.get_level_values(0)
            if not isinstance(date_index, pd.DatetimeIndex):
                raise ValueError("DataFrameのインデックスの最初のレベルが日付ではありません。")
        else:
            # 単一インデックスの場合
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("DataFrameのインデックスが日付ではありません。")
            date_index = X.index
        
        # 期間ごとの予測結果を格納
        results = []
        
        for start_date, end_date, model_name in self.model_periods:
            # 日付範囲でフィルタリング
            if end_date is None:
                # 終了日が指定されていない場合は無期限
                mask = date_index >= start_date
            else:
                mask = (date_index >= start_date) & (date_index <= end_date)
                
            if not mask.any():
                # この期間に該当するデータがない場合はスキップ
                continue
                
            # この期間のデータを抽出
            X_period = X.loc[mask]
            
            # モデルの取得と予測
            model = self.models[model_name]
            
            if isinstance(model, (ModelContainer, BaseContainer)):
                # ModelContainerの場合
                pred_df = model.predict(X_period)
            else:
                # 単一モデルの場合
                preds = model.predict(X_period)
                pred_df = pd.DataFrame({'Pred': preds}, index=X_period.index)
            
            # 予測結果を結果リストに追加
            results.append(pred_df)
        
        # 結果がない場合は空のDataFrameを返す
        if not results:
            return pd.DataFrame(columns=['Pred'])
        
        # 全期間の予測結果を結合
        combined_results = pd.concat(results)
        
        # インデックスに基づいてソート
        return combined_results.sort_index()
    
    def get_active_model_for_date(self, date: datetime) -> Tuple[str, Union[BaseModel, BaseContainer]]:
        """
        指定された日付でアクティブなモデルを取得
        
        Args:
            date: 確認する日付
            
        Returns:
            Tuple[str, Union[BaseModel, BaseContainer]]: 
                モデル名とモデルのタプル
                
        Raises:
            ValueError: 指定された日付に対応するモデルがない場合
        """
        for start_date, end_date, model_name in self.model_periods:
            if start_date <= date and (end_date is None or date <= end_date):
                return model_name, self.models[model_name]
                
        raise ValueError(f"日付 {date} に対応するモデルがありません。")
    
    def get_period_summary(self) -> pd.DataFrame:
        """
        設定された期間とモデルの概要を返す
        
        Returns:
            pd.DataFrame: 期間とモデル名の概要
        """
        data = []
        for start_date, end_date, model_name in self.model_periods:
            end_str = str(end_date) if end_date else "無期限"
            data.append({
                '開始日': start_date,
                '終了日': end_str,
                'モデル名': model_name,
                'モデルタイプ': type(self.models[model_name]).__name__
            })
            
        return pd.DataFrame(data)
    
    def _get_state_dict(self) -> Dict[str, Any]:
        """コンテナの状態を辞書形式で取得する"""
        state = super()._get_state_dict()
        
        # 期間情報を保存
        state['model_periods'] = []
        for start_date, end_date, model_name in self.model_periods:
            state['model_periods'].append({
                'start_date': start_date,
                'end_date': end_date,
                'model_name': model_name
            })
        
        # モデル名のリストを保存
        state['model_names'] = list(self.models.keys())
        
        return state
    
    def _set_state_dict(self, state_dict: Dict[str, Any]):
        """辞書からコンテナの状態を復元する"""
        super()._set_state_dict(state_dict)
        
        # 期間情報を復元
        if 'model_periods' in state_dict:
            self.model_periods = []
            for period in state_dict['model_periods']:
                self.model_periods.append((
                    period['start_date'],
                    period['end_date'],
                    period['model_name']
                ))
            
            # 開始日でソート
            self.model_periods.sort(key=lambda x: x[0])
    
    def _export_models(self, models_dir: str):
        """コンテナ内のモデルをエクスポートする"""
        for model_name, model in self.models.items():
            # モデル固有のディレクトリを作成
            model_dir = os.path.join(models_dir, model_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # モデルの種類に応じたエクスポート
            if isinstance(model, BaseContainer):
                # 他のコンテナの場合はそのエクスポート機能を使用
                sub_dir = os.path.join(model_dir, "container")
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                model.export(sub_dir)
                
                # コンテナの種類を記録
                container_type_path = os.path.join(model_dir, "container_type.txt")
                with open(container_type_path, 'w') as f:
                    f.write(model.__class__.__name__)
            else:
                # 通常のモデルの場合は直接シリアライズ
                model_path = os.path.join(model_dir, "model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
    
    def _import_models(self, models_dir: str):
        """保存されたモデルをインポートする"""
        # ディレクトリ内の各モデルをロード
        for model_name in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, model_name)
            
            # コンテナの場合
            container_type_path = os.path.join(model_dir, "container_type.txt")
            if os.path.exists(container_type_path):
                with open(container_type_path, 'r') as f:
                    container_type = f.read().strip()
                
                # コンテナタイプに応じてロード
                container_dir = os.path.join(model_dir, "container")
                
                # コンテナタイプに基づいてインポート
                if container_type == "ModelContainer":
                    from models2.containers.model_container import ModelContainer
                    model = ModelContainer.load(container_dir)
                elif container_type == "EnsembleModel":
                    from models2.containers.ensemble import EnsembleModel
                    model = EnsembleModel.load(container_dir)
                elif container_type == "PeriodSwitchingModel":
                    model = PeriodSwitchingModel.load(container_dir)
                else:
                    raise ValueError(f"不明なコンテナタイプ: {container_type}")
                
                self.models[model_name] = model
                
            else:
                # 通常のモデルの場合
                model_path = os.path.join(model_dir, "model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        self.models[model_name] = model