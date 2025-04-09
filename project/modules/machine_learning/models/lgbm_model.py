"""
単一のLightGBMモデルを管理するクラス
"""
from typing import Optional, List, Union, Dict, Any
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import norm

from models2.base.base_model import BaseModel
from models2.base.params import LgbmParams


class LgbmModel(BaseModel):
    """
    単一のLightGBMモデルを管理するクラス
    """
    
    def __init__(self):
        """初期化"""
        self._model = None
        self._scaler = None
        self._prediction = None
        self._feature_names = None
        self._feature_importance_df = None
    
    def train(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame], params: LgbmParams | None = None, **kwargs):
        """
        LightGBMモデルを学習する
        
        Args:
            X: 特徴量DataFrame
            y: 目的変数Series
            params: LgbmParams (ハイパーパラメータをまとめたデータクラス)
            **kwargs: パラメータ辞書（paramsが指定されていない場合に使用）
        """
        # パラメータの取得
        if params:
            # LgbmParamsが提供されている場合はそれを使用
            model_params = params.get_model_params()
            num_boost_round = params.num_boost_round
            early_stopping_rounds = params.early_stopping_rounds
            categorical_features = params.categorical_features
        else:
            # kwargsから直接パラメータを取得
            num_boost_round = kwargs.pop("num_boost_round", 100000)
            early_stopping_rounds = kwargs.pop("early_stopping_rounds", None)
            categorical_features = kwargs.pop("categorical_features", None)
            
            # LightGBMのモデルパラメータとして使用
            model_params = kwargs
        
        # 特徴量名を保存
        self._feature_names = X.columns.tolist()
        
        # DataFrameやSeries をnumpyに変換
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y_values = y.iloc[:, 0]
        else:
            y_values = y
        
        # データセット作成
        train_data = lgb.Dataset(
            X, 
            label=y_values, 
            categorical_feature=categorical_features,
            feature_name=X.columns.tolist(),
            free_raw_data=False
        )
        
        # カスタム評価関数の設定
        feval = None
        if "numerai_corr" in model_params.get("metric", ""):
            feval = self._numerai_corr_lgbm
        
        # 学習実行
        self._model = lgb.train(
            model_params, 
            train_data, 
            num_boost_round=num_boost_round,
            valid_sets=[train_data] if early_stopping_rounds else None,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                lgb.log_evaluation(100),
            ] if early_stopping_rounds else None,
            feval=feval
        )
        
        # 特徴量重要度の計算
        self._calculate_feature_importances()
        
        return self
    
    def _numerai_corr_lgbm(self, preds, data):
        """
        Numerai相関係数を計算するカスタム評価関数
        この関数はLightGBMの評価関数の形式に従う
        
        Args:
            preds: 予測値
            data: Dataset
            
        Returns:
            tuple: (評価指標名, 評価値, is_higher_better)
        """
        import numpy as np
        import pandas as pd
        from scipy.stats import norm
        
        # データセットからターゲットを取得
        target = data.get_label()
        # 日付情報を取得（あれば）
        date = None
        if hasattr(data, 'get_field') and callable(getattr(data, 'get_field')):
            try:
                date = data.get_field('date')
            except:
                # 日付フィールドがない場合は無視
                pass
        
        # predsとtargetをDataFrameに変換
        if date is not None:
            df = pd.DataFrame({'Pred': preds, 'Target': target, 'Date': date})
            # Target_rankとPred_rankを計算
            df['Target_rank'] = df.groupby('Date')['Target'].rank(ascending=False)
            df['Pred_rank'] = df.groupby('Date')['Pred'].rank(ascending=False)
        else:
            df = pd.DataFrame({'Pred': preds, 'Target': target})
            # 日付がない場合は全体でランク付け
            df['Target_rank'] = df['Target'].rank(ascending=False)
            df['Pred_rank'] = df['Pred'].rank(ascending=False)
        
        # 日次のnumerai_corrを計算する関数
        def _get_daily_numerai_corr(target_rank, pred_rank):
            pred_rank = np.array(pred_rank)
            scaled_pred_rank = (pred_rank - 0.5) / len(pred_rank)
            gauss_pred_rank = norm.ppf(scaled_pred_rank)
            pred_pow = np.sign(gauss_pred_rank) * np.abs(gauss_pred_rank) ** 1.5
            
            target = np.array(target_rank)
            centered_target = target - target.mean()
            target_pow = np.sign(centered_target) * np.abs(centered_target) ** 1.5
            
            return np.corrcoef(pred_pow, target_pow)[0, 1]
        
        # 日付ごとに相関を計算し平均を取る
        if date is not None:
            numerai_corr = df.groupby('Date').apply(
                lambda x: _get_daily_numerai_corr(x['Target_rank'], x['Pred_rank'])
            ).mean()
        else:
            # 日付がない場合は全体で一度だけ計算
            numerai_corr = _get_daily_numerai_corr(df['Target_rank'], df['Pred_rank'])
        
        # LightGBMのカスタムメトリックの形式で返す
        return 'numerai_corr', numerai_corr, True
    
    def _calculate_feature_importances(self):
        """特徴量重要度を計算してDataFrameに格納"""
        if self._model is None or self._feature_names is None:
            return
        
        # LightGBMから特徴量重要度を取得
        importance_type = 'gain'  # 'gain'、'split'、'weight' のいずれか
        importance = self._model.feature_importance(importance_type=importance_type)
        
        # DataFrameに変換
        self._feature_importance_df = pd.DataFrame({
            'Feature': self._feature_names,
            'Importance': importance
        })
        
        # 重要度降順でソート
        self._feature_importance_df = self._feature_importance_df.sort_values(
            'Importance', ascending=False
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        学習済みモデルを使用して予測を行う
        
        Args:
            X: 特徴量DataFrame
        
        Returns:
            np.ndarray: 予測値の配列
        """
        if self._model is None:
            raise ValueError("モデルが学習されていません。先にtrainメソッドを呼び出してください。")
        
        # 欠損値のある行を特定
        not_na_indices = X.dropna(how='any').index
        
        # 完全な予測結果配列を初期化
        full_predictions = np.full(X.shape[0], np.nan)
        
        # 欠損値のない行のみで予測を実行
        if len(not_na_indices) > 0:
            X_filtered = X.loc[not_na_indices, :]
            predictions = self._model.predict(X_filtered, num_iteration=self._model.best_iteration)
            full_predictions[X.index.get_indexer(not_na_indices)] = predictions
        
        self._prediction = full_predictions
        return full_predictions
    
    @property
    def model(self):
        """内部モデルを取得"""
        return self._model
    
    @property
    def scaler(self):
        """スケーラーを取得"""
        return self._scaler

    @property
    def prediction(self):
        """予測結果を取得する"""
        return self._prediction

    @property
    def feature_importance(self) -> pd.DataFrame:
        """特徴量重要度を取得する"""
        return self._feature_importance_df

    @property
    def metadata(self) -> Dict[str, Any]:
        """メタデータを取得する"""
        return self._metadata