import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import lightgbm as lgb
from scipy.stats import norm
from machine_learning.params import LgbmParams
from machine_learning.models import BaseModel


class LgbmModel(BaseModel):
    """LightGBM回帰モデル"""
    
    def __init__(self, name: str, params: Optional[LgbmParams] = None):
        """
        Args:
            name: モデル名
            params: モデルパラメータ。指定しない場合はデフォルトパラメータが使用される。
        """
        super().__init__(name, params or LgbmParams())
        self.model = None
        self.feature_importances = None
    
    def train(self) -> None:
        """モデルを学習する"""
        if self.target_train_df is None or self.features_train_df is None:
            raise ValueError("訓練データがセットされていません。load_dataset()を先に実行してください。")
        
        # 学習データの取得
        X_train = self.features_train_df
        y_train = self.target_train_df['Target']
        
        # パラメータの設定
        params_dict = self.params.to_dict()
        
        # カテゴリカル特徴量の取得
        categorical_features = params_dict.pop('categorical_features', None)
        num_boost_round = params_dict.pop('num_boost_round', 100000)
        early_stopping_rounds = params_dict.pop('early_stopping_rounds', None)
        
        # LightGBMのデータセット作成
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            categorical_feature=categorical_features,
            feature_name=list(X_train.columns)
        )
        
        # 学習の実行
        self.model = lgb.train(
            params_dict,
            train_data,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            feval=self._numerai_corr_lgbm
        )
        
        # 特徴量重要度の計算
        self.feature_importances = self._get_feature_importances()
        
        print(f"Model {self.name} trained with {self.model.best_iteration} iterations")
    
    def predict(self) -> pd.DataFrame:
        """予測を実行する"""
        if self.model is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        
        if self.target_test_df is None or self.features_test_df is None:
            raise ValueError("テストデータがセットされていません。load_dataset()を先に実行してください。")
        
        # テストデータの取得
        X_test = self.features_test_df
        
        # 予測の実行
        pred_result_df = self.target_test_df.copy()
        pred_result_df['Pred'] = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        
        # 予測結果を保存
        self.pred_result_df = pred_result_df
        
        return self.pred_result_df
    
    def _numerai_corr_lgbm(self, preds, data):
        """
        Numerai相関係数を計算するカスタム評価関数
        """
        # データセットからターゲットを取得
        target = data.get_label()
        
        # Numerai相関係数の計算
        target_array = np.array(target)
        centered_target = target_array - target_array.mean()
        target_pow = np.sign(centered_target) * np.abs(centered_target) ** 1.5
        
        pred_array = np.array(preds)
        scaled_pred = (pred_array - pred_array.min()) / (pred_array.max() - pred_array.min())
        scaled_pred = scaled_pred * 0.98 + 0.01  # [0.01, 0.99]の範囲に収める
        gauss_pred = norm.ppf(scaled_pred)
        pred_pow = np.sign(gauss_pred) * np.abs(gauss_pred) ** 1.5
        
        # 相関係数の計算
        numerai_corr = np.corrcoef(pred_pow, target_pow)[0, 1]
        
        # LightGBMのカスタムメトリックの形式で返す
        return 'numerai_corr', numerai_corr, True
    
    def _get_feature_importances(self) -> pd.DataFrame:
        """
        モデルの特徴量重要度を取得する
        
        Returns:
            特徴量とその重要度を格納したデータフレーム
        """
        importance_type = 'gain'
        feature_names = self.model.feature_name()
        
        # 特徴量重要度の取得
        importance = self.model.feature_importance(importance_type=importance_type)
        
        # データフレームの作成
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # 重要度でソート
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def get_feature_importances(self) -> pd.DataFrame:
        """学習済みモデルの特徴量重要度を取得する"""
        if self.feature_importances is None:
            raise ValueError("モデルが学習されていません。train()を先に実行してください。")
        return self.feature_importances