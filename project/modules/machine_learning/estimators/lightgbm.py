import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Optional, List, Dict, Any
from scipy.stats import norm

from machine_learning.core.model_base import ModelBase


class LightGBMModel(ModelBase):
    """LightGBMモデルの実装"""
    
    def __init__(self, 
                name: str, 
                objective: str = 'regression',
                metric: str = 'rmse',
                boosting_type: str = 'gbdt',
                learning_rate: float = 0.001,
                num_leaves: int = 7,
                num_boost_round: int = 100000,
                lambda_l1: float = 0.5,
                random_seed: int = 42,
                categorical_features: List[str] = None,
                **kwargs):
        """
        Args:
            name: モデル名
            objective: 目的関数
            metric: 評価指標
            boosting_type: ブースティング方法
            learning_rate: 学習率
            num_leaves: 葉の数
            num_boost_round: ブースト回数
            lambda_l1: L1正則化の強さ
            random_seed: 乱数シード
            categorical_features: カテゴリ特徴量のリスト
            **kwargs: その他のLightGBMパラメータ
        """
        super().__init__(name)
        self.objective = objective
        self.metric = metric
        self.boosting_type = boosting_type
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.num_boost_round = num_boost_round
        self.lambda_l1 = lambda_l1
        self.random_seed = random_seed
        self.categorical_features = categorical_features or []
        self.extra_params = kwargs
    
    def train(self) -> None:
        """LightGBMモデルを学習する"""
        if self.target_train_df is None or self.features_train_df is None:
            raise ValueError("訓練データがセットされていません。load_dataset()を先に実行してください。")
        
        # 学習データの取得
        X_train = self.features_train_df
        y_train = self.target_train_df['Target']
        
        # LightGBMのパラメータ設定
        params = {
            'objective': self.objective,
            'metric': self.metric,
            'boosting_type': self.boosting_type,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'lambda_l1': self.lambda_l1,
            'verbose': -1,
            'random_seed': self.random_seed
        }
        params.update(self.extra_params)
        
        # LightGBMデータセットの作成
        train_data = lgb.Dataset(
            X_train, 
            label=y_train, 
            categorical_feature=self.categorical_features,
            feature_name=list(X_train.columns)
        )
        
        # 学習の実行
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.num_boost_round,
            feval=self._numerai_corr_lgbm
        )
        
        # 特徴量重要度の計算
        self.feature_importances = self._get_feature_importances()
        self.trained = True
        
        print(f"Model {self.name} trained with {self.model.best_iteration} iterations")
    
    def predict(self) -> pd.DataFrame:
        """学習済みモデルで予測を実行する"""
        if not self.trained or self.model is None:
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
        
        Args:
            preds: モデルによる予測値
            data: LightGBMのデータセット
            
        Returns:
            (評価指標名, 評価値, 大きい方が良いかの真偽値)のタプル
        """
        # データセットからターゲットを取得
        target = data.get_label()
        
        # Numerai相関係数の計算
        target_array = np.array(target)
        centered_target = target_array - target_array.mean()
        target_pow = np.sign(centered_target) * np.abs(centered_target) ** 1.5
        
        pred_array = np.array(preds)
        scaled_pred = (pred_array - pred_array.min()) / (pred_array.max() - pred_array.min() + 1e-8)
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
        if self.model is None:
            raise ValueError("モデルが学習されていません。")
            
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