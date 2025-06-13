import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Optional
from scipy.stats import norm
from models.machine_learning.outputs import TrainerOutputs
from models.machine_learning.trainers.base_trainer import BaseTrainer


class LgbmTrainer(BaseTrainer):
    """LightGBMモデルのトレーナークラス"""
    
    def train(self, categorical_features: Optional[List[str]] = None, **kwargs) -> TrainerOutputs:
        """
        lightGBMによる学習を行います。シングルセクターとマルチセクターの双方に対応しています。
        
        Args:
            categorical_features (Optional[List[str]]): カテゴリ変数の一覧
            **kwargs: lightgbmのハイパーパラメータを任意で設定可能
            
        Returns:
            TrainerOutputs: モデルを設定したデータクラス
        """
        X_train = self.features_train_df
        y_train = self.target_train_df['Target']
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)

        # ハイパーパラメータの設定
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.001,
            'num_leaves': 7,
            'verbose': -1,
            'random_seed': 42,
            'lambda_l1': 0.5,
        }
        params.update(kwargs)

        model = lgb.train(
            params, 
            train_data, 
            feval=self._numerai_corr_lgbm, 
            num_boost_round=100000
        )
        
        return TrainerOutputs(model=model)

    def _numerai_corr_lgbm(self, preds, data):
        """
        LightGBM用のカスタム評価関数（numerai correlation）
        """
        # データセットからターゲットを取得
        target = data.get_label()

        # predsとtargetをDataFrameに変換
        df = pd.DataFrame({
            'Pred': preds, 
            'Target': target, 
            'Date': data.get_field('date')
        })

        # Target_rankとPred_rankを計算
        df['Target_rank'] = df.groupby('Date')['Target'].rank(ascending=False)
        df['Pred_rank'] = df.groupby('Date')['Pred'].rank(ascending=False)

        # 日次のnumerai_corrを計算
        def _get_daily_numerai_corr(target_rank, pred_rank):
            pred_rank = np.array(pred_rank)
            scaled_pred_rank = (pred_rank - 0.5) / len(pred_rank)
            gauss_pred_rank = norm.ppf(scaled_pred_rank)
            pred_pow = np.sign(gauss_pred_rank) * np.abs(gauss_pred_rank) ** 1.5

            target = np.array(target_rank)
            centered_target = target - target.mean()
            target_pow = np.sign(centered_target) * np.abs(centered_target) ** 1.5

            return np.corrcoef(pred_pow, target_pow)[0, 1]

        numerai_corr = df.groupby('Date').apply(
            lambda x: _get_daily_numerai_corr(x['Target_rank'], x['Pred_rank'])
        ).mean()

        # LightGBMのカスタムメトリックの形式で返す
        return 'numerai_corr', numerai_corr, True