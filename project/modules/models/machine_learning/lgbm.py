import pandas as pd
import lightgbm as lgb
from models.machine_learning.dataclass import TrainerOutputs

# TODO Lassoとlgbmでファイルを分ける。
class LgbmTrainer:
    def __init__(self, target_train_df: pd.DataFrame, features_train_df: pd.DataFrame):
        '''
        lightGBMによる学習を管理します。
        Args:
            target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
            features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
        '''
        self.target_train_df = target_train_df
        self.features_train_df = features_train_df
    
    def train(self, categorical_features: list[str] = None, **kwargs) -> list[lgb.train]:
        '''
        lightGBMによる学習を行います。シングルセクターとマルチセクターの双方に対応しています。
        Args:
            categorical_featurse (list[str]): カテゴリ変数の一覧
            **kwargs: lightgbmのハイパーパラメータを任意で設定可能
        Returns:
            TrainerOutputs: モデルのリストとスケーラーのリストを設定したデータクラス
        '''
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
            'random_seed':42,
            'lambda_l1':0.5,
        }
        params.update(kwargs)

        models = [lgb.train(params, train_data, feval=self._numerai_corr_lgbm, num_boost_round=100000)]
        return TrainerOutputs(models = models)

    def _numerai_corr_lgbm(self, preds, data):
        import numpy as np
        import pandas as pd
        from scipy.stats import norm

        # データセットからターゲットを取得
        target = data.get_label()

        # predsとtargetをDataFrameに変換
        df = pd.DataFrame({'Pred': preds, 'Target': target, 'Date': data.get_field('date')})

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

        numerai_corr = df.groupby('Date').apply(lambda x: _get_daily_numerai_corr(x['Target_rank'], x['Pred_rank'])).mean()

        # LightGBMのカスタムメトリックの形式で返す
        return 'numerai_corr', numerai_corr, True


class LgbmPredictor:
    def __init__(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, models: list[lgb.train]):
        '''
        lightGBMによる学習と予測を管理します。
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (list[train]): lightGBMモデルを格納したリスト
        '''
        self.target_test_df = target_test_df
        self.features_test_df = features_test_df
        self.models = models

    def predict(self) -> pd.DataFrame:
        '''
        lightGBMによる予測を行います。シングルセクターとマルチセクターの双方に対応しています。
        Returns:
            pd.DataFrame: 予測結果を格納したデータフレーム
        '''
        X_test = self.features_test_df

        pred_result_df = self.target_test_df.copy()
        pred_result_df['Pred'] = self.models[0].predict(X_test, num_iteration=self.models[0].best_iteration)
        return pred_result_df
    

class LgbmModel:
    def __init__(self):
        '''
        lightGBMによる学習と予測を管理します。
        '''
        pass
    
    def train(self, 
              target_train_df: pd.DataFrame, features_train_df: pd.DataFrame, 
              categorical_features: list[str] = None, **kwargs) -> TrainerOutputs:
        '''
        lightGBMによる学習を管理します。
        Args:
            target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
            features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
            categorical_features (list[str]): カテゴリ変数として使用する特徴量
            **kwargs: lightGBMのハイパーパラメータを任意で設定可能
        Returns:
            TrainerOutputs: モデルのリストとスケーラーのリストを格納したデータクラス
        '''
        trainer = LgbmTrainer(target_train_df, features_train_df)
        trainer_outputs = trainer.train(categorical_features, **kwargs)
        return trainer_outputs
    
    def predict(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, models: list[lgb.train]) -> pd.DataFrame:
        '''
        lightGBMによる学習と予測を管理します。
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (list[object]): LASSOモデルを格納したリスト
        Returns:
            pd.DataFrame: 予測結果のデータフレーム
        '''
        predictor = LgbmPredictor(target_test_df, features_test_df, models)
        pred_result_df = predictor.predict()
        return pred_result_df