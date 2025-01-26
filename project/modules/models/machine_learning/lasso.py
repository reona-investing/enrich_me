import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import scipy
from IPython.display import display
from models.machine_learning.dataclass import TrainerOutputs


class LassoTrainer:
    def __init__(self,
                 target_train_df: pd.DataFrame, features_train_df: pd.DataFrame):
        '''
        LASSOによる学習を管理します。
        Args:
            target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
            features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
        '''
        self.target_train_df = target_train_df
        self.features_train_df = features_train_df


    def train(self, max_features: int = 5, min_features: int = 3, **kwargs) -> TrainerOutputs:
        '''
        LASSOによる学習を行います。シングルセクターとマルチセクターの双方に対応しています。
        Args:
            max_features (int): 採用する特徴量の最大値
            min_features (int): 採用する特徴量の最小値
            **kwargs: LASSOのハイパーパラメータを任意で設定可能
        Returns:
            TrainerOutputs: モデルのリストとスケーラーのリストを設定したデータクラス
        '''
        if self.target_train_df.index.nlevels == 1:
            #シングルセクターの場合、単回学習とする。
            model, scaler = self._learn_single_sector(self.target_train_df, self.features_train_df, 
                                                            max_features, min_features, **kwargs)
            models = [model]
            scalers = [scaler]
        else:
            #マルチセクターの場合、セクターごとに学習する。
            models, scalers = self._learn_multi_sectors(self.target_train_df, self.features_train_df, 
                                                              max_features, min_features, **kwargs)
        return TrainerOutputs(models = models, scalers = scalers)


    def _learn_single_sector(self, y: pd.DataFrame, X: pd.DataFrame, max_features: int, min_features: int, **kwargs) -> Tuple[Lasso, StandardScaler]:
        '''
        LASSOで学習して，モデルとスケーラーを返す関数
        '''
        # 欠損値のある行を削除
        not_na_indices =X.dropna(how='any').index
        y = y.loc[not_na_indices, :]
        X = X.loc[not_na_indices, :]

        # 特徴量の標準化
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        #ランダムサーチで適切なアルファを探索
        alpha = self._search_alpha(X_scaled, y, max_features, min_features)

        #確定したモデルで学習
        model = Lasso(alpha=alpha, max_iter=100000, tol=0.00001, **kwargs)
        model.fit(X_scaled, y[['Target']])

        #特徴量重要度のデータフレームを返す
        feature_importances_df = self._get_feature_importances_df(model, feature_names=X.columns)
        print(alpha)
        display(feature_importances_df)

        return model, scaler


    def _learn_multi_sectors(self, target_train: pd.DataFrame, features_train: pd.DataFrame, 
                                   max_features: int, min_features: int, **kwargs) -> Tuple[list, list]:
        '''
        複数セクターに関して、LASSOで学習してモデルとスケーラーを返す関数
        '''
        models = []
        scalers = []
        sectors = target_train.index.get_level_values('Sector').unique()

        #セクターごとに学習する
        for sector in sectors:
            print(sector)
            y = target_train[target_train.index.get_level_values('Sector')==sector]
            X = features_train[features_train.index.get_level_values('Sector')==sector]
            model, scaler = self._learn_single_sector(y, X, max_features, min_features, **kwargs)
            models.append(model)
            scalers.append(scaler)

        return models, scalers


    def _search_alpha(self, X: np.array, y: pd.DataFrame, max_features: int, min_features: int) -> float:
        '''
        適切なalphaの値をサーチする。
        残る特徴量の数が、min_features以上、max_feartures以下となるように
        '''
        # alphaの探索範囲の初期値を事前指定しておく
        min_alpha = 0.000005
        max_alpha = 0.005
        is_searching = True
        while is_searching:
            # ランダムサーチの準備
            model = Lasso(max_iter=100000, tol=0.00001)
            param_grid = {'alpha': scipy.stats.uniform(min_alpha, max_alpha - min_alpha)}
            random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=3, cv=5, random_state=42)

            # ランダムサーチを実行
            random_search.fit(X, y)

            # 最適なalphaを取得
            alpha = random_search.best_params_['alpha']

            # Lassoモデルを作成し、特徴量の数を確認
            model = Lasso(alpha=alpha, max_iter=100000, tol=0.00001)
            model.fit(X, y[['Target']])
            num_features = len(model.coef_[model.coef_ != 0])

            # 特徴量の数が範囲内に収まるか判定
            if num_features < min_features and max_alpha > alpha:
                max_alpha = alpha
            elif num_features > max_features and min_alpha < alpha:
                min_alpha = alpha
            else:
                is_searching = False

        return alpha


    def _get_feature_importances_df(self, model:Lasso, feature_names:pd.core.indexes.base.Index) -> pd.DataFrame:
        '''
        feature importancesをdf化して返す
        '''
        feature_importances_df = pd.DataFrame(model.coef_, index=feature_names, columns=['FI'])
        feature_importances_df = feature_importances_df[feature_importances_df['FI']!=0]
        feature_importances_df['abs'] = abs(feature_importances_df['FI'])
        return feature_importances_df.sort_values(by='abs', ascending=False)[['FI']]



class LassoPredictor:
    def __init__(self,
                 target_test_df: pd.DataFrame, features_test_df: pd.DataFrame,
                 models: list[Lasso] = None, scalers: list[StandardScaler] = None,
                 ):
        '''
        LASSOによる学習と予測を管理します。
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (list[Lasso]): LASSOモデルを格納したリスト
            scalers (list[SrandaedScaler]): LASSOモデルに対応したスケーラーを格納したリスト
        '''
        self.target_test_df = target_test_df
        self.features_test_df = features_test_df
        self.models = models
        self.scalers = scalers

    def predict(self) -> pd.DataFrame:
        '''
        LASSOによる予測を行います。シングルセクターとマルチセクターの双方に対応しています。
        Returns:
            pd.DataFrame: 予測結果を格納したデータフレーム
        '''
        assert len(self.models) == len(self.scalers), 'モデルとスケーラーには同じ数を設定してください。'
        assert len(self.models) > 0, '予測のためには1つ以上のモデルが必要です。' 
        if self.target_test_df.index.nlevels == 1:
            # シングルセクターの場合、単回で予測する。
            pred_result_df = self._pred_single_sector(self.target_test_df, self.features_test_df, self.models[0], self.scalers[0])
        else:
            #マルチセクターの場合、セクターごとに予測する。
            pred_result_df = self._pred_multi_sectors(self.target_test_df, self.features_test_df, self.models, self.scalers)
        return pred_result_df


    def _pred_single_sector(self, y_test: pd.DataFrame, X_test: pd.DataFrame, model:Lasso, scaler:StandardScaler) -> pd.DataFrame:
        '''
        LASSOモデルで予測して予測結果を返す関数
        '''
        y_test = y_test.loc[X_test.dropna(how='any').index, :]
        X_test = X_test.loc[X_test.dropna(how='any').index, :]
        X_test = scaler.transform(X_test) #標準化
        y_test['Pred'] = model.predict(X_test) #学習

        return y_test


    def _pred_multi_sectors(self, target_test: pd.DataFrame, features_test: pd.DataFrame, 
                            models:list[Lasso], scalers:list[StandardScaler]) -> pd.DataFrame:
        '''
        複数セクターに関して、LASSOモデルで予測して予測結果を返す関数
        '''
        y_tests = []
        sectors = target_test.index.get_level_values('Sector').unique()
        #セクターごとに予測する
        for i, sector in enumerate(sectors):
            y_test = target_test[target_test.index.get_level_values('Sector')==sector]
            X_test = features_test[features_test.index.get_level_values('Sector')==sector]
            y_test = self._pred_single_sector(y_test, X_test, models[i], scalers[i])
            y_tests.append(y_test)

        pred_result_df = pd.concat(y_tests, axis=0).sort_index()

        return pred_result_df



class LassoModel:
    def __init__(self):
        '''
        LASSOによる学習と予測を管理します。
        '''
        pass
    
    def train(self, target_train_df: pd.DataFrame, features_train_df: pd.DataFrame, max_features: int = 5, min_features: int = 3, **kwargs) -> TrainerOutputs:
        '''
        LASSOによる学習を管理します。
        Args:
            target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
            features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
            max_features (int): 採用する特徴量の最大値
            min_features (int): 採用する特徴量の最小値
            **kwargs: LASSOのハイパーパラメータを任意で設定可能
        Returns:
            TrainerOutputs: モデルのリストとスケーラーのリストを格納したデータクラス
        '''
        trainer = LassoTrainer(target_train_df, features_train_df)
        trainer_outputs = trainer.train(max_features, min_features, **kwargs)
        return trainer_outputs
    
    def predict(self, target_test_df: pd.DataFrame, features_test_df: pd.DataFrame, models: list[Lasso], scalers: list[StandardScaler]) -> pd.DataFrame:
        '''
        LASSOによる学習と予測を管理します。
        Args:
            target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
            features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
            models (list[object]): LASSOモデルを格納したリスト
            scalers (list[object]): LASSOモデルに対応したスケーラーを格納したリスト
        Returns:
            pd.DataFrame: 予測結果のデータフレーム
        '''
        predictor = LassoPredictor(target_test_df, features_test_df, models, scalers)
        pred_result_df = predictor.predict()
        return pred_result_df