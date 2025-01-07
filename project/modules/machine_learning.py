#%% モジュールのインポート
from models import MLDataset

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
import scipy
from IPython.display import display

#%% 関数群
def _lasso_learn_single_sector(y: pd.DataFrame, X: pd.DataFrame, max_features: int, min_features: int, **kwargs) -> Tuple[Lasso, StandardScaler]:
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
    alpha = _search_lasso_alpha(X_scaled, y, max_features, min_features)

    #確定したモデルで学習
    model = Lasso(alpha=alpha, max_iter=100000, tol=0.00001, **kwargs)
    model.fit(X_scaled, y[['Target']])

    #特徴量重要度のデータフレームを返す
    feature_importances_df = _get_feature_importances_df(model, feature_names=X.columns)
    print(alpha)
    display(feature_importances_df)

    return model, scaler

def _lasso_learn_multi_sectors(target_train: pd.DataFrame, features_train: pd.DataFrame, max_features: int, min_features: int, **kwargs) -> Tuple[list, list]:
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
        model, scaler = _lasso_learn_single_sector(y, X, max_features, min_features, **kwargs)
        models.append(model)
        scalers.append(scaler)

    return models, scalers


def _lasso_pred_single_sector(y_test: pd.DataFrame, X_test: pd.DataFrame, model:Lasso, scaler:StandardScaler) -> pd.DataFrame:
    '''
    LASSOモデルで予測して予測結果を返す関数
    '''
    y_test = y_test.loc[X_test.dropna(how='any').index, :]
    X_test = X_test.loc[X_test.dropna(how='any').index, :]
    X_test = scaler.transform(X_test) #標準化
    y_test['Pred'] = model.predict(X_test) #学習

    return y_test


def _lasso_pred_multi_sectors(target_test: pd.DataFrame, features_test: pd.DataFrame, models:list, scalers:list) -> pd.DataFrame:
    '''
    複数セクターに関して、LASSOモデルで予測して予測結果を返す関数
    '''
    y_tests = []
    sectors = target_test.index.get_level_values('Sector').unique()

    #セクターごとに予測する
    for i, sector in enumerate(sectors):
        y_test = target_test[target_test.index.get_level_values('Sector')==sector]
        X_test = features_test[features_test.index.get_level_values('Sector')==sector]
        y_test = _lasso_pred_single_sector(y_test, X_test, models[i], scalers[i])
        y_tests.append(y_test)

    pred_result_df = pd.concat(y_tests, axis=0).sort_index()

    return pred_result_df

def _search_lasso_alpha(X: np.array, y: pd.DataFrame, max_features: int, min_features: int) -> float:
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


def _get_feature_importances_df(model:object, feature_names:pd.core.indexes.base.Index) -> pd.DataFrame:
    '''
    feature importancesをdf化して返す
    '''
    feature_importances_df = pd.DataFrame(model.coef_, index=feature_names, columns=['FI'])
    feature_importances_df = feature_importances_df[feature_importances_df['FI']!=0]
    feature_importances_df['abs'] = abs(feature_importances_df['FI'])
    feature_importances_df = feature_importances_df.sort_values(by='abs', ascending=False)[['FI']]

    return feature_importances_df

def _numerai_corr_lgbm(preds, data):
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

def lasso(ml_dataset: MLDataset, dataset_path:str, learn: bool = True, max_features: int = 5, min_features: int = 3, **kwargs) -> MLDataset:
    '''
    ml_dataset: 機械学習用のデータセット
    dataset_path: データセットのファイルパス
    learn: Trueなら学習を実施，Falseなら既存モデルを用いて予測のみ
    **kwargs: 学習時，ハイパーパラメータをデフォルト値から変更する場合のみ使用
    '''

    if learn == False and (ml_dataset.ml_models is None or ml_dataset.ml_scalers is None):
        # モデルがない場合，強制的に学習を実施
        print('learn=Falseが設定されましたが，モデルがありません．学習を実施します．')
        learn = True

    if learn:
        # learn=Trueのときのみ学習
        if ml_dataset.target_train_df.index.nlevels == 1:
            #シングルセクターの場合、単回学習とする。
            model, scaler = _lasso_learn_single_sector(ml_dataset.target_train_df, ml_dataset.features_train_df, max_features, min_features, **kwargs)
            ml_dataset.archive_ml_objects(ml_models=[model], ml_scalers=[scaler])
        else:
            #マルチセクターの場合、セクターごとに学習する。
            ml_dataset.ml_models, ml_dataset.ml_scalers = _lasso_learn_multi_sectors(ml_dataset.target_train_df, ml_dataset.features_train_df, max_features, min_features, **kwargs)

    if ml_dataset.target_train_df.index.nlevels == 1:
        # シングルセクターの場合、単回で予測する。
        ml_dataset.pred_result_df = _lasso_pred_single_sector(ml_dataset.target_test_df, ml_dataset.features_test_df, ml_dataset.ml_models[0], ml_dataset.ml_scalers[0])
    else:
        #マルチセクターの場合、セクターごとに予測する。
        ml_dataset.pred_result_df = _lasso_pred_multi_sectors(ml_dataset.target_test_df, ml_dataset.features_test_df, ml_dataset.ml_models, ml_dataset.ml_scalers)

    #データセットの保存と復元
    if dataset_path is None:
        print('データセットの出力パスが指定されていないため、出力しません。')
    else:
        ml_dataset.save_instance(dataset_path)
        ml_dataset = MLDataset(dataset_path)

    return ml_dataset


def lgbm(ml_dataset: MLDataset, dataset_path: str, learn: bool = True, categorical_features: list = None, **kwargs):
    # データの準備
    X_train = ml_dataset.features_train_df
    y_train = ml_dataset.target_train_df['Target']
    X_test = ml_dataset.features_test_df
    y_test = ml_dataset.target_test_df['Target']

    if learn:
        # LightGBM用のデータセットを作成
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, categorical_feature=categorical_features)

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

        # コールバックの設定
        callbacks = [lgb.early_stopping(stopping_rounds=100)]
        # モデルのトレーニング
        ml_dataset.ml_models = ml_dataset.ml_scalers = []
        ml_dataset.ml_models.append(lgb.train(params, train_data, feval=_numerai_corr_lgbm, num_boost_round=100000))

    # 予測
    ml_dataset.pred_result_df = ml_dataset.target_test_df.copy()
    ml_dataset.pred_result_df['Pred'] = ml_dataset.ml_models[0].predict(X_test, num_iteration=ml_dataset.ml_models[0].best_iteration)

    #データセットの保存と復元
    if dataset_path is None:
        print('データセットの出力パスが指定されていないため、出力しません。')
    else:
        ml_dataset.save_instance(dataset_path)
        ml_dataset = MLDataset(dataset_path)
    
    return ml_dataset

def ensemble_by_rank(ml_datasets: list, ensemble_rates: list) -> pd.Series:
    '''
    2つ以上のモデルの結果をアンサンブルする（予測順位ベース）
    ml_datasets: アンサンブルしたいモデルのMLDatasetをリストに格納
    ensemble_rates: 各モデルの予測結果を合成する際の重みづけ
    '''
    assert len(ml_datasets) == len(ensemble_rates), "ml_datasetsとensemble_ratesには同じ個数のデータをセットしてください。"
    for i in range(len(ml_datasets)):
        if i == 0:
            ensembled_rank = ml_datasets[i].pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * ensemble_rates[i]
        else:
            ensembled_rank += ml_datasets[i].pred_result_df.groupby('Date')['Pred'].rank(ascending=False) * ensemble_rates[i]
    
    ensembled_rank = pd.DataFrame(ensembled_rank, index=ml_datasets[len(ml_datasets) - 1].pred_result_df.index, columns=['Pred'])

    return ensembled_rank.groupby('Date')[['Pred']].rank(ascending=False)
    
