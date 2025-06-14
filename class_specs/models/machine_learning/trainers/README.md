# models/machine_learning/trainers のクラス仕様書

## base_trainer.py

### class BaseTrainer
機械学習モデルのトレーナーの抽象基底クラス
- __init__: Args:
    target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
    features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
- train: モデルの学習を行う抽象メソッド

Args:
    **kwargs: モデル固有のハイパーパラメータ
    
Returns:
    TrainerOutputs: 学習済みモデルとスケーラーを格納したデータクラス
- _is_multi_sector: マルチセクターかどうかを判定
- _get_sectors: セクター一覧を取得

## lasso_trainer.py

### class LassoTrainer
LASSOモデルのトレーナークラス
- train: LASSOによる学習を行います。シングルセクターとマルチセクターの双方に対応しています。

Args:
    max_features (int): 採用する特徴量の最大値
    min_features (int): 採用する特徴量の最小値
    **kwargs: LASSOのハイパーパラメータを任意で設定可能
    
Returns:
    TrainerOutputs: モデルとスケーラーを設定したデータクラス
- _train_single_sector: LASSOで学習して，モデルとスケーラーを返す関数
- _search_alpha: 適切なalphaの値をサーチする。
残る特徴量の数が、min_features以上、max_feartures以下となるように
- _get_feature_importances_df: feature importancesをdf化して返す

## lgbm_trainer.py

### class LgbmTrainer
LightGBMモデルのトレーナークラス
- train: lightGBMによる学習を行います。シングルセクターとマルチセクターの双方に対応しています。

Args:
    categorical_features (Optional[List[str]]): カテゴリ変数の一覧
    **kwargs: lightgbmのハイパーパラメータを任意で設定可能
    
Returns:
    TrainerOutputs: モデルを設定したデータクラス
- _numerai_corr_lgbm: LightGBM用のカスタム評価関数（numerai correlation）

