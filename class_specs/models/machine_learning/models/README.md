# models/machine_learning/models のクラス仕様書

## base_model.py

### class BaseModel
機械学習モデルの抽象基底クラス
- train: 学習を実行する抽象メソッド

Args:
    target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
    features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
    **kwargs: モデル固有のハイパーパラメータ
    
Returns:
    TrainerOutputs: 学習済みモデルとスケーラーを格納したデータクラス
- predict: 予測を実行する抽象メソッド

Args:
    target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
    features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
    model (Any): 学習済みモデル
    scaler (Optional[Any]): スケーラー（必要な場合）
    
Returns:
    pd.DataFrame: 予測結果のデータフレーム

## lasso_model.py

### class LassoModel
LASSOモデルのファサードクラス
- train: LASSOによる学習を管理します。

Args:
    target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
    features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
    max_features (int): 採用する特徴量の最大値
    min_features (int): 採用する特徴量の最小値
    **kwargs: LASSOのハイパーパラメータを任意で設定可能
    
Returns:
    TrainerOutputs: モデルとスケーラーを格納したデータクラス
- predict: LASSOによる予測を管理します。

Args:
    target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
    features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
    model (Lasso): LASSOモデル
    scaler (StandardScaler): LASSOモデルに対応したスケーラー
    
Returns:
    pd.DataFrame: 予測結果のデータフレーム

## lgbm_model.py

### class LgbmModel
LightGBMモデルのファサードクラス
- train: lightGBMによる学習を管理します。

Args:
    target_train_df (pd.DataFrame): 訓練用の目的変数データフレーム
    features_train_df (pd.DataFrame): 訓練用の特徴量データフレーム
    categorical_features (Optional[List[str]]): カテゴリ変数として使用する特徴量
    **kwargs: lightGBMのハイパーパラメータを任意で設定可能
    
Returns:
    TrainerOutputs: モデルとスケーラーを格納したデータクラス
- predict: lightGBMによる予測を管理します。

Args:
    target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
    features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
    model (lgb.train): LightGBMモデル
    scaler (Optional[object]): 使用しない（LightGBMでは不要）
    
Returns:
    pd.DataFrame: 予測結果のデータフレーム

