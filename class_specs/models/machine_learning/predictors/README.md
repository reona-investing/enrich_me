# models/machine_learning/predictors のクラス仕様書

## base_predictor.py

### class BasePredictor
機械学習モデルの予測器の抽象基底クラス
- __init__: Args:
    target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
    features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
    models (List[Any]): 学習済みモデルのリスト
    scalers (Optional[List[Any]]): スケーラーのリスト（必要な場合）
- _validate_inputs: 入力の妥当性をチェック
- predict: 予測を行う抽象メソッド

Returns:
    pd.DataFrame: 予測結果を格納したデータフレーム
- _is_multi_sector: マルチセクターかどうかを判定
- _get_sectors: セクター一覧を取得

## lasso_predictor.py

### class LassoPredictor
LASSOモデルの予測器クラス
- __init__: LASSOによる予測を管理します。

Args:
    target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
    features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
    models (List[Lasso]): LASSOモデルを格納したリスト
    scalers (List[StandardScaler]): LASSOモデルに対応したスケーラーを格納したリスト
- predict: LASSOによる予測を行います。シングルセクターとマルチセクターの双方に対応しています。

Returns:
    pd.DataFrame: 予測結果を格納したデータフレーム
- _pred_single_sector: LASSOモデルで予測して予測結果を返す関数
- _pred_multi_sectors: 複数セクターに関して、LASSOモデルで予測して予測結果を返す関数

## lgbm_predictor.py

### class LgbmPredictor
LightGBMモデルの予測器クラス
- __init__: lightGBMによる予測を管理します。

Args:
    target_test_df (pd.DataFrame): テスト用の目的変数データフレーム
    features_test_df (pd.DataFrame): テスト用の特徴量データフレーム
    models (List[lgb.train]): lightGBMモデルを格納したリスト
    categorical_features (Optional[List[str]]): カテゴリ変数の一覧
- predict: lightGBMによる予測を行います。シングルセクターとマルチセクターの双方に対応しています。

Returns:
    pd.DataFrame: 予測結果を格納したデータフレーム

