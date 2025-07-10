# calculation/target のクラス仕様書

## pca_for_sector_target.py

### class PCAforMultiSectorTarget
機械学習用途特化PCA前処理ファサード（簡略化版）

特定用途（ML目的変数前処理）に特化したシンプルなファサードパターン。
内部でPCAHandlerを使用し、ML特化の前後処理を提供。
時系列対応により大幅に簡略化。

Parameters
----------
n_components : int
    抽出する主成分の数
fit_duration : Duration or None, optional
    学習期間を表すDuration。Noneの場合は全期間を使用
target_column : str, default='Target'
    対象となる列名
mode : str, default='residuals'
    'residuals': 残差を抽出
    'components': 主成分を抽出
    'transform': PCA変換結果を直接取得
copy : bool, default=True
    データをコピーするかどうか
random_state : int, optional
    乱数シード
- __init__: 
- apply_pca: ML用途特化のPCA前処理を実行

初回実行時は学習も同時に行い、2回目以降は学習済みパラメータで変換のみ実行。
BasePreprocessorの時系列機能により大幅に簡略化。

Parameters
----------
X : pd.DataFrame
    処理対象データ（二階層インデックス必須）
    
Returns
-------
X_transformed : pd.DataFrame
    PCA処理後のデータ
- _prepare_data: 学習・変換用データフレームを準備（ML特化処理）
- _restore_dataframe_format: 変換後の配列をDataFrame形式に復元
- get_explained_variance_ratio: 各主成分の寄与率を取得
- get_cumulative_explained_variance_ratio: 累積寄与率を取得
- get_components: 主成分ベクトルをDataFrame形式で取得
- get_feature_names_out: 変換後の特徴量名を取得
- get_fit_info: fit状態と設定情報を取得（PCAHandlerの情報も含む）
- is_fitted: fit状態を確認
- fit_start: fit開始日を取得
- fit_end: fit終了日を取得
- underlying_pca: 内部のPCAHandlerインスタンスにアクセス（上級者向け）

## target_calculator.py

### class TargetCalculator
- daytime_return: 日中生リターンを算出する。
Args:
    df (pd.DataFrame): 元データ（Open, Closeの各列が必須）
Returns:
    pd.DataFrame: 日中リターン（Target列）を含むDataFrame

### class Pc1RemovedIntradayReturn
PC1 を除去した日内リターンを計算するファサード。
 - `calculate(fit_duration: Duration)`: PCA の学習期間を指定して計算を実行。
 - `processed_return`: PCA 適用後のリターンを取得。
 - `raw_return`: PCA 適用前の生リターンを取得。

