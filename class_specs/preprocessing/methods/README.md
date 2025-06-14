# preprocessing/methods のクラス仕様書

## base_preprocessor.py

### class BasePreprocessor
前処理クラスの抽象基底クラス

全ての前処理クラスが継承すべき共通インターフェースを定義。
scikit-learn互換のTransformerとして動作。
時系列データに対応し、指定期間でfitして全期間でtransformすることが可能。

Parameters
----------
copy : bool, default=True
    データをコピーするかどうか
fit_start : str, pd.Timestamp, or None, default=None
    fitに使用する開始日時。Noneの場合は全期間を使用
fit_end : str, pd.Timestamp, or None, default=None
    fitに使用する終了日時。Noneの場合は全期間を使用
time_column : str or None, default='Date'
    時間列の名前。Noneの場合はindexを時間として使用
- __init__: 
- _filter_data_by_time: 指定された期間でデータをフィルタリング

Parameters
----------
X : pd.DataFrame or np.ndarray
    入力データ
start : str, pd.Timestamp, or None
    開始日時
end : str, pd.Timestamp, or None
    終了日時
    
Returns
-------
Union[pd.DataFrame, np.ndarray]
    フィルタリングされたデータ
- fit: 前処理のパラメータを学習
- transform: 学習したパラメータを使って変換を実行
- fit_transform: fit と transform を同時に実行
- _validate_input: 入力データの妥当性をチェック
- _mark_as_fitted: fit完了をマークし、重要な属性を記録

このメソッドを継承クラスのfit()の最後で呼び出すだけで
バリデーションが機能するようになる

Parameters
----------
**kwargs : 任意のキーワード引数
    fit時に設定された重要な属性を渡す
    例: self._mark_as_fitted(pca_=self.pca_, n_components_=self.n_components)
- _check_is_fitted: fitが実行済みかチェック（簡素化版）

Parameters
----------
additional_attributes : List[str], optional
    追加でチェックしたい属性があれば指定
- _store_fit_metadata: fit時に共通メタデータを保存するヘルパーメソッド
- get_feature_names_out: 変換後の特徴量名を取得（sklearn互換）
- _prepare_output: 出力データの準備（コピーの処理など）
- _get_feature_names_from_input: 入力データから特徴量名を取得
- get_fit_info: fit状態と設定された属性の情報を取得

## feature_neutralizer.py

### class FeatureNeutralizer
特徴量の直交化を行うTransformer

BasePreprocessorを継承し、統一されたインターフェースを提供。
指定期間でfitし、全期間でtransformすることが可能。

Parameters
----------
target_features : str, list of str, or None
    直交化対象の列名。Noneの場合は全列を互いに直交化
neutralize_features : str, list of str, or None
    直交化に使用する列名。target_featuresがNoneでない場合に必須
mode : str, default='mutual'
    'mutual': 全列を互いに直交化
    'specific': 指定列を指定列で直交化
copy : bool, default=True
    データをコピーするかどうか
fit_intercept : bool, default=False
    線形回帰で切片を含めるかどうか
fit_start : str, pd.Timestamp, or None, default=None
    fitに使用する開始日時
fit_end : str, pd.Timestamp, or None, default=None
    fitに使用する終了日時
time_column : str or None, default='Date'
    時間列の名前
- __init__: 
- _validate_params: パラメータの妥当性をチェック
- fit: 直交化のパラメータを学習（指定期間のデータを使用）
- _fit_regression_coefficients: 特定の直交化のための回帰係数を学習
- _fit_mutual_neutralization: 相互直交化のためのパラメータを学習
- transform: 直交化を実行（全期間のデータに適用）
- _apply_specific_neutralization: 特定の直交化を適用
- _apply_mutual_neutralization: 相互直交化を適用（簡単な実装例）
- _ensure_list: ヘルパーメソッド

## pca_handler.py

### class PCAHandler
汎用PCA処理クラス

numpy配列またはDataFrameの数値データに対してPCAを適用し、
主成分の抽出、残差の取得、またはPCA変換結果の取得を行う。
指定期間でfitし、全期間でtransformすることが可能。

Parameters
----------
n_components : int
    抽出する主成分の数
mode : str, default='components'
    'components': 主成分を抽出（逆変換して元空間に戻す）
    'residuals': 残差を抽出
    'transform': PCA変換結果を直接取得
copy : bool, default=True
    データをコピーするかどうか
random_state : int, optional
    乱数シード
fit_start : str, pd.Timestamp, or None, default=None
    fitに使用する開始日時
fit_end : str, pd.Timestamp, or None, default=None
    fitに使用する終了日時
time_column : str or None, default='Date'
    時間列の名前
- __init__: 
- _validate_params: パラメータの妥当性をチェック
- fit: PCAのパラメータを学習（指定期間のデータを使用）
- transform: PCA変換を実行（全期間のデータに適用）
- _to_numeric_array: ヘルパーメソッド - 数値データを配列に変換
- _handle_missing_values: 欠損値を処理（fit時の統計で補完）
- _extract_components: ヘルパーメソッド - 主成分を抽出
- _extract_residuals: ヘルパーメソッド - 残差を抽出
- get_explained_variance_ratio: 寄与率を取得
- get_cumulative_explained_variance_ratio: 累積寄与率を取得
- get_components: 主成分ベクトルを取得

## standardizer.py

### class Standardizer
指定期間でfitする標準化Transformer

BasePreprocessorを継承し、統一されたインターフェースを提供。
指定期間でStandardScalerをfitし、全期間でtransformすることが可能。

Parameters
----------
with_mean : bool, default=True
    平均を0にするかどうか（標準化）
with_std : bool, default=True
    標準偏差を1にするかどうか（標準化）
target_columns : list of str or None, default=None
    標準化対象の列名。Noneの場合は全ての数値列を対象
copy : bool, default=True
    データをコピーするかどうか
fit_start : str, pd.Timestamp, or None, default=None
    fitに使用する開始日時
fit_end : str, pd.Timestamp, or None, default=None
    fitに使用する終了日時
time_column : str or None, default='Date'
    時間列の名前
    
Attributes
----------
scaler_ : StandardScaler
    fit済みのStandardScaler
target_columns_ : list of str
    実際に標準化対象となった列名
mean_ : array-like
    fit期間の平均値
scale_ : array-like
    fit期間のスケール（標準偏差）
var_ : array-like
    fit期間の分散
- __init__: 
- fit: 標準化のパラメータを学習（指定期間のデータを使用）
- transform: 標準化を実行（全期間のデータに適用）
- inverse_transform: 標準化を逆変換（標準化前の値に戻す）
- _get_target_columns: 標準化対象の列を決定
- _extract_target_data: 標準化対象のデータを配列として抽出
- _restore_data_format: 変換結果を元の形式に復元
- get_feature_names_out: 変換後の特徴量名を取得（sklearn互換）
- get_statistics: fit期間の統計量を取得

