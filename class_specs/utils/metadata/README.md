# utils/metadata のクラス仕様書

## feature_metadata.py

### class FeatureMetadata
時系列データ取得用の特徴量メタデータを定義します。
name (str): 特徴量名
group (str): 特徴量グループ（commodity, currencyなど）
parquet_path (str): 時系列データ保存先のparquetファイルパス
url (str): 時系列データ取得元のURL
is_adopted (bool): 特徴量として採用するかどうか

