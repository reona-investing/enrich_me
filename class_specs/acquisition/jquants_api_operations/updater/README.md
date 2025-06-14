# acquisition/jquants_api_operations/updater のクラス仕様書

## fin_updater.py

### class FinUpdater
- __init__: 財務情報の更新を行い、指定されたパスにParquet形式で保存する。
:param path: Parquetファイルの保存先パス
- _load_column_configs: 
- _load_existing_data: 既存データを読み込む。ファイルが存在しない場合は空のDataFrameを返す。
- _fetch_data: 財務情報をAPIから取得する。
- _merge: 既存データと新規データを結合し、重複を排除する。
- _format: データを指定された形式に整形する。

## list_updater.py

### class ListUpdater
- __init__: 銘柄一覧の更新を行い、指定されたパスにParquet形式で保存する。

:param path: Parquetファイルの保存先パス
- _fetch: JQuants APIからデータを取得する
- _merge: 取得したデータと既存のデータをマージする。

:param fetched_stock_list_df: 取得した銘柄一覧
:param existing_stock_list_df: 既存の銘柄一覧
:return: マージ後の銘柄一覧
- _format: データフレームを指定された形式に整形する。

:param stock_list_df: 整形対象のデータフレーム
:return: 整形後のデータフレーム

## price_updater.py

### class PriceUpdater
- __init__: 株価情報を更新し、指定されたパスにParquet形式で保存します。
:param basic_path: パスのテンプレート（例: "path_to_data/0000.parquet"）
- _load_column_configs: 
- _generate_file_path: 指定された年に基づいてファイルパスを生成します。
- _update_yearly_stock_price: 特定の年の株価情報を取得し、更新します。
:param year: 対象の年
:param yearly_path: 年ごとのファイルパス
- _update_yearly_price: 年次データを取得し、既存データを更新します。
:param year: 対象の年
:param existing_data: 既存の価格情報データ
- _fetch_full_year_stock_price: 指定された年の全期間の価格情報を取得します。
- _fetch_new_stock_price: 最新の日付までの新しい価格情報を取得します。
- _set_adjustment_flag: AdjustmentFactorが変更された場合のフラグを設定します。
- _update_raw_stock_price: 既存の価格情報に新しいデータを追加し、重複を削除します。
- needs_processing: 価格データの再加工が必要かどうかを示す。

