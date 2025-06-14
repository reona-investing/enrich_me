# acquisition/jquants_api_operations/processor のクラス仕様書

## fin_processor.py

### class FinProcessor
財務データの前処理を行うクラス。

Jquants APIで取得した財務情報データを機械学習に適した形式に変換し、整形・加工・合併処理を行う。
加工後のデータは Parquet 形式で保存する。
- __init__: インスタンス生成と同時にデータの加工と保存を行う。

Args:
    raw_path (str): 生の財務データの保存パス
    processing_path (str): 加工後の財務データの保存パス
    yaml_path (str): YAML設定ファイルのパス
- _get_col_info: 
- _get_column_mapping: 指定されたキーリストに基づいて、ColumnConfigsGetter からカラム名のマッピング辞書を作成する。

Parameters:
    keys (list): 取得するキーのリスト
    column_config_getter (ColumnConfigsGetterの関数): カラム名を取得用の関数

Returns:
    dict: 指定されたキーと取得したカラム名の辞書
- _load_fin_data: 生の財務データを読み込み、空白値を NaN に変換する。

Args:
    raw_path (str): 財務データのファイルパス

Returns:
    pd.DataFrame: 読み込んだ財務データ
- _rename_columns: 設定ファイルをもとにカラム名を変換する。

Args:
    fin_df (pd.DataFrame): フィルタリング後の財務データ

Returns:
    pd.DataFrame: カラム名変換後の財務データ
- _format_dtypes: 設定ファイルをもとに各カラムに適切なデータ型を設定。

Args:
    fin_df (pd.DataFrame): カラム名変換後の財務データ

Returns:
    pd.DataFrame: データ型変換後の財務データ
- _drop_duplicated_data: 重複データを削除し、最新の発表データを保持する。

Args:
    fin_df (pd.DataFrame): データ型変換後の財務データ

Returns:
    pd.DataFrame: 重複削除後の財務データ
- _calculate_additional_fins: 追加の財務指標を計算。

Args:
    fin_df (pd.DataFrame): 重複データ削除後の財務データ

Returns:
    pd.DataFrame: 追加計算後の財務データ
- _merge_forecast_eps: 
- _calculate_outstanding_shares: 期末発行済株式数の算出。

Args:
    fin_df (pd.DataFrame): 重複データ削除後の財務データ

Returns:
    pd.DataFrame: 追加計算後の財務データ
- _append_fiscal_year_related_columns: 年度に関する追加カラムを追加します。

Args:
    fin_df (pd.DataFrame): 重複データ削除後の財務データ

Returns:
    pd.DataFrame: 追加計算後の財務データ
- _process_merger: 企業合併時の財務情報の合成を行います。

Args:
    fin_df (pd.DataFrame): 重複データ削除後の財務データ

Returns:
    pd.DataFrame: 追加計算後の財務データ
- _finalize_df: 最終データの整形処理を行う。

Args:
    stock_fin (pd.DataFrame): 財務データの最終処理前の状態

Returns:
    pd.DataFrame: 最終処理後のデータ

## formatter.py

### class Formatter
- format_stock_code: 普通株の銘柄コードを4桁に変換するヘルパー関数

## list_processor.py

### class ListProcessor
- __init__: 銘柄リストデータを加工して、機械学習用に整形します。

Args:
    raw_path (str): 生の銘柄リストデータが保存されているパス
    processing_path (str): 加工後の銘柄リストデータを保存するパス
- _format_dtypes: 銘柄リストのデータ型をフォーマットする。
- _extract_individual_stocks: ETF等を除き、個別銘柄のみを抽出します。

## price_processor.py

### class PriceProcessor
- __init__: 価格情報を加工して、機械学習用に整形します。

Args:
    raw_basic_path (str): 生の株価データが保存されているパス。
    processing_basic_path (str): 加工後の株価データを保存するパス。
- _get_col_info: 
- _get_column_mapping: 指定されたキーリストに基づいて、ColumnConfigsGetter からカラム名のマッピング辞書を作成する。

Parameters:
    keys (list): 取得するキーのリスト
    column_config_getter (ColumnConfigsGetterの関数): カラム名を取得用の関数

Returns:
    dict: 指定されたキーと取得したカラム名の辞書
- _load_yearly_raw_data: 取得したままの年次株価データを読み込みます。
- _get_dict_for_rename: 
- _process_stock_price: 価格データを加工します。
Args:
    stock_price (pd.DataFrame): 加工前の株価データ
    temp_cumprod (dict[str, float]): 
        処理時点での銘柄ごとの暫定の累積調整係数を格納（キー: 銘柄コード、値：暫定累積調整係数）
    is_latest_file (bool): stock_priceが最新期間のファイルかどうか
Returns:
    pd.DataFrame: 加工された株価データ。
- _replace_code: ルールに従い、銘柄コードを置換します。
- _fill_suspension_period: 銘柄コード変更前後の欠損期間のデータを埋めます。
- _get_missing_dates: データが欠損している日付を取得します。
- _create_missing_rows: 欠損期間の行を作成します。
- _format_dtypes: データ型を整形します。
- _remove_system_failure_day: システム障害によるデータ欠損日を除外します。
- _apply_cumulative_adjustment_factor: 価格データ（OHLCV）に累積調整係数を適用します。
- _calculate_cumulative_adjustment_factor: 累積調整係数を計算します。
- _inherit_cumulative_values: 計算途中の暫定累積調整係数を引き継ぎます。
- _apply_manual_adjustments: 元データで未掲載の株式分割・併合について、累積調整係数をマニュアルで調整する
- _finalize_price_data: 最終的なデータ整形を行う。
- _save_yearly_data: 年次の加工後価格データを保存する。

