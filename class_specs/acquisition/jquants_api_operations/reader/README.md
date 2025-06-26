# acquisition/jquants_api_operations/reader のクラス仕様書

## reader.py

### class Reader
- __init__: 抽出条件を指定します。filterとfiltered_code_listを両方設定した場合、filterの条件が優先されます。
Args:
    filter (str): 読み込み対象銘柄のフィルタリング条件（クエリ）
    filtered_code_list (list[str]): フィルタリングする銘柄コードのをリストで指定
- read_list: 銘柄一覧を読み込みます。filterとfiltered_code_listを両方設定した場合、filterの条件が優先されます。
Args:
    path (str): 銘柄一覧のparquetファイルのパス
Returns:
    pd.DataFrame: 銘柄一覧
- read_fin: 財務情報データを読み込みます。filterとfiltered_code_listを両方設定した場合、filterの条件が優先されます。
Args:
    path (str): 財務情報のparquetファイルのパス
    list_path (str): 銘柄一覧のparquetファイルのパス（フィルタリング用）
    end_date (datetime): データの終了日
Returns:
    pd.DataFrame: 財務情報
- read_price: 価格情報を読み込み、調整を行います。
Args:
    basic_path (str): 株価データのparquetファイルのパス
    list_path (str): 銘柄一覧のparquetファイルのパス（フィルタリング用）
    end_date (datetime): データの終了日
Returns:
    pd.DataFrame: 価格情報
- _generate_price_df: 年次の株価データから、通期の価格データフレームを生成します。
- _recalc_adjustment_factors: 全銘柄の最終日の累積調整係数が1となるように再計算します。

