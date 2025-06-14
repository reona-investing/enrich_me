# calculation/sector_index のクラス仕様書

## sector_index.py

### class SectorIndex
Stock informationからセクターインデックスを計算するメインクラス。
- __init__: 各種データのパスとデータフレームを受け取り初期化を行う。

Args:
    stock_dfs_dict (dict): ``'price'`` および ``'fin'`` をキーにもつデータフレーム辞書。
    sector_redefinitions_csv (str): セクター再定義 CSV へのパス。
    sector_index_parquet (str): 計算結果を保存する parquet ファイルのパス。
- _get_column_names: 
- calc_sector_index: セクターインデックスを算出して ``order_price_df`` も返す。

``stock_dfs_dict`` で渡された株価データと財務データを用いて
時価総額を計算し、CSV で指定されたセクター定義を結合して
セクターごとの指数を計算する。結果は ``sector_index_parquet``
に保存される。

Returns:
    tuple[pd.DataFrame, pd.DataFrame]:
        計算したセクターインデックスと発注処理用株価データ。
- calc_sector_index_by_dict: 辞書形式のセクター定義からインデックスを計算する。

Args:
    sector_stock_dict (dict):
        セクター名をキー、銘柄コードのリストを値とする辞書。
        例: ``{"JPY感応株": ["6758", "6501"], "JPbond感応株": ["6751", ...]}``
    stock_price_data (pd.DataFrame):
        ``calc_marketcap`` の結果と同じ形式の株価データ。

Returns:
    pd.DataFrame: セクターインデックスのデータフレーム。
- get_sector_index_dict: セクター別のデータフレーム辞書を取得する。

:meth:`calc_sector_index` が未実行の場合は内部で呼び出して
計算結果をキャッシュする。

Returns
-------
tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
    セクターインデックスと発注処理用の価格データを、それぞれセクター別に分割した辞書を返す。
- calc_marketcap: 各銘柄の日次時価総額を計算する。

Args:
    stock_price (pd.DataFrame): 株価データ。
    stock_fin (pd.DataFrame): 財務データ。

Returns:
    pd.DataFrame: 時価総額および補正値を付加した株価データ。

