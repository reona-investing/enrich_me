# acquisition/jquants_api_operations/facades のクラス仕様書

## stock_acquisition_facade.py

### class StockAcquisitionFacade
- __init__: インスタンス生成時に株式データの一括読み込みを行います。
引数:
    update (bool): True の場合、データを更新します。
    process (bool): True の場合、データを加工します。
    filter (str): 読み込み対象銘柄のフィルタリング条件（クエリ）
    filtered_code_list (list[str]): フィルタリングする銘柄コードのをリストで指定
- get_stock_data: targetに指定した文字列をもとに、適切なデータフレームを返します。
Args:
    target (Literal['list', 'fin', 'price']): どのデータフレームを取得したいか
Returns:
    pd.DataFrame: targetの文字列に合わせたデータフレーム
- get_stock_data_dict: list, fin, priceを一つの辞書として返す。
Returns:
    dict[pd.DataFrame]

