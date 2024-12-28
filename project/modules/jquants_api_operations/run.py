from jquants_api_operations.updater import update_list, update_fin, update_price
from jquants_api_operations.processor import process_list, process_price, process_fin
from jquants_api_operations.reader import read_list, read_fin, read_price
from typing import List
import pandas as pd

def run_jquants_api_operations(update: bool = False, process: bool = False, read: bool = False,
                               filter: str = None, filtered_code_list: str = None) -> List[pd.DataFrame]:
    """
    一括処理を行います。
    引数:
        update (bool): True の場合、データを更新します。
        process (bool): True の場合、データを加工します。
        read (bool): True の場合、データを読み込み。
        filter (str): 読み込み対象銘柄のフィルタリング条件（クエリ）
        filtered_code_list (list[str]): フィルタリングする銘柄コードのをリストで指定
    戻り値:
        pd.DataFrame: 銘柄一覧
        pd.DataFrame: 財務情報
        pd.DataFrame: 株価情報
    """
    list_df = None
    fin_df = None
    price_df = None

    if update:
        update_list()
        update_fin()
        update_price()

    if process:
        process_list()
        process_fin()
        process_price()

    if read:
        list_df = read_list(filter=filter, filtered_code_list=filtered_code_list)
        fin_df = read_fin(filter=filter, filtered_code_list=filtered_code_list)
        price_df = read_price(filter=filter, filtered_code_list=filtered_code_list)

    return list_df, fin_df, price_df

if __name__ == "__main__":
    # Example usage
    filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    list_df, fin_df, price_df = run_jquants_api_operations(update=True, process=True, read=True, filter=filter)