import pandas as pd
import paths
from jquants_api_operations.processor.formatter import Formatter
from jquants_api_operations.utils import FileHandler

def process_list(raw_path: str = paths.RAW_STOCK_LIST_PARQUET, 
                 processing_path: str = paths.STOCK_LIST_PARQUET) -> None:
    """
    銘柄リストデータを加工して、機械学習用に整形します。

    Args:
        raw_path (str): 生の銘柄リストデータが保存されているパス
        processing_path (str): 加工後の銘柄リストデータを保存するパス
    """
    raw_stock_list = FileHandler.read_parquet(raw_path)
    stock_list = _format_dtypes(raw_stock_list)
    stock_list = _extract_individual_stocks(stock_list)
    FileHandler.write_parquet(stock_list, processing_path)

def _format_dtypes(raw_stock_list: pd.DataFrame) -> pd.DataFrame:
    '''銘柄リストのデータ型をフォーマットする。'''
    str_columns = ['Code', 'CompanyName', 'MarketCodeName', 'Sector33CodeName', 'Sector17CodeName', 'ScaleCategory']
    int_columns = ['Listing']
    stock_list = raw_stock_list[str_columns + int_columns].copy()
    stock_list[str_columns] = stock_list[str_columns].astype(str)
    stock_list[int_columns] = stock_list[int_columns].astype(int)
    return Formatter.format_stock_code(stock_list)

def _extract_individual_stocks(stock_list: pd.DataFrame):
    '''ETF等を除き、個別銘柄のみを抽出します。'''
    return stock_list[stock_list['Sector33CodeName'] != 'その他'].drop_duplicates()


if __name__ == '__main__':
    process_list()