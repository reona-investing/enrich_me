#%%
import os
from pathlib import Path


#%%
#本ファイルの親フォルダをプロジェクトフォルダとする。
PROJECT_FOLDER = Path.resolve(Path(__file__)).parents[1]

MODULES_FOLDER = Path.resolve(Path(__file__)).parents[0]
CONFIG_AND_SETTINGS_FOLDER = f'{PROJECT_FOLDER}/config_and_settings'
STOCK_RAWDATA_FOLDER = f'{PROJECT_FOLDER}/rawdata_JQuantsAPI'
STOCK_PROCESSED_DATA_FOLDER = f'{PROJECT_FOLDER}/processed_data_JQuantsAPI'
SCRAPED_DATA_FOLDER = f'{PROJECT_FOLDER}/scraped_data'
EXECUTION_SCRIPTS_FOLDER = f'{PROJECT_FOLDER}/execution_scripts'
SECTOR_REDEFINITIONS_FOLDER = f'{PROJECT_FOLDER}/sector_redefinitions'
SECTOR_PRICE_FOLDER = f'{SECTOR_REDEFINITIONS_FOLDER}/sector_price'
ML_DATASETS_FOLDER = f'{PROJECT_FOLDER}/ml_datasets'
TRADE_HISTORY_FOLDER = f'{PROJECT_FOLDER}/trade_history'
SUMMARY_REPORTS_FOLDER = f'{PROJECT_FOLDER}/summary_reports'
DEBUG_FILES_FOLDER = f'{PROJECT_FOLDER}/debug_files'
DOWNLOAD_FOLDER = f'{PROJECT_FOLDER}/temp_files'

CODES_TO_REPLACE_CSV = f'{CONFIG_AND_SETTINGS_FOLDER}/codes_to_replace.csv'
DTYPES_STOCK_FIN_CSV = f'{CONFIG_AND_SETTINGS_FOLDER}/dtypes_stock_fin.csv'
FEATURES_TO_SCRAPE_CSV = f'{CONFIG_AND_SETTINGS_FOLDER}/features_to_scrape.csv'

RAW_STOCK_LIST_PARQUET = f'{STOCK_RAWDATA_FOLDER}/raw_stock_list.parquet'
RAW_STOCK_PRICE_PARQUET = f'{STOCK_RAWDATA_FOLDER}/raw_stock_price_0000.parquet'
RAW_STOCK_FIN_PARQUET = f'{STOCK_RAWDATA_FOLDER}/raw_stock_fin.parquet'

STOCK_LIST_PARQUET = f'{STOCK_PROCESSED_DATA_FOLDER}/stock_list.parquet'
STOCK_PRICE_PARQUET = f'{STOCK_PROCESSED_DATA_FOLDER}/stock_price_0000.parquet'
STOCK_FIN_PARQUET = f'{STOCK_PROCESSED_DATA_FOLDER}/stock_fin.parquet'

LONG_ORDERS_CSV = f'{DEBUG_FILES_FOLDER}/long_list.csv'
SHORT_ORDERS_CSV = f'{DEBUG_FILES_FOLDER}/short_list.csv'

TRADE_HISTORY_CSV = f'{TRADE_HISTORY_FOLDER}/trade_history.csv'
BUYING_POWER_HISTORY_CSV = f'{TRADE_HISTORY_FOLDER}/buying_power_history.csv'
DEPOSIT_HISTORY_CSV = f'{TRADE_HISTORY_FOLDER}/deposit_history.csv'

#%%
if __name__ == '__main__':
    import pandas as pd
    from IPython.display import display
    df = pd.read_csv(TRADE_HISTORY_CSV)
    display(df)
