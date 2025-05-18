#%%
import os
from pathlib import Path
from enum import Enum

class Paths:
    # 設定ファイル・モジュール・実行ファイルはGitHubで管理するためローカルに保存
    PROJECT_FOLDER = Path.resolve(Path(__file__)).parents[3]
    MODULES_FOLDER = Path.resolve(Path(__file__)).parents[2]

    CONFIG_AND_SETTINGS_FOLDER = f'{PROJECT_FOLDER}/config_and_settings'
    EXECUTION_SCRIPTS_FOLDER = f'{PROJECT_FOLDER}/execution_scripts'
    SECTOR_REDEFINITIONS_FOLDER = f'{PROJECT_FOLDER}/sector_redefinitions'
    SECTOR_PRICE_FOLDER = f'{SECTOR_REDEFINITIONS_FOLDER}/sector_price'
    SECTOR_FIN_FOLDER = f'{SECTOR_REDEFINITIONS_FOLDER}/sector_fin'

    CODES_TO_REPLACE_CSV = f'{CONFIG_AND_SETTINGS_FOLDER}/codes_to_replace.csv'
    FEATURES_TO_SCRAPE_CSV = f'{CONFIG_AND_SETTINGS_FOLDER}/features_to_scrape.csv'
    
    RAW_STOCK_PRICE_COLUMNS_YAML = f'{CONFIG_AND_SETTINGS_FOLDER}/raw_stock_price_cols.yaml'
    RAW_STOCK_FIN_COLUMNS_YAML = f'{CONFIG_AND_SETTINGS_FOLDER}/raw_stock_fin_cols.yaml'
    STOCK_PRICE_COLUMNS_YAML = f'{CONFIG_AND_SETTINGS_FOLDER}/stock_price_cols.yaml'
    STOCK_FIN_COLUMNS_YAML = f'{CONFIG_AND_SETTINGS_FOLDER}/stock_fin_cols.yaml'
    SECTOR_INDEX_COLUMNS_YAML = f'{CONFIG_AND_SETTINGS_FOLDER}/sector_index_cols.yaml'
    SECTOR_FIN_COLUMNS_YAML = f'{CONFIG_AND_SETTINGS_FOLDER}/sector_fin_cols.yaml'


    # オンラインバックアップのパス
    ONLINE_STORAGE_FOLDER = 'H:/マイドライブ/enrich_me'
    ONLINE_PROJECT_FOLDER = f'{ONLINE_STORAGE_FOLDER}/project'

    STOCK_RAWDATA_FOLDER = f'{ONLINE_PROJECT_FOLDER}/rawdata_JQuantsAPI'
    STOCK_PROCESSED_DATA_FOLDER = f'{ONLINE_PROJECT_FOLDER}/processed_data_JQuantsAPI'
    SCRAPED_DATA_FOLDER = f'{ONLINE_PROJECT_FOLDER}/scraped_data'
    ML_DATASETS_FOLDER = f'{ONLINE_PROJECT_FOLDER}/ml_datasets'
    TRADE_HISTORY_FOLDER = f'{ONLINE_PROJECT_FOLDER}/trade_history'
    SUMMARY_REPORTS_FOLDER = f'{ONLINE_PROJECT_FOLDER}/summary_reports'
    DEBUG_FILES_FOLDER = f'{ONLINE_PROJECT_FOLDER}/debug_files'
    DOWNLOAD_FOLDER = f'{ONLINE_PROJECT_FOLDER}/temp_files'
    ORDERS_FOLDER = f'{ONLINE_PROJECT_FOLDER}/orders'
    POSITIONS_FOLDER = f'{ONLINE_PROJECT_FOLDER}/positions'

    ERROR_LOG_CSV = f'{DEBUG_FILES_FOLDER}/error_log.csv'


    RAW_STOCK_LIST_PARQUET = f'{STOCK_RAWDATA_FOLDER}/raw_stock_list.parquet'
    RAW_STOCK_PRICE_PARQUET = f'{STOCK_RAWDATA_FOLDER}/raw_stock_price_0000.parquet'
    RAW_STOCK_FIN_PARQUET = f'{STOCK_RAWDATA_FOLDER}/raw_stock_fin.parquet'

    STOCK_LIST_PARQUET = f'{STOCK_PROCESSED_DATA_FOLDER}/stock_list.parquet'
    STOCK_PRICE_PARQUET = f'{STOCK_PROCESSED_DATA_FOLDER}/stock_price_0000.parquet'
    STOCK_FIN_PARQUET = f'{STOCK_PROCESSED_DATA_FOLDER}/stock_fin.parquet'

    ORDERS_CSV = f'{DEBUG_FILES_FOLDER}/orders_list.csv'
    FAILED_ORDERS_CSV = f'{DEBUG_FILES_FOLDER}/failed_list.csv'

    TRADE_HISTORY_CSV = f'{TRADE_HISTORY_FOLDER}/trade_history.csv'
    BUYING_POWER_HISTORY_CSV = f'{TRADE_HISTORY_FOLDER}/buying_power_history.csv'
    DEPOSIT_HISTORY_CSV = f'{TRADE_HISTORY_FOLDER}/deposit_history.csv'


    TRADE_HISTORY_BACKUP = f'{ONLINE_STORAGE_FOLDER}/trade_history.csv'
    BUYING_POWER_HISTORY_BACKUP = f'{ONLINE_STORAGE_FOLDER}/buying_power_history.csv'
    DEPOSIT_HISTORY_BACKUP = f'{ONLINE_STORAGE_FOLDER}/deposit_history.csv'
    ERROR_LOG_BACKUP = f'{ONLINE_STORAGE_FOLDER}/error_log_backup.csv'


#%%
if __name__ == '__main__':
    import pandas as pd
    from IPython.display import display
    df = pd.read_csv(Paths.TRADE_HISTORY_CSV)
    display(df)
