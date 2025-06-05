#%% 株価データ辞書の読み込み
from acquisition.jquants_api_operations import StockAcquisitionFacade
universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
saf = StockAcquisitionFacade(filter=universe_filter)
stock_dfs = saf.get_stock_data_dict()

#%% セクターインデックスの算出（旧法）
from utils.paths import Paths
import pandas as pd
SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet'

from calculation.sector_index.sector_index import SectorIndex
sic = SectorIndex()
sector_df, order_price_df = sic.calc_sector_index(stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)


print(sector_df)
print(sic.sector_index_df)
