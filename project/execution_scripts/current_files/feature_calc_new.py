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

from calculation.sector_index_calculator import SectorIndexCalculator
sector_df, order_price_df = SectorIndexCalculator.calc_sector_index(stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)

print(sector_df)

#%% セクターインデックスの算出（新法）
from calculation.sector_index import SectorIndex

si = SectorIndex()
si.calculate_sector_index()