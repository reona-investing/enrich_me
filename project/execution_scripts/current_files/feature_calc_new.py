#%% 株価データ辞書の読み込み
from acquisition.jquants_api_operations import StockAcquisitionFacade

saf = StockAcquisitionFacade()
stock_dfs = saf.get_stock_data_dict()

#%% セクター情報の読み込み
from utils.paths import Paths
import pandas as pd
SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
sector_definitions_df = pd.read_csv(SECTOR_REDEFINITIONS_CSV)
print(sector_definitions_df)
#%% セクターインデックスの算出
from calculation.sector_index_calculator import SectorIndexCalculator

sic = SectorIndexCalculator.calc_marketcap()