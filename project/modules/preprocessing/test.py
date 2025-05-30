from utils.paths import Paths
from acquisition.jquants_api_operations import StockAcquisitionFacade
from calculation.facades import CalculatorFacade

SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv' #別でファイルを作っておく
SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet' #出力のみなのでファイルがなくてもOK
univerce_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
saf = StockAcquisitionFacade(filter=univerce_filter)
stock_dfs = saf.get_stock_data_dict()
stock_dfs['price']
new_sector_price, stock_price_for_order, features_df = CalculatorFacade.calculate_all(stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET,
                                                                                      adopts_features_price=False)
print(features_df.index)