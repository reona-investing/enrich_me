from utils.paths import Paths
from acquisition.jquants_api_operations import StockAcquisitionFacade
from calculation.facades import CalculatorFacade
from preprocessing import PreprocessingPipeline
from preprocessing.methods import PCAHandler, FeatureNeutralizer, Standardizer
from datetime import datetime

train_start = datetime(2014,1,1)
train_end = datetime(2021,12,31)

SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv' #別でファイルを作っておく
SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet' #出力のみなのでファイルがなくてもOK
univerce_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
saf = StockAcquisitionFacade(filter=univerce_filter)
stock_dfs = saf.get_stock_data_dict()

fn = FeatureNeutralizer(mode='mutual')
std = Standardizer(fit_start=train_start, fit_end=train_end)
ppp = PreprocessingPipeline([('Standardizer', std)])


new_sector_price, stock_price_for_order, features_df = CalculatorFacade.calculate_all(stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)
df = features_df.loc[(features_df.index.get_level_values('Date')<=train_end)&(features_df.index.get_level_values('Date')>=train_start)]

new_sector_price, stock_price_for_order, features_df = CalculatorFacade.calculate_all(stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET,
                                                                                      indices_preprocessing_pipeline=ppp,
                                                                                      price_preprocessing_pipeline=ppp)
#TODO FeaturesCalculator.calculate_features_price()で、rankやcatの追加を処理後にする。

df = features_df.loc[(features_df.index.get_level_values('Date')<=train_end)&(features_df.index.get_level_values('Date')>=train_start)]
print(df.describe().T)