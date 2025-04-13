from facades.stock_acquisition_facade import StockAcquisitionFacade
from calculation.sector_index_calculator import SectorIndexCalculator
from calculation.target_calculator import TargetCalculator
from calculation.features_calculator import FeaturesCalculator
from utils.paths import Paths
from datetime import datetime

from machine_learning.factory import CollectionFactory


universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
SECTOR_REDEFINITIONS_CSV = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv' #別でファイルを作っておく
SECTOR_INDEX_PARQUET = f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet' #出力のみなのでファイルがなくてもOK
train_start_date = datetime(2014, 1, 1)
train_end_date = datetime(2021, 12, 31)

stock_dfs = StockAcquisitionFacade(filter=universe_filter).get_stock_data_dict()
sector_index_df, _ = SectorIndexCalculator.calc_new_sector_price(stock_dfs, SECTOR_REDEFINITIONS_CSV, SECTOR_INDEX_PARQUET)


raw_target_df, target_df = \
    TargetCalculator.daytime_return_PCAresiduals(sector_index_df, reduce_components=1, train_start_day=train_start_date, train_end_day=train_end_date)
features_df = FeaturesCalculator.calculate_features(sector_index_df, None, None,
                                                    adopts_features_indices = True, adopts_features_price = False,
                                                    groups_setting = None, names_setting = None, currencies_type = 'relative',
                                                    adopt_1d_return = True, mom_duration = None, vola_duration = None,
                                                    adopt_size_factor = False, adopt_eps_factor = False,
                                                    adopt_sector_categorical = False, add_rank = False)


sectors = target_df.index.get_level_values('Sector').unique().tolist()

collection_path = 'C:/Users/ryosh/enrich_me/project/model_collection/test.pkl'
lasso_collection = CollectionFactory.get_collection()

for sector in sectors:
    target_for_sector = target_df[target_df.index.get_level_values('Sector') == sector]
    features_for_sector = features_df[features_df.index.get_level_values('Sector') == sector]
    
    lasso_collection.generate_model(name = sector, type='lasso')
    single_model = lasso_collection.get_model(name = sector)
    single_model.load_dataset(target_for_sector, features_for_sector, train_start_date, train_end_date)
    lasso_collection.set_model(model = single_model)

lasso_collection.train_all()
lasso_collection.predict_all()

pred_result_df = lasso_collection.get_result_df()
lasso_collection.save(path = collection_path)

print(pred_result_df)