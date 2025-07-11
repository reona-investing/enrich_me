from datetime import datetime

from utils.paths import Paths
from utils.timeseries import Duration
from acquisition.jquants_api_operations import StockAcquisitionFacade
from calculation import SectorIndex

from timeseries_data.public import ReturnTimeseries, ReturnTimeseriesCollection
from timeseries_data.calculation_method import IntradayReturn
from timeseries_data.preprocessing import PreprocessingPipeline, PCAHandler
# from yyy import RemovingPC #前処理クラス群を定義（コンストラクタの引数にデータフレームを取り，calculateメソッドで同じ形のデータフレームを返す）

universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400')|(ScaleCategory=='TOPIX Small 1'))"
sector_redef_path = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/56sectors_2024-2025.csv"
sector_index_path = f"test.parquet"
train_duration = Duration(start=datetime(2014, 1, 1), end=datetime(2021, 12, 31))
train_duration.extract_from_df

stock_dfs = StockAcquisitionFacade(filter=universe_filter).get_stock_data_dict()

sector_index, _ = SectorIndex(stock_dfs, sector_redef_path, sector_index_path).calc_sector_index()

sectors: list[str] = sector_index.index.get_level_values('Sector').unique().tolist()

sector_dfs = {sector: sector_index[sector_index.index.get_level_values('Sector')==sector].droplevel(1, axis=0) for sector in sectors}

return_timeseries_collection = ReturnTimeseriesCollection()

for sector, sector_df in sector_dfs.items():
    return_timeseries = ReturnTimeseries(original_timeseries = sector_df, calculated_column = sector, sector_column=None)
    return_timeseries.calculate(method=IntradayReturn(return_column=sector))
    print(return_timeseries.raw_return)
    return_timeseries_collection.append(return_timeseries)

'''
ppp = PreprocessingPipeline(steps = [
    ('remove_pc1', PCAHandler(n_components=1, mode='components', fit_duration=train_duration))
    ])
return_timeseries.preprocess(pipeline = ppp)
'''

print(return_timeseries_collection.raw_merged_df)
