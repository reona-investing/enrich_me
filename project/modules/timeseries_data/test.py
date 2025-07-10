from datetime import datetime

from utils.paths import Paths
from utils.timeseries import Duration
from acquisition.jquants_api_operations import StockAcquisitionFacade
from calculation import SectorIndex

from timeseries_data.public import StockReturnTimeseries
from timeseries_data.calculation_method import IntradayReturn, OvernightReturn, DailyReturn
from timeseries_data.preprocessing import PreprocessingPipeline, PCAHandler
# from yyy import RemovingPC #前処理クラス群を定義（コンストラクタの引数にデータフレームを取り，calculateメソッドで同じ形のデータフレームを返す）

universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400')|(ScaleCategory=='TOPIX Small 1'))"
sector_redef_path = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/56sectors_2024-2025.csv"
sector_index_path = f"test.parquet"
train_duration = Duration(start=datetime(2014, 1, 1), end=datetime(2021, 12, 31))
train_duration.extract_from_df

stock_dfs = StockAcquisitionFacade(filter=universe_filter).get_stock_data_dict()

sector_index, _ = SectorIndex(stock_dfs, sector_redef_path, sector_index_path).calc_sector_index()

return_timeseries = StockReturnTimeseries(original_timeseries = sector_index, date_column = 'Date', sector_column = 'Sector')
return_timeseries.calculate(method=IntradayReturn())

ppp = PreprocessingPipeline(steps = [
    ('remove_pc1', PCAHandler(n_components=1, mode='components', fit_duration=train_duration))
    ])
return_timeseries.preprocess(pipeline = ppp)

raw_return_df = return_timeseries.raw_return
pc1_removed_intraday_return_df = return_timeseries.processed_return # 
print(pc1_removed_intraday_return_df) # TODO 完成したので、既存のTargetCalculatorを置き換える！！