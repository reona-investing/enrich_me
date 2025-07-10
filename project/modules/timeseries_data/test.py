from utils.paths import Paths
from acquisition.jquants_api_operations import StockAcquisitionFacade
from calculation import SectorIndex


from timeseries_data.public import StockReturnTimeseries
from timeseries_data.calculation_method import IntradayReturn # リターンの種類をクラスとして独立（OvernightReturn, DailyReturnも定義）
# from yyy import RemovingPC #前処理クラス群を定義（コンストラクタの引数にデータフレームを取り，calculateメソッドで同じ形のデータフレームを返す）

universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400')|(ScaleCategory=='TOPIX Small 1'))"
sector_redef_path = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/56sectors_2024-2025.csv"

print('a')

stock_dfs = StockAcquisitionFacade(filter=universe_filter).get_stock_data_dict()

print(len(stock_dfs))

sector_index, _ = SectorIndex(stock_dfs, sector_redef_path, None).calc_sector_index()

print(sector_index)

return_timeseries = StockReturnTimeseries(original_timeseries = sector_index, date_column = 'Date', sector_column = 'Sector')
return_timeseries.calculate(method=IntradayReturn(), open_column = 'Open', close_column = 'Close')

#remove_pc1 = RemovingPC(components=1)
#return_timeseries.preprocess(pipeline = [remove_pc1])

raw_intraday_return_df = return_timeseries.raw_return # プロパティとして定義しておく．
pc1_removed_intraday_return_df = return_timeseries.processed_return # プロパティとして定義しておく．
print(raw_intraday_return_df)