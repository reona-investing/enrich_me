from xxx.aaa import StockReturnTimeseries
from xxx.bbb import IntradayReturn # リターンの種類をクラスとして独立（OvernightReturn, DailyReturnも定義）
from yyy import RemovingPC #前処理クラス群を定義（コンストラクタの引数にデータフレームを取り，calculateメソッドで同じ形のデータフレームを返す）


# -------------------------------
# price_dfの作成コードは省略
# -------------------------------

return_timeseries = StockReturnTimeseries(timeseries_return = price_df, date_column = 'Date', sector_column = 'Sector')
return_timeseries.calculate(method=IntradayReturn(), open_column = 'Open', close_column = 'Close')

remove_pc1 = RemovingPC(components=1)
return_timeseries.preprocess(pipeline = [remove_pc1])

raw_intraday_return_df = return_timeseries.raw_return # プロパティとして定義しておく．
pc1_removed_intraday_return_df = return_timeseries.processed_return # プロパティとして定義しておく．