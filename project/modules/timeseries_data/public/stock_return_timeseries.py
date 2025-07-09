import pandas as pd


class StockReturnTimeseries:
    def __init__(self, return_timeseries: pd.DataFrame, date_column: str = 'Date', sector_column:str = 'Sector'):
        self.return_timeseries = return_timeseries
        self.date_column = date_column
        self.sector_column = sector_column

    def calculate(method):
        pass
    
    def evaluate():
        pass



# -------------------------------
# price_dfの作成コードは省略
# -------------------------------

return_timeseries = StockReturnTimeseries(timeseries_return = price_df, date_column = 'Date', sector_column = 'Sector')
return_timeseries.calculate(method=IntradayReturn(), open_column = 'Open', close_column = 'Close')

remove_pc1 = RemovingPC(components=1)
return_timeseries.preprocess(pipeline = [remove_pc1])

raw_intraday_return_df = return_timeseries.raw_return # プロパティとして定義しておく．
pc1_removed_intraday_return_df = return_timeseries.processed_return # プロパティとして定義しておく．