#%% モジュールのインポート
import paths
import data_pickler

from datetime import datetime
import os
import pandas as pd

#%% 関数群
def filter_stocks(stock_dfs_dict:dict, filter:str=None, filtered_code_list:list=None): # 対象銘柄の抜き取り
    '''
    対象銘柄を抜き取ります。
    stock_dfs_dict: 銘柄一覧、価格情報、財務情報を格納した辞書
    filter: 銘柄の絞り込み条件をstr型で指定
    filtered_code_list: 絞り込み対象の銘柄をリスト型で指定
    filter と filtered_code_listはどちらかを入力する。
    どちらも入力されている場合、filterが優先される
    '''
    stock_list = stock_dfs_dict['stock_list']
    stock_price = stock_dfs_dict['stock_price']
    stock_fin = stock_dfs_dict['stock_fin']
    #フィルターの設定
    if filter is not None:
        filtered_code_list = stock_list.query(filter)['Code'].astype(str).values
    #dfからの抜き取り
    stock_dfs_dict['stock_list'] = stock_list[stock_list['Code'].astype(str).isin(filtered_code_list)]
    stock_dfs_dict['stock_fin'] = stock_fin[stock_fin['Code'].astype(str).isin(filtered_code_list)]
    stock_dfs_dict['stock_price'] = stock_price[stock_price['Code'].astype(str).isin(filtered_code_list)]
    return stock_dfs_dict

def read_stock_price(end_date:datetime = datetime.today()) ->pd.DataFrame: # 価格情報の読み込み
    '''stock_priceの読み込み'''
    stock_prices = []
    for my_year in range(2013, end_date.year + 1):
        if os.path.exists(paths.STOCK_PRICE_PARQUET.replace('0000', str(my_year))):
            stock_price = pd.read_parquet(paths.STOCK_PRICE_PARQUET.replace('0000', str(my_year)))
            stock_price = stock_price[['Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume',
                                        'AdjustmentFactor', 'CumulativeAdjustmentFactor', 'TurnoverValue']]
            stock_prices.append(stock_price)
    stock_prices = pd.concat(stock_prices, axis=0)
    stock_prices = stock_prices.drop_duplicates(subset=['Date', 'Code'], keep='last')
    #end_dateまでのデータでdfを再構成。累積係数も再計算。
    stock_prices = stock_prices[stock_prices['Date']<=end_date]
    adjustment_factors = stock_prices.groupby('Code')['CumulativeAdjustmentFactor'].transform('last')
    stock_prices[['Open', 'High', 'Low', 'Close', 'Volume']] *= adjustment_factors.values[:, None]

    return stock_prices

def read_stock_dfs(filter:str=None, filtered_code_list:list=None, end_date:datetime = datetime.today()) -> dict: # 一括読み込み
    print('stock_dfデータ群の読み込みを開始します。')
    stock_list = pd.read_parquet(paths.STOCK_LIST_PARQUET)
    print('銘柄一覧の読み込みが完了しました。')
    stock_price = read_stock_price(end_date)
    print('価格情報の読み込みが完了しました。')
    stock_fin = pd.read_parquet(paths.STOCK_FIN_PARQUET)
    print('財務情報の読み込みが完了しました。')
    print('stock_dfデータ群の読み込みが完了しました。')
    print('----------------------------------------------')
    stock_dfs_dict = {'stock_list':stock_list, 'stock_price':stock_price, 'stock_fin':stock_fin}
    if filtered_code_list is not None:
        stock_dfs_dict = filter_stocks(stock_dfs_dict, filtered_code_list=filtered_code_list)
    if filter is not None:
        stock_dfs_dict = filter_stocks(stock_dfs_dict, filter=filter)
    return stock_dfs_dict

#%% デバッグ
if __name__ == '__main__':
    from IPython.display import display
    stock_dfs_dict = read_stock_dfs(end_date=datetime(2024,4,2))
    display(stock_dfs_dict['stock_price'])