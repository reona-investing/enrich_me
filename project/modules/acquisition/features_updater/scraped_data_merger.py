from utils.paths import Paths
from utils.browser import BrowserUtils
import asyncio
from bs4 import BeautifulSoup as soup
from datetime import datetime, timedelta
import pandas as pd
import pytz
from typing import Tuple
import time

from utils.timekeeper import timekeeper

class ScrapedDataMerger:
    def __init__(self):
        pass

    
    def merge_scraped_data(self, existing_df: pd.DataFrame, additional_df: pd.DataFrame):
        df = self._merge_data(existing_df = existing_df, additional_df = additional_df)
        df = self._format_merged_df(df)
        return df


    def _merge_data(self, existing_df: pd.DataFrame, additional_df: pd.DataFrame):
        #2つのデータフレームを結合
        existing_df['Date'] = pd.to_datetime(existing_df['Date'], format='%Y-%m-%d')
        additional_df['Date'] = pd.to_datetime(additional_df['Date'], format='%Y-%m-%d')

        existing_df = existing_df.loc[~existing_df['Date'].isin(additional_df['Date'].unique())] #df_to_addにない（結合したい）列のみ抽出。
        return pd.concat([existing_df, additional_df])


    def _format_merged_df(self, df: pd.DataFrame) -> pd.DataFrame:
        #欠損値削除・型変換・重複削除
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df.sort_values(by='Date').reset_index(drop=True)
        df[['Open', 'Close', 'High', 'Low']] = df[['Open', 'Close', 'High', 'Low']].replace(',', '', regex=True).astype(float)
        return df.drop_duplicates(subset=['Date'], keep='last')



    
if __name__ == '__main__':
    from utils.paths import Paths
    existing_df = pd.read_parquet(f'{Paths.SCRAPED_DATA_FOLDER}/currency/raw_USDJPY_price.parquet')
    additional_df = pd.read_csv('mock.csv')
    sdm = ScrapedDataMerger()
    df = sdm.merge_scraped_data(existing_df, additional_df)
    print(df)
        