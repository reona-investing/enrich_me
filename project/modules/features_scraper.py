import paths
import asyncio
import nodriver as uc
import os
from bs4 import BeautifulSoup as soup
import re
import json
from datetime import datetime, timedelta
import pandas as pd
import pytz
from typing import Tuple
import time
from io import StringIO

def _get_features_to_scrape_df() -> pd.DataFrame:
    features_to_scrape_df = pd.read_csv(paths.FEATURES_TO_SCRAPE_CSV)
    features_to_scrape_df['Path'] = paths.SCRAPED_DATA_FOLDER + '/' + \
                                    features_to_scrape_df['Group'] + '/' + \
                                    features_to_scrape_df['Path']
    return features_to_scrape_df

async def _scrape_from_ARCA(code:str,  browser:object) -> Tuple[str, datetime]:
    '''ARCAからのスクレイピング'''
    url = f'https://www.nyse.com/quote/index/{code}'
    tab = await browser.get(url)
    await tab.wait(10)
    html = await tab.get_content()

    s = soup(html, 'html.parser')
    #価格
    value = s.find_all('span', class_='d-dquote-x3')[0].text
    #日時
    s_time = s.find_all('div', class_='d-dquote-time')[0]
    time_elem = s_time.find_all('span')[1].text
    date_info = time_elem[1:11]
    date_info = datetime.strptime(date_info, '%m/%d/%Y')

    return value, date_info

async def _scrape_from_investing(url:str, browser:uc.core.browser.Browser) -> pd.DataFrame:
    '''investingからのスクレイピング'''
    for i in range(10):
        try:
            if i == 0:
                tab = await browser.tabs[0].get(url)
            else:
                print('reloading...')
                await tab.reload()
            _ = await tab.wait_for(text='時間枠')
            html = await tab.get_content()
            dfs = pd.read_html(StringIO(html))
            if len(dfs) < 2:
                continue
            else:
                for df in dfs:
                    if df.columns[0] == '日付け':
                        df_to_add = df
                        break
                if df_to_add is not None:
                    break
        except:
            continue
    else:
        raise ValueError(f'DataFrame is not found: {url}')
    
    df_to_add = df_to_add.iloc[:, :5]
    df_to_add.columns = ['Date', 'Close', 'Open', 'High', 'Low']

    try:
        df_to_add['Date'] = pd.to_datetime(df_to_add['Date'], format='%Y年%m月%d日')
    except:
        df_to_add['Date'] = pd.to_datetime(df_to_add['Date'], format='%m月 %d, %Y')  # datetime型に変換

    return df_to_add

async def scrape_all_indices(should_scrape_features:bool = True) -> pd.DataFrame:

    features_to_scrape_df = _get_features_to_scrape_df()

    if should_scrape_features:
        browser = await uc.start(browser_executable_path='C:\Program Files\Google\Chrome\Application\chrome.exe')

    for _, row in features_to_scrape_df.iterrows():
        start_time = time.time()
        print(row['Name'])
        #既存のデータフレームを読み込み
        df = pd.read_parquet(row['Path'])
        #必要な場合のみ新規データをスクレイピング
        if should_scrape_features:
            url = 'https://jp.investing.com/' + str(row['URL']) + '-historical-data'
            df_to_add = await _scrape_from_investing(url, browser)

            #バルチック海運指数はinvestingでは遅れ配信なので、当日分のみ公式よりスクレイピング
            #更新時刻：イギリス時間で13時
            if row['Name'] == 'BalticDry':
                tab = await browser.get('https://www.balticexchange.com/en/index.html')
                await tab.wait(1)
                element = await tab.wait_for('#ticker > div > div > div:nth-child(1) > span.value')
                value = float(element.text.replace(',', ''))
                UK_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('Europe/London')) #現在のイギリス時間を取得
                if UK_time.hour >= 13: #時刻が13時を過ぎているか判定
                    latest_day = UK_time.date() #当日の日付
                else:
                    latest_datetime = UK_time - timedelta(days=1)
                    latest_day = latest_datetime.date() #前日の日付
                df_to_add2 = pd.DataFrame({'Date':latest_day, 'Open':value, 'Close':value, 'High':value, 'Low':value}, index=[0])
                df_to_add = pd.concat([df_to_add, df_to_add2], ignore_index=True) #行を追加（後で重複行は削除）
            #IronOREも遅れ配信なので、当日分のみtradingviewからスクレイピング
            if row['Name'] == 'IronORE62':
                tab = await browser.get('https://www.tradingview.com/symbols/COMEX-TIO1!/')
                await tab.wait(5)
                element = await tab.wait_for('#js-category-content > div.tv-react-category-header > div.js-symbol-page-header-root > div > div > div > div.quotesRow-pAUXADuj > div:nth-child(1) > div > div.lastContainer-JWoJqCpY > span.last-JWoJqCpY.js-symbol-last > span')
                value = float(element.text)
                chicago_time = datetime.now().astimezone(pytz.utc).astimezone(pytz.timezone('America/Chicago')) #現在のイギリス時間を取得
                if chicago_time.hour >= 8: #時刻が8時を過ぎているか判定
                    latest_day = chicago_time.date() #当日の日付
                    df_to_add2 = pd.DataFrame({'Date':latest_day, 'Open':value, 'Close':value, 'High':value, 'Low':value}, index=[0])
                    df_to_add = pd.concat([df_to_add, df_to_add2], ignore_index=True) #行を追加（後で重複行は削除）
            #ARCA Grobal Airlinesはinvestingでは遅れ配信なので、当日分のみ公式Webよりスクレイピング
            if 'ARCA' in row['Name']:
                if row['Name'] == 'ARCA GlobalAirline':
                    value, latest_day = await _scrape_from_ARCA('AXGAL', browser) #NYSEのWebサイトからスクレイピング
                if row['Name'] == 'ARCA China':
                    value, latest_day = await _scrape_from_ARCA('CZH', browser) #NYSEのWebサイトからスクレイピング
                latest_day = latest_day.date() #前日の日付
                df_to_add2 = pd.DataFrame({'Date':latest_day, 'Open':value, 'Close':value, 'High':value, 'Low':value}, index=[0])
                df_to_add = pd.concat([df_to_add, df_to_add2], ignore_index=True) #行を追加（後で重複行は削除）

            #2つのデータフレームを結合
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df_to_add.columns = df.columns[:len(df_to_add.columns)] #2つのデータフレームのカラム名を揃える
            df_to_add['Date'] = pd.to_datetime(df_to_add['Date'])  # 日付をdatetime型に変換
            df = df.loc[~df['Date'].isin(df_to_add['Date'].unique())] #df_to_addにない（結合したい）列のみ抽出。
            df = pd.concat([df, df_to_add]) #元のデータフレームに新規データを結合


        #欠損値削除・型変換・重複削除
        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df.sort_values(by='Date').reset_index(drop=True)
        df[['Open', 'Close', 'High', 'Low']] = df[['Open', 'Close', 'High', 'Low']].replace(',', '', regex=True).astype(float)
        df = df.drop_duplicates(subset=['Date'], keep='last')

        #データフレームの出力
        print(df.tail(2))
        df.to_parquet(row['Path'])
        print(time.time() - start_time)
        print('-----------------------------------------------------')
    print('全データのスクレイピング完了')
    print('-----------------------------------------------------')

    return df
    
if __name__ == '__main__':
    features_to_scrape_df = _get_features_to_scrape_df()
    df = uc.loop().run_until_complete(scrape_all_indices())
        