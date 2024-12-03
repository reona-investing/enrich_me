#%% モジュールのインポート
import paths
import file_utilities
import error_handler

import shutil
import pandas as pd
import time
from bs4 import BeautifulSoup as soup
import csv
import nodriver as uc
import re
import unicodedata
from datetime import datetime
from typing import Tuple
from pathlib import Path
from IPython.display import display
from pathlib import Path
import asyncio

import os
from dotenv import load_dotenv
load_dotenv()

#%% 取引パラメータの格納
def _get_order_param_dicts() -> dict:
    order_param_dicts = {
                        '取引':{
                          "現物買": "genK",
                          "現物売": "genU",
                          "信用新規買": "shinK",
                          "信用新規売": "shinU",
                          },
                        '注文タイプ':{
                          "指値": 0,
                          "成行": 1,
                          "逆指値": 2,
                          },
                        '成行タイプ':{
                          "寄成":'Y',
                          "引成":'H',
                          "IOC成":'I'
                          },
                        '逆指値タイプ':{
                          "指値": 1,
                          "成行": 2,
                          },
                        '期間':{
                          "当日中": 0,
                          "今週中": 1,
                          "期間指定": 2
                          },
                        '預り区分':{
                          "一般預り": 0,
                          "特定預り": 1,
                          "NISA預り": 2
                          },
                        '信用取引区分':{
                          "制度": 0,
                          "一般": 1,
                          "日計り": 2
                          }
                        }
    return order_param_dicts

# リトライを設定するためのデコレーター
def _retry(max_attempts: int = 3, delay: float = 3.0):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    print(f"エラーが発生しました: {e}. リトライ中... (試行回数: {attempts})")
            print(f"{func.__name__}は最大試行回数に達しました。")
        return wrapper
    return decorator

#%% サブ関数
async def select_pulldown(tab: uc.core.tab.Tab, css_selector:str) -> uc.core.tab.Tab:
    # オプションをJavaScriptを使用してクリック
    await tab.evaluate(f'''
        var option = document.querySelector('{css_selector}');
        option.selected = true;
        var event = new Event('change', {{ bubbles: true }});
        option.parentElement.dispatchEvent(event);
    ''')
    await tab.wait(1)  # 更新を待つ
    return tab

async def _input_sashinari_params(tab:uc.core.tab.Tab, order_param_dicts:dict,
                            order_type:str, nariyuki_type:str, limit_order_price:float,
                            stop_order_type:str, stop_order_trigger_price:float, stop_order_price:float):
    '''指値・成行関係のパラメータを入力'''
    button = await tab.find(order_type)
    await button.click() # 注文タイプ
    if nariyuki_type is not None:
        selector = f'select[name="nariyuki_condition"] option[value="{order_param_dicts["成行タイプ"][nariyuki_type]}"]'
        tab = await select_pulldown(tab, selector)

    if order_type == "指値":
        if limit_order_price is None:
            raise ValueError("指値価格を設定してください。limit_order_price")
        else:
            form = await tab.select('#gsn0 > input[type=text]')
            await form.send_keys(limit_order_price) # 指値価格

    if order_type == "逆指値":
        choice = await tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > label:nth-child(5) > input[type=radio]')
        await choice.click()
        # 逆指値トリガー価格
        if stop_order_trigger_price is None:
            raise ValueError("逆指値のトリガー価格を設定してください。stop_order_trigger_price")
        else:
            await tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6)')
            choice = await tab.select(f'#gsn2 > table > tbody > tr > td:nth-child(2) > select:nth-child(6) > option:nth-child({order_param_dicts["逆指値タイプ"][stop_order_type]})')
            await choice.send_keys(stop_order_trigger_price)

        # 逆指値価格
        if stop_order_type == "指値":
            if stop_order_price is None:
                raise ValueError("逆指値価格を設定してください。stop_order_type")
            else:
                form = await tab.select('#gsn2 > table > tbody > tr > td:nth-child(2) > input[type=text]:nth-child(7)')
                await form.send_keys(stop_order_price)

async def _get_duration_params(tab:uc.core.tab.Tab, order_param_dicts:dict,
                         period_type:str, period_value:str, period_index:int):
    '''期間指定関連のパラメータ'''
    button = await tab.find(period_type)
    await button.click() # 期間タイプ
    
    # 期間指定
    if period_type=="期間指定":
        if period_value is None and period_index is None:
            raise ValueError("期間を指定してください。period_value or period_index")
        else:
            # 期間指定の要素を取得
            period_option_div = await tab.select('select[name="limit_in"]')

            # 期間を設定
            if period_value is not None:
                # すべてのオプションを取得
                options = await period_option_div.select('option')
                # オプションのvalue属性をリストに格納
                period_value_list = [await option.get_attribute('value') for option in options]
                
                if period_value not in period_value_list:
                    raise ValueError("期間の値が存在しません。period_value\n指定可能日: {}".format(",".join(period_value_list)))
                else:
                    # value属性が一致するオプションを選択
                    selector = f'option[value="{period_value}"]'
                    tab = await select_pulldown(tab, selector)
                    
            if period_index is not None:
                # indexに基づいてオプションを選択
                options = await period_option_div.select('option')
                if period_index < len(options):
                    await options[period_index].click()
                else:
                    raise ValueError("指定されたインデックスがオプションの範囲外です。")

def _append_order_to_list(tr:object, data:list) -> list:
    # 行のデータを格納する変数を定義
    row = []
    row.append(tr.findAll("td")[0].getText().strip()) # 注文番号
    row.append(tr.findAll("td")[1].getText().strip()) # 注文状況
    row.append(tr.findAll("td")[2].getText().strip()) # 注文種別
    # 銘柄、コード 同じtdにまとまっているので分割してそれぞれのデータを取得する
    text = unicodedata.normalize("NFKC", tr.findAll("td")[3].getText().strip())
    row.append(text.splitlines()[0].strip().split(" ")[0])
    row.append(text.splitlines()[0].strip().split(" ")[-1])
    """
    ここからは次行のデータを取得
    """
    #取引、預り、手数料　同じtd内にまとまっているので分割してそれぞれのデータを取得する
    tmp_data = []
    for t in tr.findNext("tr").findAll("td")[0].getText().strip().replace(" ", "").splitlines():
        if t!="" and t!="/":
            tmp_data.append(t)
    # 信用取引の場合は取得される要素数が増えるので、要素数によって処理を分ける
    if len(tmp_data)==4:
        row.extend([tmp_data[0]+tmp_data[1], tmp_data[2], tmp_data[3]])
    else:
        row.extend(tmp_data)
    # 注文日、注文期間　同じtd内にまとまっているので分割してそれぞれのデータを取得する
    row.extend(tr.findNext("tr").findAll("td")[1].getText().replace(" ", "").strip().splitlines())
    # 注文株数、（未約定）　同じtd内にまとまっているので分割してそれぞれのデータを取得する
    row.append(tr.findNext("tr").findAll("td")[2].getText().replace(" ", "").strip().splitlines()[0])
    row.append(tr.findNext("tr").findAll("td")[2].getText().replace(" ", "").strip().splitlines()[-1])
    # 執行条件
    
    row.append(tr.findNext("tr").findAll("td")[3].getText().strip())
    # 注文単価、現在値
    row.extend(tr.findNext("tr").findAll("td")[4].getText().strip().replace(" ", "").splitlines())
    """
    ここからは2行後のデータから条件を取得　該当データがない場合は--とする
    """
    # 条件
    if not tr.findNext("tr").findNext("tr").find("td").find("a"):
        row.append(tr.findNext("tr").findNext("tr").find("td").getText().strip())
    else:
        row.append("--")

    # dataに行データを追加
    data.append(row)

    return data

def _append_margin_to_list(tr:object, data:list) -> list:
    # 行のデータを格納する変数を定義
    row = []
    # 銘柄、コード 同じtdにまとまっているので分割してそれぞれのデータを取得する
    text = unicodedata.normalize("NFKC", tr.findAll("td")[0].getText().strip())
    row.append(text[-4:]) #コード
    row.append(text[:-4]) #銘柄
    row.append(tr.findAll("td")[1].getText().strip()) # 売・買建
    # 建株数
    text = unicodedata.normalize("NFKC", tr.findAll("td")[5].getText().strip())
    row.append(text.splitlines()[0].strip().split(" ")[0]) #建株数
    # 建単価、現在値 同じtdにまとまっているので分割してそれぞれのデータを取得する
    text = unicodedata.normalize("NFKC", tr.findAll("td")[6].getText().strip())
    numbers = text.split("\n")
    row.append(numbers[0]) #建単価
    row.append(numbers[1]) #現在値
      # dataに行データを追加
    data.append(row)
    return data

def _append_contract_to_list(tr:object, data:list) -> list:
    # 行のデータを格納する変数を定義
    row = []
    #日付
    text = unicodedata.normalize("NFKC", tr.findAll("td")[2].getText().strip())
    row.append(datetime(2000+int(text[0:2]), int(text[3:5]), int(text[6:8])))
    # 売・買建
    text = unicodedata.normalize("NFKC", tr.findAll("td")[1].getText().strip())
    text = text[:3]
    if text == '信新売':
        row.append('売')
    elif text == '信新買':
        row.append('買')
    # 銘柄、コード 同じtdにまとまっているので分割してそれぞれのデータを取得する
    text = unicodedata.normalize("NFKC", tr.findAll("td")[0].getText().strip())
    row.append(text[-4:]) #コード
    row.append(text[:-4]) #銘柄
    row.append(tr.findAll("td")[3].getText().replace(",", "").strip()) #株数
    row.append(tr.findAll("td")[4].getText().replace(",", "").strip()) #約定単価
    row.append(tr.findNext("tr").findAll("td")[3].getText().replace(",", "").strip()) #決済単価
    data.append(row) #dataに行データを追加
    return data

def _append_spot_to_list(tr:object, data:list) -> list:
        # 行のデータを格納する変数を定義
        row = []
        # 日付
        text = unicodedata.normalize("NFKC", tr.findAll("td")[2].getText().strip())
        row.append(datetime(2000+int(text[0:2]), int(text[3:5]), int(text[6:8]))) #日付
        # 売・買建
        text = unicodedata.normalize("NFKC", tr.findAll("td")[1].getText().strip())
        text = text[:3]
        if text == '現物売':
            row.append('売')
        elif text == '現物買':
            row.append('買')
        # 銘柄、コード 同じtdにまとまっているので分割してそれぞれのデータを取得する
        text = unicodedata.normalize("NFKC", tr.findAll("td")[0].getText().strip())
        row.append(text[-4:]) #コード
        row.append(text[:-4]) #銘柄
        #受渡金額/決済損益
        text = unicodedata.normalize("NFKC", tr.findAll("td")[7].getText().replace(",", "").strip())
        row.append(text[:text.index("(")])
        data.append(row)
        return data

def _convert_margin_csv_to_df(margin_csv:str) -> pd.DataFrame:
    #元のcsvから必要行のみを切り出してデータフレーム化
    extracted_rows = []
    found_code = False
    if os.name == 'posix':
        with open(margin_csv, newline='', encoding='shift_jis') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not found_code:
                    if len(row) > 0 and row[0] == "コード":
                        found_code = True
                        columns = row
                else:
                    extracted_rows.append(row)
    else:
        with open(margin_csv, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not found_code:
                    if len(row) > 0 and row[0] == "コード":
                        found_code = True
                        columns = row
                else:
                    extracted_rows.append(row)
    margin_df = pd.DataFrame(extracted_rows, columns=columns)
    return margin_df

def _format_margin_df(margin_df:pd.DataFrame, sector_list_df:pd.DataFrame) -> pd.DataFrame:
    '''
    信用取引結果データフレームを成型する。
    '''
    #データ型を整える
    margin_df['銘柄コード'] = margin_df['銘柄コード'].astype(str)
    margin_df['株数'] = margin_df['株数'].astype(int)
    margin_df['取得単価'] = margin_df['取得単価'].astype(float)
    margin_df['決済単価'] = margin_df['決済単価'].astype(float)

    #必要な数値を計算する
    margin_df['取得価格'] = (margin_df['取得単価'] * margin_df['株数']).astype(int)
    margin_df['決済価格'] = (margin_df['決済単価'] * margin_df['株数']).astype(int)
    margin_df['手数料'] = 0
    margin_df['利益（税引前）'] = 0
    margin_df.loc[margin_df['売or買']=='買', '利益（税引前）'] = margin_df['決済価格'] - margin_df['取得価格'] - margin_df['手数料']
    margin_df.loc[margin_df['売or買']=='売', '利益（税引前）'] = margin_df['取得価格'] - margin_df['決済価格'] - margin_df['手数料']
    margin_df['利率（税引前）'] = margin_df['利益（税引前）'] / margin_df['取得価格']

    #sector_list_dfの型変換
    sector_list_df['Code'] = sector_list_df['Code'].astype(str)

    #業種一覧と結合
    margin_df = pd.merge(margin_df, sector_list_df[['Code', 'Sector']], left_on='銘柄コード', right_on='Code', how='left')
    margin_df = margin_df.drop('Code', axis=1).rename(columns={'Sector':'業種'})
    margin_df = margin_df[['日付', '売or買', '業種', '銘柄コード', '社名', '株数', '取得単価', '決済単価', '取得価格', '決済価格', '手数料', '利益（税引前）', '利率（税引前）']]

    return margin_df

def _get_deal_history_df(margin_df:pd.DataFrame, sector_list_df:pd.DataFrame) -> pd.DataFrame:
    '''
    '''
    margin_df = margin_df[['約定日', '取引', '銘柄コード', '銘柄', '約定数量', '約定単価', '手数料/諸経費等']].copy()
    margin_df = margin_df.rename(columns={'約定日':'日付', '銘柄':'社名', '約定数量':'株数', '手数料/諸経費等':'手数料'})
    margin_df['取得単価'] = 0
    margin_df.loc[margin_df['取引'].isin(['信用新規買', '信用新規売']), '取得単価'] = margin_df['約定単価']
    margin_df['決済単価'] = 0
    margin_df.loc[margin_df['取引'].isin(['信用返済買', '信用返済売']), '決済単価'] = margin_df['約定単価']
    margin_df['売or買'] = '売'
    margin_df.loc[margin_df['取引'].isin(['信用新規買', '信用返済売']), '売or買'] = '買'
    margin_df = margin_df.drop(['取引', '約定単価'], axis=1)
    margin_df[['取得単価', '決済単価']] = margin_df[['取得単価', '決済単価']].astype(float)
    margin_df[['株数', '手数料']] = margin_df[['株数', '手数料']].astype(int)
    margin_df = margin_df.groupby(['日付', '銘柄コード', '社名', '株数', '売or買']).sum().reset_index(drop=False)

    #データを成型して最終的なデータフレームを得る。
    margin_df = _format_margin_df(margin_df, sector_list_df)

    return margin_df

async def _extract_order_list(tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''注文リストの抽出'''
    # 注文ページへ遷移
    tab = await sign_in(tab)
    trade_button = await tab.wait_for('img[title="取引"]')
    await trade_button.click()
    # 注文照会、取り消し・訂正ページへ遷移
    button = await tab.wait_for(text='注文照会')
    await button.click()
    await tab.wait(3)
    # htmlを取得
    html_content = await tab.get_content()
    html = soup(html_content, "html.parser")
    # tableを抽出, 「注文状況」というカラムを検索して、そのtableを取得
    table = html.find("th", string=re.compile("注文状況"))

    if table is None:
        print('発注中の注文はありません。')
        return tab, pd.DataFrame()
    table = table.findParent("table")

    data = [] # データを格納する変数を定義
    # tableの各行を処理する
    for tr in table.find("tbody").findAll("tr"): #tbodyの全行（tr）を取得
        #最初のtdにa要素があれば注文番号があると判断する
        if tr.find("td").find("a"):
            data = _append_order_to_list(tr, data)

    # カラム名を定義
    columns = ["注文番号", "注文状況", "注文種別", "銘柄", "コード", "取引", "預り", "手数料", "注文日",
              "注文期間", "注文株数", "（未約定）", "執行条件", "注文単価", "現在値", "条件"]
    # DataFrameに変換
    order_list_df = pd.DataFrame(data, columns=columns)
    # 型変換
    order_list_df["注文番号"] = order_list_df["注文番号"].astype(int)
    order_list_df["コード"] = order_list_df["コード"].astype(str)
    # 注文状況が取消中の行を削除してindexをリセット
    order_list_df = order_list_df[order_list_df["注文状況"]!="取消中"].reset_index(drop=True)

    return tab, order_list_df

async def _extract_margin_list(tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''信用建玉リストの抽出'''
    # 信用建玉ページへ遷移
    tab = await sign_in(tab)
    button = await tab.wait_for('img[title=口座管理]')
    await button.click()
    button = await tab.wait_for('area[title=信用建玉]')
    await button.click()
    await tab.wait(3)
    # htmlを取得
    html_content = await tab.get_content()
    html = soup(html_content, "html.parser")
    # tableを抽出, 「銘柄」というカラムを検索して、そのtableを取得
    table = html.find("td", string=re.compile("銘柄"))

    if table is None:
        print('保有建玉はありません。')
        return tab, pd.DataFrame()
    table = table.findParent("table")

    # データを格納する変数を定義
    data = []
    # tableの各行を処理する
    for tr in table.find("tbody").findAll("tr"): #tbodyの全行（tr）を取得
        #最初のtdにa要素があれば信用建玉があると判断する
        if tr.find("td").find("a"):
            data = _append_margin_to_list(tr, data)
    # カラム名を定義
    columns = ["証券コード", "銘柄", "売・買建", "建株数", "建単価", "現在値"]
    # DataFrameに変換
    margin_list_df = pd.DataFrame(data, columns=columns)
    # 型変換
    margin_list_df["証券コード"] = margin_list_df["証券コード"].astype(str)
    margin_list_df["建株数"] = margin_list_df["建株数"].str.replace(',', '').astype(int)
    margin_list_df["建単価"] = margin_list_df["建単価"].str.replace(',', '').astype(float)
    margin_list_df["現在値"] = margin_list_df["現在値"].str.replace(',', '').astype(float)
    # 評価額
    margin_list_df["建価格"] = margin_list_df["建株数"] * margin_list_df["建単価"]
    margin_list_df["評価額"] = margin_list_df["建株数"] * margin_list_df["現在値"]
    #評価損益
    margin_list_df['評価損益'] = margin_list_df["評価額"] - margin_list_df["建価格"]
    margin_list_df.loc[margin_list_df['売・買建'] == '売建', '評価損益'] = margin_list_df["建価格"] - margin_list_df["評価額"]

    return tab, margin_list_df

#%% 
@_retry()
async def sign_in(tab:uc.core.tab.Tab=None) -> uc.core.tab.Tab:
    '''未サインインの場合のみサインイン操作を行う'''
    if tab is None:
        if os.name == 'nt':
            browser = await uc.start(browser_executable_path='C:\Program Files\Google\Chrome\Application\chrome.exe')
        else:
            browser = await uc.start(browser_executable_path='/opt/google/chrome/chrome', headless=True)
        
        #SBI証券のログインページにアクセス
        url = "https://www.sbisec.co.jp/ETGate"
        tab = await browser.get(url)
        await tab.wait(2)
        #ユーザーネームとパスワードを送信
        username = await tab.wait_for('input[name="user_id"]')
        await username.send_keys(os.getenv('SBI_USERNAME'))
        password = await tab.wait_for('input[name="user_password"]')
        await password.send_keys(os.getenv('SBI_LOGINPASS'))
        # ログインボタンをクリック
        login = await tab.wait_for('input[name="ACT_login"]')
        await login.click()
        await tab.wait(3)
    return tab

@_retry()
async def fetch_deal_history(tab:uc.core.tab.Tab=None, sector_list_df:pd.DataFrame=None, mydate:datetime=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''過去の信用約定情報を取得'''
    myyear = f'{mydate.year}'
    mymonth = f'{mydate.month:02}'
    myday = f'{mydate.day:02}'
    tab = await sign_in(tab)
    button = await tab.find('取引履歴')
    await button.click()
    await tab.wait(1)
    #「信用取引」をクリック
    button = await tab.select('#shinT')
    await button.click()
    #表示する日時を設定
    # 年の選択
    element_num = {'from_yyyy':myyear, 'from_mm':mymonth, 'from_dd':myday, 
                   'to_yyyy':myyear, 'to_mm':mymonth, 'to_dd':myday}
    for key, value in element_num.items():
        pulldown_selector = f'select[name="ref_{key}"] option[value="{value}"]'
        tab = await select_pulldown(tab, pulldown_selector)

    #「照会」をクリック
    button = await tab.find('照会')
    await button.click()
    await tab.wait(1)
    #CSVをダウンロード
    button = await tab.find('CSVダウンロード')
    await button.click()
    await tab.wait(3)

    #ダウンロードしたファイルのファイル名を取得。
    deal_history_csv = ""
    #ダウンロード完了まで待機（リトライ上限10回）
    for i in range(10):
        deal_history_csv, _ = file_utilities.get_newest_two_files(paths.DOWNLOAD_FOLDER)
        await tab.wait(1)
        if deal_history_csv.endswith('.csv'):
            break
    #元のcsvから必要行のみを切り出してデータフレーム化
    deal_history_df = pd.read_csv(deal_history_csv, header=None, skiprows=8)

    #データフレームの形を整える
    deal_history_df.columns = deal_history_df.iloc[0]
    deal_history_df = deal_history_df.iloc[1:]
    deal_history_df[['手数料/諸経費等', '税額', '受渡金額/決済損益']] = \
      deal_history_df[['手数料/諸経費等', '税額', '受渡金額/決済損益']].replace({'--':'0'}).astype(int)
    deal_history_df = _get_deal_history_df(deal_history_df, sector_list_df)
    deal_history_df['日付'] = pd.to_datetime(deal_history_df['日付']).dt.date

    os.remove(deal_history_csv)

    return tab, deal_history_df

@_retry()
async def fetch_in_out(tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''入出金明細ページへ遷移'''
    tab = await sign_in(tab)
    button = await tab.find('入出金明細')
    await button.click()
    await tab.wait(1)
    #検索期間を3ヶ月に指定
    button = await tab.find('3ヶ月')
    await button.click()
    await tab.wait(1)
    #表示件数を200件に設定
    pulldown_selector = 'select[name=in_v_list_count] option[value="200"]'
    tab = await select_pulldown(tab, pulldown_selector)
    button = await tab.select('img[title=照会]')
    await button.click()
    await tab.wait(1)
    # htmlを取得
    html_content = await tab.get_content()
    html = soup(html_content, "html.parser")
    # tableを抽出, 「入出金日」というカラムを検索して、そのtableを取得
    table = html.find("th", string=re.compile("区分"))

    if table is None:
        print('直近1週間の入出金履歴はありません。')
        return tab, None
    table = table.findParent("table")

    # データを格納する変数を定義
    data = []
    # tableの各行を処理する
    for tr in table.find("tbody").findAll("tr"): #tbodyの全行（tr）を取得
        row = [] # 行のデータを格納する変数を定義
        row.append(tr.findAll("td")[0].getText().strip()) # 日付
        row.append(tr.findAll("td")[2].getText().strip()) # 摘要
        row.append(tr.findAll("td")[3].getText().replace("-", "0").replace(",", "").strip()) # 出金額
        row.append(tr.findAll("td")[4].getText().replace("-", "0").replace(",", "").strip()) # 入金額
        row.append(tr.findAll("td")[5].getText().replace("-", "0").replace(",", "").strip()) # 振替出金額
        row.append(tr.findAll("td")[6].getText().replace("-", "0").replace(",", "").strip()) # 振替入金額
        data.append(row) #dataに行データを追加

    columns = ["日付", "摘要", "出金額", "入金額", "振替入金額", "振替出金額"] # カラム名を定義
    in_out_df = pd.DataFrame(data, columns=columns) # DataFrameに変換
    # データ型の設定
    in_out_df['日付'] = pd.to_datetime(in_out_df['日付']).dt.date
    for x in ['入金額', '出金額', '振替入金額', '振替出金額']:
        in_out_df[x] = in_out_df[x].astype(int)
    in_out_df['入出金額'] = in_out_df['入金額'] + in_out_df['振替入金額'] - in_out_df['出金額'] - in_out_df['振替出金額']
    in_out_df = in_out_df.loc[~in_out_df['摘要'].str.contains('譲渡益税')]
    in_out_df = in_out_df[['日付', '摘要', '入出金額']]
    print('入出金の履歴')
    display(in_out_df)

    return tab, in_out_df

@_retry()
async def fetch_today_contracts(tab:uc.core.tab.Tab=None, sector_list_df:pd.DataFrame=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''当日の信用約定情報を取得'''
    #当日約定一覧ページへ遷移
    tab = await sign_in(tab)
    button = await tab.select('img[title=口座管理]')
    await button.click()
    await tab.wait(1)
    button = await tab.find('当日約定一覧')
    await button.click()
    await tab.wait(1)
    button = await tab.find('国内株式(信用)')
    await button.click()
    await tab.wait(1)
    # htmlを取得
    html_content = await tab.get_content()
    html = soup(html_content, 'html.parser')
    # tableを抽出, 「銘柄」というカラムを検索して、そのtableを取得
    table = html.find("td", string=re.compile("銘柄"))
    if table is None:
        print('本日約定の注文はありません。')
        return tab, None
    table = table.findParent("table")

    # データを格納する変数を定義
    data = []
    # tableの各行を処理する
    for tr in table.find("tbody").findAll("tr"): #tbodyの全行（tr）を取得
        #最初のtdにa要素があれば銘柄名であると判断する
        if tr.find("td").find("a"):
            data = _append_contract_to_list(tr, data)

    # カラム名を定義
    columns = ["日付", "売or買", "銘柄コード", "社名", "株数", "取得単価", "決済単価"]
    # DataFrameに変換
    today_margin_df = pd.DataFrame(data, columns=columns)
    today_margin_df = today_margin_df[(today_margin_df['売or買']=='買')|(today_margin_df['売or買']=='売')]

    #データを成型して最終的なデータフレームを得る。
    today_margin_df = _format_margin_df(today_margin_df, sector_list_df)


    return tab, today_margin_df

@_retry()
async def fetch_today_spots(tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, pd.DataFrame]:
    '''当日の現物取引による増減を確認'''
    #当日約定一覧ページへ遷移
    tab = await sign_in(tab)
    button = await tab.select('img[title=取引]')
    await button.click()
    await tab.wait(1)
    button = await tab.find('当日約定一覧')
    await button.click()
    await tab.wait(1)
    button = await tab.find('国内株式(現物)')
    await button.click()
    await tab.wait(1)
    # htmlを取得
    html_content = await tab.get_content()
    html = soup(html_content, 'html.parser')
    # tableを抽出, 「銘柄」というカラムを検索して、そのtableを取得
    table = html.find('td', string=re.compile('銘柄'))

    if table is None:
        print('本日約定の注文はありません。')
        return tab, None
    table = table.findParent("table")

    # データを格納する変数を定義
    data = []
    # tableの各行を処理する
    for tr in table.find("tbody").findAll("tr"): #tbodyの全行（tr）を取得
        #最初のtdにa要素があれば銘柄名であると判断する
        if tr.find("td").find("a"):
            data = _append_spot_to_list(tr, data)

    # カラム名を定義
    columns = ["日付", "売or買", "銘柄コード", "社名", "買付余力増減"]

    # DataFrameに変換
    today_spots_df = pd.DataFrame(data, columns=columns)
    #データ型変換
    today_spots_df['日付'] = pd.to_datetime(today_spots_df['日付']).dt.date
    today_spots_df['銘柄コード'] = today_spots_df['銘柄コード'].astype(str)
    today_spots_df['買付余力増減'] = today_spots_df['買付余力増減'].astype(int)
    #"買"の場合はマイナスとする（資金が減るので）
    today_spots_df.loc[today_spots_df['売or買']=='買', '買付余力増減'] = - today_spots_df.loc[today_spots_df['売or買']=='買', '買付余力増減']
    print('現物売買')
    display(today_spots_df)

    return tab, today_spots_df

@_retry()
async def get_buying_power(tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, int, int]:
    '''信用建余力と買付余力(2営業日後)の取得'''
    # 口座管理ページへ遷移
    tab = await sign_in(tab)
    button = await tab.select('img[title="口座管理"]')
    await button.click()
    await tab.wait(3)  # 3秒待機

    # HTMLを取得
    html_content = await tab.get_content()
    html = soup(html_content, "html.parser")

    # 信用建余力をint型で取得
    div = html.find("div", string=re.compile("信用建余力"))
    margin_buying_power = div.findNext("div").getText().strip()
    margin_buying_power = int(margin_buying_power.replace(',', ''))  # 信用建余力をint型で取得

    # 買付余力(2営業日後)
    div = html.find("div", string=re.compile("買付余力\\(2営業日後\\)"))
    buying_power = div.findNext("div").getText().strip()
    buying_power = int(buying_power.replace(',', ''))  # 買付余力をint型で取得

    return tab, margin_buying_power, buying_power

@_retry()
async def get_trade_possibility(tab:uc.core.tab.Tab=None) -> Tuple[uc.core.tab.Tab, dict, dict]:
    '''日計り信用可能銘柄かを判定'''
    # 何らかのエラーで前回downloads.htmが残ってしまった場合に削除する処理
    filelist = os.listdir(paths.DOWNLOAD_FOLDER)
    if len(filelist) > 0:
        for file in filelist:
            os.remove(f'{paths.DOWNLOAD_FOLDER}/{file}')

    #国内株式トップページにアクセス
    tab = await sign_in(tab)
    elem = await tab.wait_for('#navi01P > ul > li:nth-child(3) > a')
    await elem.click()
    #一般信用売り銘柄一覧のページに移動
    elem = await tab.wait_for('#rightNav\ mt-8 > div:nth-child(1) > ul > li:nth-child(5) > a')
    await elem.click()
    await tab.wait(5)
    await tab.set_download_path(Path(paths.DOWNLOAD_FOLDER))
    #CSVのダウンロードリンクをクリック
    download_link = await tab.wait_for('#csvDownload')
    await download_link.click()
    #ダウンロード完了まで待機（リトライ上限10回）
    csv_path = None
    for i in range(10):
        print(i)
        await tab.wait(2)
        filelist = os.listdir(paths.DOWNLOAD_FOLDER)
        if len(filelist) > 0:
            for file in filelist:
                if file.endswith('.csv'):
                    print(file)
                    csv_path = f'{paths.DOWNLOAD_FOLDER}/{file}'
                    break
        if csv_path is not None:
            break
    #元のcsvから必要行のみを切り出してデータフレーム化
    margin_df = _convert_margin_csv_to_df(csv_path)
    #ダウンロードしたcsvを削除しておく。
    #for file in filelist:
    #    os.remove(f'{paths.DOWNLOAD_FOLDER}/{file}')

    #日計り信用売り可の銘柄と上限単位数を辞書で格納
    buy_possibility = {margin_df['コード'].astype(str): margin_df['一人あたり建玉上限数'].astype(int)}
    sell_possibility = {margin_df.loc[(margin_df['売建受注枠']!='受付不可') & (margin_df['信用区分（HYPER）']==''), 'コード'].astype(str):
                        margin_df.loc[(margin_df['売建受注枠']!='受付不可') & (margin_df['信用区分（HYPER）']==''), '一人あたり建玉上限数'].astype(int)}

    return tab, buy_possibility, sell_possibility

@_retry()
async def make_order(tab: uc.core.tab.Tab = None,
                     trade_type: str = "信用新規買", ticker: str = None, unit: int = 100, order_type: str = "成行", nariyuki_value: str = '寄成',
                     limit_order_price: float = None, stop_order_trigger_price: float = None, stop_order_type: str = "成行", stop_order_price: float = None,
                     period_type: str = "当日中", period_value: str = None, period_index: int = None, trade_section: str = "特定預り",
                     margin_trade_section: str = "制度"):
    '''注文を発注する'''
    # 購入注文処理を実装
    order_param_dicts = _get_order_param_dicts()

    # 証券コードをチェック
    if ticker is None:
        raise ValueError("証券コードを設定してください。ticker")

    # 注文ページへ遷移
    tab = await sign_in(tab)
    trade_button = await tab.wait_for('img[title="取引"]')
    await trade_button.click()
    await tab.wait(2)  # 2秒待機

    # 基本パラメータ
    trade_type_button = await tab.wait_for(f'#{order_param_dicts["取引"][trade_type]}')  # 取引タイプ
    await trade_type_button.click()
    stock_code_input = await tab.select('input[name="stock_sec_code"]')
    await stock_code_input.send_keys(ticker)  # 証券コード
    quantity_input = await tab.select('input[name="input_quantity"]')
    await quantity_input.send_keys(str(unit))  # 株数


    # 指値・成行関係のパラメータ設定
    await _input_sashinari_params(tab, order_param_dicts, order_type, nariyuki_value, limit_order_price,
                                  stop_order_type, stop_order_trigger_price, stop_order_price)

    # 期間指定関係のパラメータ設定
    await _get_duration_params(tab, order_param_dicts, period_type, period_value, period_index)

    # 基本パラメータ2
    deposit_type_button = await tab.find(trade_section)
    await deposit_type_button.click()  # 預り区分
    credit_trade_type_button = await tab.find(margin_trade_section)
    await credit_trade_type_button.click()  # 信用取引区分
    trade_password_input = await tab.select('input[id="pwd3"]')
    await trade_password_input.send_keys(os.getenv('SBI_TRADEPASS'))  # 取引パスワード

    # 注文画面を省略
    skip_button = await tab.select('input[id="shouryaku"]')
    await skip_button.click()
    order_button = await tab.select('img[title="注文発注"]')
    await order_button.click()
    await tab.wait(1)

    # 注文確認
    html_content = await tab.get_content()
    if "ご注文を受け付けました。" in html_content:
        print(f"{ticker} {unit}株 {trade_type} {order_type}：正常に注文完了しました。")
        has_successfully_ordered = True
    else:
        print(f"{ticker} {unit}株 {trade_type} {order_type}：発注できませんでした。")
        await tab.save_screenshot(f'{paths.DEBUG_FILES_FOLDER}/{ticker}_error.png') #debug用
        shutil.copy(f'{paths.DEBUG_FILES_FOLDER}/{ticker}_error.png', f'{paths.ONLINE_BACKUP_FOLDER}/{ticker}_error.png')
        has_successfully_ordered = False

    return tab, has_successfully_ordered

@_retry()
async def settle_all_margins(tab:uc.core.tab.Tab=None) -> tuple[uc.core.tab.Tab, list]:
    '''
    信用建玉の一括決済
    [返り値]
    tab: nodriverのタブ
    error_tickers: 決済注文できなかった銘柄のリスト
    '''
    retry = 0
    error_tickers = []


    tab = await sign_in(tab)
    tab, margin_list = await _extract_margin_list(tab)
    tab, order_list = await _extract_order_list(tab)

    if margin_list is None:
        print('信用建玉がありません。決済処理を中断します。')
        return
    else:
        margin_tickers = margin_list.sort_values(by="証券コード")['証券コード'].unique().tolist()
    if len(order_list) > 0:
        ordered_tickers = order_list.sort_values(by="コード")['コード'].unique().tolist()
        if sorted(margin_tickers) == sorted(ordered_tickers):
            print('すべての信用建玉の決済注文を発注済みです。')
            return None, error_tickers
    else:
        ordered_tickers = []
    print(f'保有建玉：{len(margin_tickers)}件')
    print(f'発注済み：{len(ordered_tickers)}件')
    i = n = 0
    while i < len(margin_tickers):
        if margin_tickers[i] in ordered_tickers:
            print(f'{margin_tickers[i]}はすでに決済発注済です。')
            i += 1
            continue
        else:
            # 決済注文処理を実装
            button = await tab.wait_for('img[title=取引]')
            await button.click()
            # 信用返済・現引現渡ページへ遷移
            button = await tab.wait_for(text='信用返済')
            await button.click()
            # 個別のポジションのページへ移動
            for _ in range(10):
                # 1銘柄で複数ポジションを持っている場合に備えて繰り返し処理
                try:
                    position_link = await tab.wait_for(
                        f'#MAINAREA02_780 > form > table:nth-child(18) > tbody > tr > td > \
                        table > tbody > tr > td > table > tbody > tr:nth-child({i +  n + 1}) > \
                        td:nth-child(10) > a:nth-child(1) > u > font', 
                        timeout = 5
                        )
                    await position_link.click()
                    break
                except:
                    await tab.save_screenshot('debug.png')
                    n += 1
            # "全株指定"ボタンをクリック
            all_shares_button = await tab.wait_for('input[value="全株指定"]')
            await all_shares_button.click()
            # "注文入力"ボタンをクリック
            order_input_button = await tab.wait_for('input[value="注文入力へ"]')
            await order_input_button.click()
            await tab.wait(1)
            # 注文条件の設定
            order_type_elements = await tab.select_all('input[name="in_sasinari_kbn"]')
            await order_type_elements[1].click()  # 成行にチェックを入れる
            # 「引成」に設定
            selector = f'select[name="nariyuki_condition"] option[value="H"]'
            tab = await select_pulldown(tab, selector)
            # 取引パスワード
            trade_password_input = await tab.wait_for('input[id="pwd3"]')
            await trade_password_input.send_keys(os.getenv('SBI_TRADEPASS'))            
            # 注文画面を省略
            skip_button = await tab.wait_for('input[id="shouryaku"]')
            await skip_button.click()
            # 注文発注
            order_button = await tab.wait_for('img[title="注文発注"]')
            await order_button.click()
            await tab.wait(1.2)  # 1.2秒待機
            # 注文確認
            try:
                await tab.wait_for(text='ご注文を受け付けました。')
                print(f"{margin_tickers[i]}：正常に注文完了しました。")
                retry = 0
                i += 1
            except:
                if retry < 3:
                    print(f"{margin_tickers[i]}：発注失敗。再度発注を試みます。")
                else:
                    print(f"{margin_tickers[i]}：発注失敗。リトライ回数の上限に達しました。")
                    error_tickers.append(margin_tickers[i])
                    retry = 0
                    i += 1
                
    print(f'全銘柄の決済処理が完了しました。')
    
    return tab, error_tickers

@_retry()
async def cancel_all_orders(tab:uc.core.tab.Tab=None) -> uc.core.tab.Tab:
    '''すべての注文をキャンセル'''
    # 注文一覧を取得
    tab = await sign_in(tab)
    tab, df = await _extract_order_list(tab)
    if df is None:
        return

    for i in range(len(df)):
        # 注文ページへ遷移
        button = await tab.wait_for('img[title=取引]')
        await button.click()
        await tab.wait(1)
        # 注文照会、取り消し・訂正ページへ遷移
        button = await tab.find('注文照会')
        await button.click()
        await tab.wait(1)
        # 任意取消をクリック ここではindexで2を指定
        button = await tab.find('取消')
        await button.click()
        await tab.wait(1)
        # 取引パスワード
        input = await tab.select('input[id="pwd3"]')
        await input.send_keys(os.getenv('SBI_TRADEPASS'))
        # 注文取消を確定
        button = await tab.select('input[value=注文取消]')
        await button.click()
        await tab.wait(1)
        # 注文取消が成功したかをチェック
        html = await tab.get_content()
        if "ご注文を受け付けました。" in html:
            print(f"{df['コード'].iloc[i]} {df['注文株数'].iloc[i]}株 {df['注文種別'].iloc[i]}：注文取消が完了しました。")
        else:
            print(f"{df['コード'].iloc[i]} {df['注文株数'].iloc[i]}株 {df['注文種別'].iloc[i]}：注文取消に失敗しました。")
    
    return tab



async def main():
    df =pd.read_csv(f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_list.csv')
    tab = await sign_in()
    #tab, mbp, bp = await get_buying_power(tab)
    #print(mbp)
    tab, bp, sp = await get_trade_possibility(tab)
    #tab = await make_order(tab, nariyuki_value='寄成',ticker='4502')
    #tab = await cancel_all_orders(tab)
    #tab, today_spots_df = await fetch_today_spots(tab)
    #tab, in_out_df = await fetch_in_out(tab)
    #tab, today_margin_df = await fetch_today_contracts(tab, df)
    '''
    tab, deal_history_df = await fetch_deal_history(tab, df, datetime(2024,6,24))
    invest_result = pd.read_csv(paths.TRADE_HISTORY_CSV)
    invest_result = pd.concat([invest_result, deal_history_df], axis=0)
    invest_result = invest_result.drop(columns=['Unnamed: 0'])
    invest_result['日付'] = pd.to_datetime(invest_result['日付'])
    invest_result = invest_result.sort_values(by=['日付', '銘柄コード'], ascending=True)
    invest_result = invest_result.drop_duplicates(keep='last')
    invest_result.to_csv(paths.TRADE_HISTORY_CSV)
    display(invest_result)
    '''

if __name__ == '__main__':
    uc.loop().run_until_complete(main())
    