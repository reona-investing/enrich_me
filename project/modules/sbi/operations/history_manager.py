import pandas as pd
from ..session.login_handler import LoginHandler
from ..browser.browser_utils import BrowserUtils
from ..browser.file_utils import FileUtils
from bs4 import BeautifulSoup as soup
import re
import unicodedata
import os
from datetime import datetime
from pathlib import Path
import paths

class HistoryManager:
    def __init__(self, login_handler: LoginHandler):
        """取引履歴管理クラス"""
        self.tab = None
        self.save_dir_path = paths.TRADE_HISTORY_FOLDER
        os.makedirs(self.save_dir_path, exist_ok=True)
        self.save_file_path = paths.TRADE_HISTORY_CSV
        self.today_margin_trades_df = pd.DataFrame()
        self.past_margin_trades_df = pd.DataFrame()
        self.today_margin_trades_df = pd.DataFrame()
        self.cashflow_transactions_df = pd.DataFrame()
        self.today_stock_trades_df = pd.DataFrame()
        self.login_handler = login_handler

    async def fetch_today_margin_trades(self, sector_list_df:pd.DataFrame=None):
        """
        過去の取引履歴をスクレイピングして取得
        self.today_margin_trades_df: 取引履歴データ
        """
        await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        self.tab = self.login_handler.session.tab

        button = await self.session.tab.select('img[title=口座管理]')
        await button.click()
        await self.session.tab.wait(1)
        button = await self.session.tab.find('当日約定一覧')
        await button.click()
        await self.session.tab.wait(1)
        button = await self.session.tab.find('国内株式(信用)')
        await button.click()
        await self.session.tab.wait(1)
        html_content = await self.session.tab.get_content()
        html = soup(html_content, 'html.parser')
        table = html.find("td", string=re.compile("銘柄"))
        if table is None:
            print('本日約定の注文はありません。')
            return
        table = table.findParent("table")

        data = []
        for tr in table.find("tbody").findAll("tr"):
            if tr.find("td").find("a"):
                data = self._append_contract_to_list(tr, data)

        columns = ["日付", "売or買", "銘柄コード", "社名", "株数", "取得単価", "決済単価"]
        self.today_margin_trades_df = pd.DataFrame(data, columns=columns)
        self.today_margin_trades_df = self.today_margin_trades_df[(self.today_margin_trades_df['売or買']=='買')|(self.today_margin_trades_df['売or買']=='売')]
        self.today_margin_trades_df = self._format_contracts_df(self.today_margin_trades_df, sector_list_df)

    def _append_contract_to_list(self, tr:object, data:list) -> list:
        row = []
        text = unicodedata.normalize("NFKC", tr.findAll("td")[2].getText().strip())
        row.append(datetime(2000+int(text[0:2]), int(text[3:5]), int(text[6:8])))
        text = unicodedata.normalize("NFKC", tr.findAll("td")[1].getText().strip())[:3]
        if text == '信新売':
            row.append('売')
        elif text == '信新買':
            row.append('買')
        text = unicodedata.normalize("NFKC", tr.findAll("td")[0].getText().strip())
        row.append(text[-4:])
        row.append(text[:-4])
        row.append(tr.findAll("td")[3].getText().replace(",", "").strip())
        row.append(tr.findAll("td")[4].getText().replace(",", "").strip())
        row.append(tr.findNext("tr").findAll("td")[3].getText().replace(",", "").strip())
        data.append(row)
        return data

    async def fetch_past_margin_trades(self, sector_list_df:pd.DataFrame=None, mydate:datetime=datetime.today()):
        """
        過去の取引履歴をスクレイピングして取得
        self.past_margin_trades_df: 取引履歴データ
        """
        await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        self.tab = self.login_handler.session.tab

        df = await self._fetch_past_margin_trades_csv(mydate=mydate)
        df[['手数料/諸経費等', '税額', '受渡金額/決済損益']] = df[['手数料/諸経費等', '税額', '受渡金額/決済損益']].replace({'--':'0'}).astype(int)
        df = df.groupby(['約定日', '銘柄', '銘柄コード', '市場', '取引']).sum().reset_index(drop=False)
        take_df = df[(df['取引']=='信用新規買')|(df['取引']=='信用新規売')]
        take_df['売or買'] = '買'
        take_df = take_df.rename(columns={'約定日': '日付',
                                          '銘柄': '社名',
                                          '約定数量': '株数',
                                          '約定単価': '取得単価'})
        take_df.loc[df['取引']=='信用新規売', '売or買'] = '売'
        settle_df = df[(df['取引']=='信用返済買')|(df['取引']=='信用返済売')]
        settle_df = settle_df.rename(columns={'約定単価': '決済単価'})
        df = pd.merge(take_df, settle_df[['銘柄コード', '決済単価']], how='outer', on='銘柄コード')

        self.past_margin_trades_df = self._format_contracts_df(df, sector_list_df)
        self.past_margin_trades_df['日付'] = pd.to_datetime(self.past_margin_trades_df['日付']).dt.date
        print(self.past_margin_trades_df)


    async def _fetch_past_margin_trades_csv(self, mydate: datetime) -> pd.DataFrame: 
        # ナビゲーション
        button = await self.tab.find('取引履歴')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.select('#shinT')
        await button.click()
        element_num = {'from_yyyy':f'{mydate.year}', 'from_mm':f'{mydate.month:02}', 'from_dd':f'{mydate.day:02}',
                       'to_yyyy':f'{mydate.year}', 'to_mm':f'{mydate.month:02}', 'to_dd':f'{mydate.day:02}'}
        for key, value in element_num.items():
            pulldown_selector = f'select[name="ref_{key}"] option[value="{value}"]'
            await BrowserUtils.select_pulldown(self.tab, pulldown_selector)
        button = await self.tab.find('照会')
        await button.click()
        await self.tab.wait(1)
        await self.tab.set_download_path(Path(paths.DOWNLOAD_FOLDER))
        button = await self.tab.find('CSVダウンロード')
        await button.click()
        await self.tab.wait(5)

        csvs = list(Path(paths.DOWNLOAD_FOLDER).glob("*.csv"))
        if not csvs:
            raise FileNotFoundError("取引可能情報のCSVが見つかりません。")
        
        newest_csv, _ = FileUtils.get_newest_two_csvs(paths.DOWNLOAD_FOLDER)
        df = pd.read_csv(newest_csv, header=0, skiprows=8, encoding='shift_jis')
        for csv in csvs:
            os.remove(csv)

        return df

    def _format_contracts_df(self, df: pd.DataFrame, sector_list_df: pd.DataFrame) -> pd.DataFrame:
        df['銘柄コード'] = df['銘柄コード'].astype(str)
        df['株数'] = df['株数'].astype(int)
        df['取得単価'] = df['取得単価'].astype(float)
        df['決済単価'] = df['決済単価'].astype(float)

        df['取得価格'] = (df['取得単価'] * df['株数']).astype(int)
        df['決済価格'] = (df['決済単価'] * df['株数']).astype(int)
        df['手数料'] = 0
        df['利益（税引前）'] = 0
        df.loc[df['売or買']=='買', '利益（税引前）'] = df['決済価格'] - df['取得価格'] - df['手数料']
        df.loc[df['売or買']=='売', '利益（税引前）'] = df['取得価格'] - df['決済価格'] - df['手数料']
        df['利率（税引前）'] = df['利益（税引前）'] / df['取得価格']

        sector_list_df['Code'] = sector_list_df['Code'].astype(str)
        df = pd.merge(df, sector_list_df[['Code', 'Sector']], left_on='銘柄コード', right_on='Code', how='left')
        df = df.drop('Code', axis=1).rename(columns={'Sector':'業種'})
        df = df[['日付', '売or買', '業種', '銘柄コード', '社名', '株数', '取得単価', '決済単価', '取得価格', '決済価格', '手数料', '利益（税引前）', '利率（税引前）']]
        return df
    

    async def fetch_cashflow_transactions(self):
        """
        直近1週間の入出金履歴をスクレイピングして取得
        self.cashflow_transactions_df: 取引履歴データ
        """
        await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        self.tab = self.login_handler.session.tab
        
        button = await self.tab.find('入出金明細')
        await button.click()
        await self.tab.wait(1)
        
        selected_element = await self.tab.select('#fc-page-size > div:nth-child(1) > div > select > option:nth-child(5)')
        await selected_element.select_option()
        await self.tab.wait(1)

        # タイトル行の取得
        parent_element = await self.tab.select('#fc-page-table > div > ul')
        elements = parent_element.children

        data_for_df = []
        for i, element in enumerate(elements):
            texts = []
            if i == 0:
                content = await element.get_html()
                html = soup(content, 'html.parser')
                titles = [div.get_text(strip=True) for div in html.find_all('div', class_='table-head')]
            else:
                children_elements = element.children
                for child_element in children_elements:
                    grandchild_element = child_element.children[0]
                    texts.append(grandchild_element.text)
                data_for_df.append(texts)
        df = pd.DataFrame(data_for_df, columns = titles)

        if len(df) == 0:
            print('直近1週間の入出金履歴はありません。')
            return
        self.cashflow_transactions_df = self._format_cashflow_transactions_df(df)

        button = await self.tab.find('総合トップ')

        print('入出金の履歴')
        print(self.cashflow_transactions_df)

    def _format_cashflow_transactions_df(self, df: pd.DataFrame) -> pd.DataFrame:
        #日付型に変換
        df['日付'] = pd.to_datetime(df['入出金日']).dt.date

        # ハイフンや空文字を0に変換して、数値型に変換
        for col in ["出金額", "入金額", "振替出金額", "振替入金額"]:
            df[col] = df[col].astype(str).replace("-", "0")
            df[col] = df[col].str.replace(",", "")
            df[col] = df[col].astype(int)

        df['入出金額'] = df['入金額'] + df['振替入金額'] - df['出金額'] - df['振替出金額']
        df = df.loc[~df['摘要'].str.contains('譲渡益税')]
        df = df[['日付', '摘要', '入出金額']]

        return df
    
    async def fetch_today_margin_trades(self, sector_list_df:pd.DataFrame=None):
        """
        直近1週間の入出金履歴をスクレイピングして取得
        self.today_margin_trades_df: 取引履歴データ
        """
        await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        self.tab = self.login_handler.session.tab
       
        button = await self.tab.select('img[title=口座管理]')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('当日約定一覧')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('国内株式(信用)')
        await button.click()
        await self.tab.wait(1)
        html_content = await self.tab.get_content()
        html = soup(html_content, 'html.parser')
        table = html.find("td", string=re.compile("銘柄"))
        if table is None:
            print('本日約定の注文はありません。')
            return
        table = table.findParent("table")

        data = []
        for tr in table.find("tbody").findAll("tr"):
            if tr.find("td").find("a"):
                data = self._append_contract_to_list(tr, data)

        columns = ["日付", "売or買", "銘柄コード", "社名", "株数", "取得単価", "決済単価"]
        df = pd.DataFrame(data, columns=columns)
        df = df[(df['売or買']=='買')|(df['売or買']=='売')]
        self.today_margin_trades_df = self._format_contracts_df(df, sector_list_df)

    def _append_contract_to_list(self, tr:object, data:list) -> list:
        row = []
        text = unicodedata.normalize("NFKC", tr.findAll("td")[2].getText().strip())
        row.append(datetime(2000+int(text[0:2]), int(text[3:5]), int(text[6:8])))
        text = unicodedata.normalize("NFKC", tr.findAll("td")[1].getText().strip())[:3]
        if text == '信新売':
            row.append('売')
        elif text == '信新買':
            row.append('買')
        text = unicodedata.normalize("NFKC", tr.findAll("td")[0].getText().strip())
        row.append(text[-4:])
        row.append(text[:-4])
        row.append(tr.findAll("td")[3].getText().replace(",", "").strip())
        row.append(tr.findAll("td")[4].getText().replace(",", "").strip())
        row.append(tr.findNext("tr").findAll("td")[3].getText().replace(",", "").strip())
        data.append(row)

        return data
    
    async def fetch_today_stock_trades(self):
        """
        今日の現物取引をスクレイピングして取得
        self.today_stock_trades_df: 現物取引データ
        """
        await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        self.tab = self.login_handler.session.tab

        button = await self.tab.select('img[title=取引]')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('当日約定一覧')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.find('国内株式(現物)')
        await button.click()
        await self.tab.wait(1)

        html_content = await self.tab.get_content()
        html = soup(html_content, 'html.parser')
        table = html.find('td', string=re.compile('銘柄'))
        if table is None:
            print('本日約定の注文はありません。')
            return
        table = table.findParent("table")

        data = []
        for tr in table.find("tbody").findAll("tr"):
            if tr.find("td").find("a"):
                data = self._append_spot_to_list(tr, data)

        columns = ["日付", "売or買", "銘柄コード", "社名", "買付余力増減"]
        self.today_stock_trades_df = pd.DataFrame(data, columns=columns)
        self.today_stock_trades_df['日付'] = pd.to_datetime(self.today_stock_trades_df['日付']).dt.date
        self.today_stock_trades_df['銘柄コード'] = self.today_stock_trades_df['銘柄コード'].astype(str)
        self.today_stock_trades_df['買付余力増減'] = self.today_stock_trades_df['買付余力増減'].astype(int)
        self.today_stock_trades_df.loc[self.today_stock_trades_df['売or買']=='買', '買付余力増減'] = \
            - self.today_stock_trades_df.loc[self.today_stock_trades_df['売or買']=='買', '買付余力増減']
        print('現物売買')
        print(self.today_stock_trades_df)

    def _append_spot_to_list(self, tr:object, data:list) -> list:
        row = []
        text = unicodedata.normalize("NFKC", tr.findAll("td")[2].getText().strip())
        row.append(datetime(2000+int(text[0:2]), int(text[3:5]), int(text[6:8])))
        text = unicodedata.normalize("NFKC", tr.findAll("td")[1].getText().strip())[:3]
        if text == '現物売':
            row.append('売')
        elif text == '現物買':
            row.append('買')
        text = unicodedata.normalize("NFKC", tr.findAll("td")[0].getText().strip())
        row.append(text[-4:])
        row.append(text[:-4])
        text = unicodedata.normalize("NFKC", tr.findAll("td")[7].getText().replace(",", "").strip())
        row.append(text[:text.index("(")])
        data.append(row)
        return data
    
    