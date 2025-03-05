import pandas as pd
from trading.sbi.session import LoginHandler
from trading.sbi.browser import PageNavigator, SBIBrowserUtils, FileUtils
from bs4 import BeautifulSoup as soup
import re
import unicodedata
import os
from datetime import datetime
from pathlib import Path
from utils.paths import Paths

class HistoryManager:
    def __init__(self, login_handler: LoginHandler):
        """取引履歴管理クラス"""
        self.save_dir_path = Paths.TRADE_HISTORY_FOLDER
        os.makedirs(self.save_dir_path, exist_ok=True)
        self.save_file_path = Paths.TRADE_HISTORY_CSV
        self.today_margin_trades_df = pd.DataFrame()
        self.past_margin_trades_df = pd.DataFrame()
        self.today_margin_trades_df = pd.DataFrame()
        self.cashflow_transactions_df = pd.DataFrame()
        self.today_stock_trades_df = pd.DataFrame()
        self.login_handler = login_handler
        self.page_navigator = PageNavigator(self.login_handler)
        self.browser_utils = SBIBrowserUtils(self.login_handler)


    async def fetch_today_margin_trades(self, sector_list_df:pd.DataFrame=None):
        """
        過去の取引履歴をスクレイピングして取得
        self.today_margin_trades_df: 取引履歴データ
        """
        await self.page_navigator.domestic_margin()
        await self.browser_utils.wait(3)

        table_elements = []
        
        while True:
            html_content = await self.browser_utils.get_html_content()
            html = soup(html_content, 'html.parser')
            table = html.find("td", string=re.compile("銘柄"))
            if table is None:
                print('本日約定の注文はありません。')
                return
            table = table.findParent("table")
            for i, tr in enumerate(table.find("tbody").findAll("tr")):
                if i > 0:
                    table_elements.append(tr)
            try:
                await self.browser_utils.wait_and_click('次へ→', timeout=3)
            except:
                break

        table_elements = self._add_previous_mtext(table_elements)   
        df = self._get_conract_summary_df(table_elements)

        df_take = df[df['売or買'].isin(['信新売', '信新買'])].rename(columns={'単価':'取得単価'})
        df_settle = df[df['売or買'].isin(['信返売', '信返買'])].rename(columns={'単価':'決済単価'})

        df = pd.merge(df_take, df_settle[['銘柄コード', '決済単価']], how='outer', on='銘柄コード').fillna(0)
        df['売or買'] = df['売or買'].replace({'信新売':'売', '信新買':'買'})
        # データフレームを表示
        print(df)

        self.today_margin_trades_df = df
        self.today_margin_trades_df = self._format_contracts_df(self.today_margin_trades_df, sector_list_df)

    def _add_previous_mtext(self, html_list: list) -> list:
        previous_mtext = None
        for html in html_list:
            mtext_td = html.find('td', class_='mtext')
            if mtext_td:
                previous_mtext = mtext_td
            else:
                previous_mtext_copy = soup(str(previous_mtext), 'html.parser')
                html.insert(0, previous_mtext_copy)
        return html_list

    def _get_conract_summary_df(self, table_elements: list) -> pd.DataFrame:
        data = []
        for row in table_elements:
            mtext = row.find('td', class_='mtext')
            mbody = row.find_all('td', class_='mbody')

            company_name = mtext.contents[0].get_text()
            stock_code = mtext.contents[2].get_text()
            type_ = mbody[0].get_text(strip=True)[:3]
            date_text = mbody[1].get_text(strip=True).split('<br/>')[0]
            date = datetime(2000 + int(date_text[0:2]), int(date_text[3:5]), int(date_text[6:8]))
            shares = int(mbody[2].get_text(strip=True).replace(",", ""))
            price = float(mbody[3].get_text(strip=True).replace(",", ""))

            data.append([date, type_, stock_code, company_name, shares, price])
        df = pd.DataFrame(data, columns=["日付", "売or買", "銘柄コード", "社名", "株数", "単価"])
        df['単価'] *=  df['株数']
        df = df.groupby(['日付', '売or買', '銘柄コード', '社名'])[['株数', '単価']].sum().reset_index(drop=False)
        df['単価'] /= df['株数']
        return df


    async def fetch_past_margin_trades(self, sector_list_df:pd.DataFrame=None, mydate:datetime=datetime.today()):
        """
        過去の取引履歴をスクレイピングして取得
        self.past_margin_trades_df: 取引履歴データ
        """
        df = await self.page_navigator.fetch_past_margin_trades_csv(mydate=mydate)
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

        csvs = list(Path(Paths.DOWNLOAD_FOLDER).glob("*.csv"))
        if not csvs:
            raise FileNotFoundError("取引可能情報のCSVが見つかりません。")
        
        newest_csv, _ = FileUtils.get_newest_two_csvs(Paths.DOWNLOAD_FOLDER)
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
        """
        await self.page_navigator.cashflow_transactions()        
        df = await self._convert_fetched_data_to_df()
        if len(df) == 0:
            print('直近1週間の入出金履歴はありません。')
            return
        self.cashflow_transactions_df = self._format_cashflow_transactions_df(df)
        print('入出金の履歴')
        print(self.cashflow_transactions_df)

    async def _convert_fetched_data_to_df(self) -> pd.DataFrame:
        try:
            selected_element = await self.browser_utils.select_element(
                selector_text = '#fc-page-size > div:nth-child(1) > div > select > option:nth-child(5)', 
                is_css = True)
        except:
            return pd.DataFrame()
        await selected_element.select_option()
        parent_element = await self.browser_utils.select_element(
            selector_text = '#fc-page-table > div > ul',
            is_css = True)
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
        return pd.DataFrame(data_for_df, columns = titles)

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
    
    
    async def fetch_today_stock_trades(self):
        """
        今日の現物取引をスクレイピングして取得
        self.today_stock_trades_df: 現物取引データ
        """
        await self.page_navigator.domestic_stock()
        html_content = await self.browser_utils.get_html_content()
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
    
    