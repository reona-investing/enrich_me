# data_fetch.py
import os
import re
import csv
from pathlib import Path
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup as soup
from .session import SBISession
from .utils.decorators import retry
from .utils.web import select_pulldown, get_newest_two_files
from .utils.formatting import format_deal_result_df, format_in_out_df
import paths
import unicodedata

class SBIDataFetcher:
    def __init__(self, session: SBISession = None):
        if session is None:
            session = SBISession()
        self.session = session
        self.deal_result_df = pd.DataFrame()
        self.in_out_df = pd.DataFrame()
        self.today_spots_df = pd.DataFrame()
        self.margin_buying_power = None
        self.buying_power = None
        self.trade_possibility_df = pd.DataFrame()
        self.buy_possibility = None
        self.sell_possibility = None

    #@retry()
    async def fetch_deal_result(self, sector_list_df:pd.DataFrame=None, mydate:datetime=datetime.today()):
        await self.session.sign_in()
        await self._fetch_deal_result_csv(mydate=mydate)
        self.deal_result_df[['手数料/諸経費等', '税額', '受渡金額/決済損益']] = \
            self.deal_result_df[['手数料/諸経費等', '税額', '受渡金額/決済損益']].replace({'--':'0'}).astype(int)
        self.deal_result_df = self.deal_result_df.groupby(['約定日', '銘柄', '銘柄コード', '市場', '取引']).sum().reset_index(drop=False)
        take_df = self.deal_result_df[(self.deal_result_df['取引']=='信用新規買')|(self.deal_result_df['取引']=='信用新規売')]
        take_df['売or買'] = '買'
        take_df = take_df.rename(columns={'約定日': '日付',
                                          '銘柄': '社名',
                                          '約定数量': '株数',
                                          '約定単価': '取得単価'})
        take_df.loc[self.deal_result_df['取引']=='信用新規売', '売or買'] = '売'
        settle_df = self.deal_result_df[(self.deal_result_df['取引']=='信用返済買')|(self.deal_result_df['取引']=='信用返済売')]
        settle_df = settle_df.rename(columns={'約定単価': '決済単価'})
        self.deal_result_df = pd.merge(take_df, settle_df[['銘柄コード', '決済単価']], how='outer', on='銘柄コード')

        self.deal_result_df = format_deal_result_df(self.deal_result_df, sector_list_df)
        self.deal_result_df['日付'] = pd.to_datetime(self.deal_result_df['日付']).dt.date
        print(self.deal_result_df)


    async def _fetch_deal_result_csv(self, mydate: datetime):
        myyear = f'{mydate.year}'
        mymonth = f'{mydate.month:02}'
        myday = f'{mydate.day:02}'
        # ナビゲーション
        button = await self.session.tab.find('取引履歴')
        await button.click()
        await self.session.tab.wait(1)
        button = await self.session.tab.select('#shinT')
        await button.click()
        element_num = {'from_yyyy':myyear, 'from_mm':mymonth, 'from_dd':myday,
                       'to_yyyy':myyear, 'to_mm':mymonth, 'to_dd':myday}
        for key, value in element_num.items():
            pulldown_selector = f'select[name="ref_{key}"] option[value="{value}"]'
            await select_pulldown(self.session.tab, pulldown_selector)
        button = await self.session.tab.find('照会')
        await button.click()
        await self.session.tab.wait(1)
        await self.session.tab.set_download_path(Path(paths.DOWNLOAD_FOLDER))
        button = await self.session.tab.find('CSVダウンロード')
        await button.click()
        await self.session.tab.wait(3)

        deal_result_csv = None
        for i in range(10):
            newest_file, second_file = get_newest_two_files(paths.DOWNLOAD_FOLDER)
            await self.session.tab.wait(1)
            if newest_file.endswith('.csv'):
                deal_result_csv = newest_file
                break
            if second_file.endswith('.csv'):
                deal_result_csv = second_file
                break
            
        self.deal_result_df = pd.read_csv(deal_result_csv, header=0, skiprows=8, encoding='shift_jis')
        os.remove(deal_result_csv)

    @retry()
    async def fetch_in_out(self):
        await self.session.sign_in()
        button = await self.session.tab.find('入出金明細')
        await button.click()
        await self.session.tab.wait(1)
        button = await self.session.tab.find('3ヶ月')
        await button.click()
        await self.session.tab.wait(1)
        pulldown_selector = 'select[name=in_v_list_count] option[value="200"]'
        await select_pulldown(self.session.tab, pulldown_selector)
        button = await self.session.tab.select('img[title=照会]')
        await button.click()
        await self.session.tab.wait(1)
        html_content = await self.session.tab.get_content()
        html = soup(html_content, "html.parser")
        table = html.find("th", string=re.compile("区分"))
        if table is None:
            print('直近1週間の入出金履歴はありません。')
            return
        table = table.findParent("table")
        self.in_out_df = format_in_out_df(table)
        print('入出金の履歴')
        print(self.in_out_df)

    @retry()
    async def fetch_today_contracts(self, sector_list_df:pd.DataFrame=None):
        await self.session.sign_in()
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
        self.deal_result_df = pd.DataFrame(data, columns=columns)
        self.deal_result_df = self.deal_result_df[(self.deal_result_df['売or買']=='買')|(self.deal_result_df['売or買']=='売')]
        self.deal_result_df = format_deal_result_df(self.deal_result_df, sector_list_df)

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

    @retry()
    async def fetch_today_spots(self):
        await self.session.sign_in()
        button = await self.session.tab.select('img[title=取引]')
        await button.click()
        await self.session.tab.wait(1)
        button = await self.session.tab.find('当日約定一覧')
        await button.click()
        await self.session.tab.wait(1)
        button = await self.session.tab.find('国内株式(現物)')
        await button.click()
        await self.session.tab.wait(1)

        html_content = await self.session.tab.get_content()
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
        self.today_spots_df = pd.DataFrame(data, columns=columns)
        self.today_spots_df['日付'] = pd.to_datetime(self.today_spots_df['日付']).dt.date
        self.today_spots_df['銘柄コード'] = self.today_spots_df['銘柄コード'].astype(str)
        self.today_spots_df['買付余力増減'] = self.today_spots_df['買付余力増減'].astype(int)
        self.today_spots_df.loc[self.today_spots_df['売or買']=='買', '買付余力増減'] = \
            - self.today_spots_df.loc[self.today_spots_df['売or買']=='買', '買付余力増減']
        print('現物売買')
        print(self.today_spots_df)

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

    @retry()
    async def get_buying_power(self):
        await self.session.sign_in()
        button = await self.session.tab.select('img[title="口座管理"]')
        await button.click()
        await self.session.tab.wait(3)
        html_content = await self.session.tab.get_content()
        html = soup(html_content, "html.parser")

        div = html.find("div", string=re.compile("信用建余力"))
        self.margin_buying_power = int(div.findNext("div").getText().strip().replace(',', ''))
        div = html.find("div", string=re.compile("買付余力\\(2営業日後\\)"))
        self.buying_power = int(div.findNext("div").getText().strip().replace(',', ''))

    @retry()
    async def get_trade_possibility(self):
        # 不要ファイル削除
        filelist = os.listdir(paths.DOWNLOAD_FOLDER)
        if len(filelist) > 0:
            for file in filelist:
                os.remove(f'{paths.DOWNLOAD_FOLDER}/{file}')

        await self.session.sign_in()
        elem = await self.session.tab.wait_for('#navi01P > ul > li:nth-child(3) > a')
        await elem.click()
        elem = await self.session.tab.wait_for('#rightNav\\ mt-8 > div:nth-child(1) > ul > li:nth-child(5) > a')
        await elem.click()
        await self.session.tab.wait(5)
        await self.session.tab.set_download_path(Path(paths.DOWNLOAD_FOLDER))
        download_link = await self.session.tab.wait_for('#csvDownload')
        await download_link.click()

        csv_path = None
        for i in range(10):
            await self.session.tab.wait(2)
            filelist = os.listdir(paths.DOWNLOAD_FOLDER)
            if len(filelist) > 0:
                for file in filelist:
                    if file.endswith('.csv'):
                        csv_path = f'{paths.DOWNLOAD_FOLDER}/{file}'
                        break
            if csv_path is not None:
                break

        self._convert_deal_history_csv_to_df(csv_path)
        self.trade_possibility_df.loc[self.trade_possibility_df['一人あたり建玉上限数'] == '-', 
                                      '一人あたり建玉上限数'] = '1000000'

        sellable_condition = \
            (self.trade_possibility_df['売建受注枠']!='受付不可') & \
            (self.trade_possibility_df['信用区分（HYPER）']=='')
        self.buy_possibility = {
            key: value for key, value in \
                zip(self.trade_possibility_df['コード'].astype(str), 
                    self.trade_possibility_df['一人あたり建玉上限数'].astype(int))
        }
        self.sell_possibility = {
            key: value for key, value in \
                zip(self.trade_possibility_df.loc[sellable_condition, 'コード'].astype(str),
                    self.trade_possibility_df.loc[sellable_condition, '一人あたり建玉上限数'].astype(int))
        }

    def _convert_deal_history_csv_to_df(self, deal_history_csv:str):
        extracted_rows = []
        found_code = False
        with open(deal_history_csv, newline='', encoding='shift_jis') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if not found_code:
                    if len(row) > 0 and row[0] == "コード":
                        found_code = True
                        columns = row
                else:
                    extracted_rows.append(row)
        self.trade_possibility_df = pd.DataFrame(extracted_rows, columns=columns)