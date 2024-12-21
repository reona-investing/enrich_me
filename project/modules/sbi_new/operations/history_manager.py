import pandas as pd
from session.login_handler import LoginHandler
from bs4 import BeautifulSoup as soup
import re
import os
from datetime import datetime
import paths

class HistoryManager:
    def __init__(self, login_handler: LoginHandler):
        """取引履歴管理クラス"""
        self.tab = None
        self.save_dir_path = paths.TRADE_HISTORY_FOLDER
        os.makedirs(self.save_dir_path, exist_ok=True)
        self.save_file_path = paths.TRADE_HISTORY_CSV
        self.trade_history_df = pd.DataFrame()
        self.login_handler = login_handler

    async def fetch_trade_history(self):
        """
        取引履歴をスクレイピングして以下のファイルに取得
        self.trade_history_df: 取引履歴データ
        """
        await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        self.tab = self.login_handler.session.tab
        
        # 取引履歴ページに遷移
        button = await self.tab.wait_for('img[title="口座管理"]')
        await button.click()
        await self.tab.wait(1)
        button = await self.tab.wait_for('a:has-text("取引履歴")')
        await button.click()
        await self.tab.wait(3)

        # ページHTMLの取得
        html_content = await self.tab.get_content()
        html = soup(html_content, "html.parser")

        # 取引履歴テーブルの取得
        table = html.find("th", string=re.compile("取引区分")).find_parent("table")
        rows = table.find("tbody").find_all("tr")

        # データの抽出
        data = []
        for row in rows:
            cols = [col.get_text(strip=True) for col in row.find_all("td")]
            if cols:  # 空行を除外
                data.append(cols)

        # DataFrameに変換
        columns = ["取引区分", "銘柄", "数量", "単価", "手数料", "日付"]
        self.trade_history_df = pd.DataFrame(data, columns=columns)
        self.trade_history_df["数量"] = self.trade_history_df["数量"].astype(int)
        self.trade_history_df["単価"] = self.trade_history_df["単価"].str.replace(",", "").astype(float)
        self.trade_history_df["日付"] = pd.to_datetime(self.trade_history_df["日付"])

        self._save_trade_history()

        self.login_handler.session.tab = self.tab

    def _save_trade_history(self):
        """
        取引履歴をCSVファイルとして保存
        Args:
            filename (str): 保存するファイル名
        """
        if self.trade_history_df.empty:
            print("保存する取引履歴がありません。")
            return

        if not filename:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv"

        self.trade_history_df.to_csv(self.save_file_path, index=False, encoding="utf-8")
        print(f"取引履歴を{self.save_file_path}に保存しました。")




    async def fetch_past_margin_trades(self, sector_list_df:pd.DataFrame=None, mydate:datetime=datetime.today()):
        """
        過去の取引履歴をスクレイピングして取得
        self.past_margin_trades_df: 取引履歴データ
        """
        await self.login_handler.sign_in()  # LoginHandlerを使ってログイン
        self.tab = self.login_handler.session.tab

        await self._fetch_past_margin_trades_csv(mydate=mydate)
        self.past_margin_trades_df[['手数料/諸経費等', '税額', '受渡金額/決済損益']] = \
            self.past_margin_trades_df[['手数料/諸経費等', '税額', '受渡金額/決済損益']].replace({'--':'0'}).astype(int)
        self.past_margin_trades_df = self.past_margin_trades_df.groupby(['約定日', '銘柄', '銘柄コード', '市場', '取引']).sum().reset_index(drop=False)
        take_df = self.past_margin_trades_df[(self.past_margin_trades_df['取引']=='信用新規買')|(self.past_margin_trades_df['取引']=='信用新規売')]
        take_df['売or買'] = '買'
        take_df = take_df.rename(columns={'約定日': '日付',
                                          '銘柄': '社名',
                                          '約定数量': '株数',
                                          '約定単価': '取得単価'})
        take_df.loc[self.past_margin_trades_df['取引']=='信用新規売', '売or買'] = '売'
        settle_df = self.past_margin_trades_df[(self.past_margin_trades_df['取引']=='信用返済買')|(self.past_margin_trades_df['取引']=='信用返済売')]
        settle_df = settle_df.rename(columns={'約定単価': '決済単価'})
        self.past_margin_trades_df = pd.merge(take_df, settle_df[['銘柄コード', '決済単価']], how='outer', on='銘柄コード')

        self.past_margin_trades_df = format_contracts_df(self.past_margin_trades_df, sector_list_df)
        self.past_margin_trades_df['日付'] = pd.to_datetime(self.past_margin_trades_df['日付']).dt.date
        print(self.past_margin_trades_df)


    async def _fetch_past_margin_trades_csv(self, mydate: datetime):
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
            
        self.past_margin_trades_df = pd.read_csv(deal_result_csv, header=0, skiprows=8, encoding='shift_jis')
        os.remove(deal_result_csv)