import pandas as pd
from trading.sbi.browser import PageNavigator, SBIBrowserManager
from utils.browser.browser_manager import BrowserManager
from trading.jpx import LoanMarginsListGetter
import os
from pathlib import Path
from utils.paths import Paths
import csv

class TradePossibilityManager:
    def __init__(self, sbi_browser_manager: SBIBrowserManager):
        """取引制限情報の管理クラス"""
        self.download_path = Paths.DOWNLOAD_FOLDER
        os.makedirs(self.download_path, exist_ok=True)
        self.sbi_browser_manager = sbi_browser_manager
        self.page_navigator = PageNavigator(browser_manager=sbi_browser_manager)
        self.browser_manager = BrowserManager()



    async def fetch(self):
        await self.sbi_browser_manager.launch()
        await self._fetch_trade_possibility()
        await self._fetch_loan_margins_list()
        
        self.trade_data = pd.merge(self.trade_data, self.loan_margins_list[['コード', '信用区分']], how='inner', on='コード')

        sellable_condition = \
            ((self.trade_data["信用区分"] == "貸借銘柄") | (self.trade_data["売建受注枠"] != "受付不可")) & \
            (self.trade_data["信用区分（HYPER）"] == "")

        # 結果の辞書化
        map_dict = {"貸借銘柄":True, "制度信用銘柄":False}
        self.data_dict = {
            "buyable_limits": dict(zip(self.trade_data["コード"], self.trade_data["一人あたり建玉上限数"])),
            "sellable_limits": dict(zip(self.trade_data.loc[sellable_condition, "コード"], 
                                        self.trade_data.loc[sellable_condition, "一人あたり建玉上限数"])),
            "borrowing_stocks": dict(zip(self.trade_data.loc[sellable_condition, "コード"], 
                                         self.trade_data.loc[sellable_condition, "信用区分"].map(map_dict)))
        }

    async def _fetch_trade_possibility(self):
        """
        取引可能性情報を取得
        Returns:
            dict: 銘柄ごとの建玉上限と売建受注枠
        """
        self._remove_files_in_download_folder()
        
        # 取引可能性情報ページに遷移
        named_tab = await self.page_navigator.domestic_top()
        await named_tab.tab.utils.wait(1)
        await named_tab.tab.utils.click_element('一般信用売り銘柄一覧')
        await named_tab.tab.utils.wait(1)
        # ダウンロード処理
        for _ in range (5):
            await named_tab.tab.utils.set_download_path(Path(self.download_path))
            await named_tab.tab.utils.wait(1)
            element = await named_tab.tab.utils.wait_for('a[id="csvDownload"]', is_css = True, timeout = 10)
            await element.click()
            await named_tab.tab.utils.wait(10)
            # CSVファイルの読み込み
            csv_file = await self._get_latest_csv()
            if csv_file is not None:
                break
        
        if csv_file is None:
            raise FileNotFoundError("取引可能情報のCSVが見つかりません。")
        self.trade_data = self._convert_csv_to_df(csv_file)
    
        # データ整形
        self.trade_data["一人あたり建玉上限数"] = self.trade_data["一人あたり建玉上限数"].replace("-", 1000000).astype(int)


    async def _fetch_loan_margins_list(self):
        getter = LoanMarginsListGetter(self.browser_manager)
        self.loan_margins_list = await getter.get()
        self.loan_margins_list = self.loan_margins_list.rename(columns={'銘柄コード': 'コード'})


    def _remove_files_in_download_folder(self):
        filelist = os.listdir(Paths.DOWNLOAD_FOLDER)
        if len(filelist) > 0:
            for file in filelist:
                os.remove(f'{Paths.DOWNLOAD_FOLDER}/{file}')

    async def _get_latest_csv(self) -> Path:
        """
        ダウンロードフォルダから最新のCSVファイルを取得
        Returns:
            Path: 最新のCSVファイルパス
        """
        named_tab = self.sbi_browser_manager.get_tab('SBI')
        for i in range(60):
            files = list(Path(self.download_path).glob("*.csv"))
            if files:
                return max(files, key=os.path.getmtime)
            await named_tab.tab.utils.wait(1)
        

    def _convert_csv_to_df(self, csvfile:str):
        extracted_rows = []
        found_code = False
        with open(csvfile, newline='', encoding='shift_jis') as mycsvfile:
            reader = csv.reader(mycsvfile)
            for row in reader:
                if not found_code:
                    if len(row) > 0 and row[0] == "コード":
                        found_code = True
                        columns = row
                else:
                    extracted_rows.append(row)
        return pd.DataFrame(extracted_rows, columns=columns)


if __name__ == "__main__":

    async def main():
        sbi_browser_manager = SBIBrowserManager()
        tpm = TradePossibilityManager(sbi_browser_manager)
        await tpm.fetch()
        print(tpm.data_dict)
    
    import asyncio
    asyncio.get_event_loop().run_until_complete(main())