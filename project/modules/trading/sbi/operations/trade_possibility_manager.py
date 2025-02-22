import pandas as pd
from trading.sbi.session import LoginHandler
from trading.sbi.browser import PageNavigator, BrowserUtils
import os
from pathlib import Path
from utils.paths import Paths
import csv

class TradePossibilityManager:
    def __init__(self, login_handler: LoginHandler):
        """取引制限情報の管理クラス"""
        self.download_path = Paths.DOWNLOAD_FOLDER
        os.makedirs(self.download_path, exist_ok=True)
        self.login_handler = login_handler
        self.page_navigator = PageNavigator(self.login_handler)
        self.browser_utils = BrowserUtils(self.login_handler)

    async def fetch(self) -> dict:
        """
        取引可能性情報を取得
        Returns:
            dict: 銘柄ごとの建玉上限と売建受注枠
        """
        self._remove_files_in_download_folder()
        
        # 取引可能性情報ページに遷移
        await self.page_navigator.domestic_top()
        await self.browser_utils.wait(1)
        await self.browser_utils.wait_and_click('一般信用売り銘柄一覧', is_css = False, timeout=60)

        # ダウンロード処理
        await self.browser_utils.wait_for('#csvDownload', is_css = True)
        await self.browser_utils.set_download_path(Path(self.download_path))
        await self.browser_utils.wait_and_click('#csvDownload', is_css = True)
        await self.browser_utils.wait(10)

        # CSVファイルの読み込み
        csv_file = self._get_latest_csv()
        trade_data = self._convert_csv_to_df(csv_file)
    
        # データ整形
        trade_data["一人あたり建玉上限数"] = trade_data["一人あたり建玉上限数"].replace("-", 1000000).astype(int)
        sellable_condition = (trade_data["売建受注枠"] != "受付不可") & (trade_data["信用区分（HYPER）"] == "")

        # 結果の辞書化
        self.data_dict = {
            "buyable_limits": dict(zip(trade_data["コード"], trade_data["一人あたり建玉上限数"])),
            "sellable_limits": dict(zip(trade_data.loc[sellable_condition, "コード"], 
                                        trade_data.loc[sellable_condition, "一人あたり建玉上限数"])),
            "borrowing_stocks": dict(zip(trade_data.loc[sellable_condition, "コード"], 
                                         trade_data.loc[sellable_condition, "信用区分（無期限）"].replace({"◎":True, "":False})))
        }

    def _remove_files_in_download_folder(self):
        filelist = os.listdir(Paths.DOWNLOAD_FOLDER)
        if len(filelist) > 0:
            for file in filelist:
                os.remove(f'{Paths.DOWNLOAD_FOLDER}/{file}')

    def _get_latest_csv(self) -> Path:
        """
        ダウンロードフォルダから最新のCSVファイルを取得
        Returns:
            Path: 最新のCSVファイルパス
        """
        for i in range(60):
            print(i)
            files = list(Path(self.download_path).glob("*.csv"))
            if files:
                return max(files, key=os.path.getmtime)
            self.browser_utils.wait(1)
        raise FileNotFoundError("取引可能情報のCSVが見つかりません。")
        

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
        lh = LoginHandler()
        tpm = TradePossibilityManager(lh)
        await tpm.fetch()
        print(tpm.data_dict['borrowing_stocks'])
    
    import asyncio
    asyncio.run(main())