import pandas as pd
from session.login_handler import LoginHandler
from bs4 import BeautifulSoup as soup
import re
import os
from datetime import datetime
import paths

class HistoryManager:
    def __init__(self):
        """取引履歴管理クラス"""
        self.tab = None
        self.save_path = paths.TRADE_HIStORY_CSV
        os.makedirs(self.save_path, exist_ok=True)
        self.trade_history_df = pd.DataFrame()
        self.login_handler = LoginHandler()  # LoginHandlerのインスタンスを作成

    async def fetch_trade_history(self) -> pd.DataFrame:
        """
        取引履歴をスクレイピングして取得
        Returns:
            pd.DataFrame: 取引履歴データ
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

        self.login_handler.session.tab = self.tab

    def save_trade_history(self, filename: str = None):
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

        save_path = os.path.join(self.save_path, filename)
        self.trade_history_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"取引履歴を{save_path}に保存しました。")
