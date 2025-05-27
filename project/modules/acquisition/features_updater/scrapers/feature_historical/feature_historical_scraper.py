from utils.browser.browser_manager import BrowserManager
from bs4 import BeautifulSoup as soup
import pandas as pd


class FeatureHistoricalScraper:
    """投資情報サイト(investing.com)からヒストリカルデータを取得するクラス"""
    
    def __init__(self, browser_manager: BrowserManager):
        self.browser_manager = browser_manager

    async def scrape_historical_data(self, investing_code: str) -> pd.DataFrame:
        """
        investing.comから指定された銘柄のヒストリカルデータを取得
        
        Args:
            investing_code (str): investing.comの銘柄コード
            
        Returns:
            pd.DataFrame: ヒストリカルデータ ['Date', 'Open', 'Close', 'High', 'Low']
        """
        investing_url = 'https://jp.investing.com/' + investing_code + '-historical-data'
        return await self._scrape_from_investing(name=investing_code, url=investing_url)

    async def _scrape_from_investing(self, name: str, url: str) -> pd.DataFrame:
        """
        investingからのスクレイピング
        
        Args:
            name (str): タブの名前
            url (str): タブで開くURL

        Returns:
            pd.DataFrame: スクレイピングで取得した価格情報df
        """
        named_tab = None
        max_retry = 10
        for i in range(max_retry):
            try:
                if named_tab:
                    print('reloading...')
                    await named_tab.tab.utils.reload()
                else:
                    named_tab = await self.browser_manager.new_tab(name=name, url=url)
                    
                # テーブル全体を読み込むのに時間がかかるようなので、要素の存在を確認した後待機時間を設ける
                await named_tab.tab.utils.wait_for('日付け')
                await named_tab.tab.utils.wait(1)
                table_header_initial = await named_tab.tab.utils.wait_for('日付け')
                table_selector = table_header_initial
                if table_selector is None:
                    raise ValueError(f'Table header not found: {url}')
                while table_selector.tag_name != "table":
                    table_selector = table_selector.parent
                    if table_selector is None:
                        raise ValueError(f'Table element not found: {url}')
                html = await table_selector.get_html()
                souped = soup(html, "html.parser")
                await self.browser_manager.close_tab(name=name)

                # ヘッダーの取得
                headers = [th.text.strip() for th in souped.select("thead th")]
                # データの取得
                rows = []
                for tr in souped.select("tbody tr"):
                    cells = [td.text.strip() for td in tr.find_all("td")]
                    rows.append(cells)         

                # DataFrame に変換
                df_to_add = pd.DataFrame(rows, columns=headers)
                break
            except:
                continue
        else:
            raise ValueError(f'DataFrame is not found: {url}')
        
        return self._format_investing_df(df_to_add)

    def _format_investing_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """investing.comから取得したDataFrameを標準形式にフォーマット"""
        df = df.iloc[:, :5].copy()
        df.columns = ['Date', 'Close', 'Open', 'High', 'Low']
        
        for fmt in ['%Y年%m月%d日', '%m月 %d, %Y', '%b %d, %Y']:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format=fmt)
                break
            except:
                continue

        # 4桁以上の値にカンマが入っているため、そのままだとstr型になってしまう。
        df[['Open', 'Close', 'High', 'Low']] =df[['Open', 'Close', 'High', 'Low']].replace(',', '', regex=True).astype(float)
        

        return df[['Date', 'Open', 'Close', 'High', 'Low']].sort_values(by='Date', ascending=True)