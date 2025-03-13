from nodriver import Tab
from pathlib import Path
import asyncio


class TabUtils:
    """
    nodriver の Tab オブジェクトに対する操作（URLオープン、リロード、クリック等）をラップするユーティリティクラス
    """
    def __init__(self, tab: Tab):
        '''
        Args:
            tab (nodriver.Tab): nodriverのタブオブジェクト
        '''
        self._tab = tab

    async def open_url(self, url: str):
        """
        指定したURLを開きます。

        Args:
            url: 開く対象のURL
        """
        await self._tab.get(url)

    async def reload(self):
        """
        タブをリロードします。
        """
        await self._tab.reload()

    async def wait(self, t: int | float):
        """
        指定した秒数待機します。

        Args:
            t (int|float): 待機秒数
        """
        if isinstance(t, (int, float)) and t > 0:
            await self._tab.wait(t)

    async def wait_for(self, selector: str, is_css: bool = False, timeout: int | float | None = 60):
        """
        指定した要素の表示を待ちます。

        Args:
            selector (str): 表示を待ちたい文字列またはcssセレクタ
            is_css (bool): Trueならcssセレクタ、Falseなら文字列
            timeout (int|float|None): タイムアウト秒数

        Returns:
            element: 要素
        """
        if is_css:
            element = await self._tab.wait_for(selector=selector, timeout=timeout)
        else:
            element = await self._tab.wait_for(text=selector, timeout=timeout)
        return element

    async def click_element(self, selector_text: str, is_css: bool = False, timeout: int | float | None = 60):
        """
        指定した要素の表示を待ってからクリックします。

        Args:
            selector_text (str): 表示を待ちたい文字列 or CSSセレクタ
            is_css (bool): TrueならCSSセレクタ、Falseなら文字列の表示を待つ
            timeout (int|float|None): タイムアウト秒数
        """
        element = await self.wait_for(selector=selector_text, is_css=is_css, timeout=timeout)
        await asyncio.sleep(0.3)
        await element.click()

    async def send_keys_to_element(self, selector_text: str, keys: str, is_css: bool = False):
        """
        指定した要素の表示を待ってから文字列を送信します。

        Args:
            selector_text (str): 表示を待ちたい文字列 or CSSセレクタ
            is_css (bool): TrueならCSSセレクタ、Falseなら文字列の表示を待つ
            keys (str): 送信する文字列
        """
        element = await self.wait_for(selector=selector_text, is_css=is_css)
        await element.send_keys(keys)

    async def select_all(self, css_selector: str) -> list:
        '''
        特定のcssセレクタを持つ要素をすべて取得します。

        Args:
            css_selector (str): 取得対象のCSSセレクタ

        Returns:
            list: 指定したcssセレクタを持つ全ての要素を格納したリスト
        '''
        return await self._tab.select_all(css_selector)

    async def select_pulldown(self, css_selector: str):
        """
        プルダウンメニューのオプションを選択します。

        Args:
            css_selector (str): 選択したいプルダウンメニューのCSSセレクタ
        """
        await self._tab.evaluate(f'''
            var option = document.querySelector('{css_selector}');
            option.selected = true;
            var event = new Event('change', {{ bubbles: true }});
            option.parentElement.dispatchEvent(event);
        ''')
        await self._tab.wait(0.5)

    async def query_selector(self, css_selector: str, is_all: bool = False) -> list:
        """
        指定したcssセレクタを持つ要素を、1つまたは複数検索します。

        Args:
            css_selector (str): 探し出すcssセレクタ
            is_all (bool): Trueならquery_selector_all()、Falseならquey_selector()を実行

        Returns:
            list: 検索結果を1つ又は複数格納したリスト
        """
        if is_all:
            return await self._tab.query_selector_all(css_selector)
        else:
            return await self._tab.query_selector(css_selector)

    async def get_html_content(self) -> str:
        '''
        表示されたページのhtmlを取得します。

        Returns:
            str: 表示されたページのhtml
        '''
        return await self._tab.get_content()

    async def set_download_path(self, path: Path):
        '''
        ファイルのダウンロード先パスを指定します。

        Args:
            path (Path): ダウンロード先パス
        '''
        await self._tab.set_download_path(path)


if __name__ == '__main__':
    import asyncio

    async def main():
        from utils.browser.browser_manager import BrowserManager
        bm = BrowserManager()
        await bm.launch_browser()
        tab = bm.get_tab('default')
        tu = TabUtils(tab.tab._tab)
        await tu.open_url('https://www.google.co.jp/')
        await asyncio.sleep(5)
    
    asyncio.get_event_loop().run_until_complete(main())