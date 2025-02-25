import nodriver as uc
from trading.sbi.session import LoginHandler
from pathlib import Path

class BrowserUtils:
    def __init__(self, login_handler: LoginHandler):
        '''
        ブラウザを用いた操作を定義します。
        Args:
            login_handler (LoginHandler): SBI証券へのログイン状態を確認します。
        '''
        self.login_handler = login_handler
        self.browser = self.login_handler.session.browser
        self.tab = None


    async def _set_tab(self):
        self.tab = await self.login_handler.sign_in()


    async def wait(self, t: int | float):
        """
        指定した秒数待機します。
        Args:
            t (int|float): 待機秒数
        """
        await self._set_tab()
        if type(t) is int or type(t) is float:
            if t > 0:
                await self.tab.wait(t)

    async def select_element(self, selector_text: str, is_css: bool = False):
        """
        指定した文字列 or cssセレクタ要素を返します。
        Args:
            selector_text (str): 表示を待ちたい文字列 or CSSセレクタ
            is_css (bool): TrueならCSSセレクタ、Falseなら文字列の表示を待つ
        """
        await self._set_tab()
        if is_css:
            element = await self.tab.wait_for(selector = selector_text)
        else:
            element = await self.tab.wait_for(text = selector_text)
        return element

    async def select_all(self, css_selector: str):
        '''
        特定のcssセレクタを持つ要素をすべて取得します。
        '''
        await self._set_tab()
        elements = await self.tab.select_all(css_selector)
        return elements

    async def click_element(self, selector_text: str, is_css: bool = False):
        """
        指定した要素の表示を待ってからクリックします。
        Args:
            selector_text (str): 表示を待ちたい文字列 or CSSセレクタ
            is_css (bool): TrueならCSSセレクタ、Falseなら文字列の表示を待つ
        """
        element = await self.select_element(selector_text, is_css)
        await element.click()


    async def send_keys_to_element(self, selector_text: str, is_css: bool = False, keys: str = None):
        """
        指定した要素の表示を待ってから文字列を送信します。
        Args:
            selector_text (str): 表示を待ちたい文字列 or CSSセレクタ
            is_css (bool): TrueならCSSセレクタ、Falseなら文字列の表示を待つ
            keys (str): 送信する文字列
        """
        element = await self.select_element(selector_text, is_css)
        await element.send_keys(keys)


    async def select_pulldown(self, css_selector: str):
        """
        プルダウンメニューのオプションを選択します。
        Args:
            css_selector (str): 選択したいプルダウンメニューのCSSセレクタ
        """
        await self._set_tab()
        await self.tab.evaluate(f'''
            var option = document.querySelector('{css_selector}');
            option.selected = true;
            var event = new Event('change', {{ bubbles: true }});
            option.parentElement.dispatchEvent(event);
        ''')
        await self.tab.wait(0.5)


    async def wait_for(self, selector: str, is_css: bool = False, timeout: int | float | None = 60):
        """
        指定した要素の表示を待ってからクリックします。
        Args:
            selector (str): 表示を待ちたい文字列またはcssセレクタ
            is_css (bool): Trueならcssセレクタ、Falseなら文字列
            timeout (int|float|None): タイムアウト秒数
        Returns:
            element: 要素
        """
        await self._set_tab()
        if is_css:
            element = await self.tab.wait_for(selector=selector, timeout=timeout)
        else:
            element = await self.tab.wait_for(text=selector, timeout=timeout)
        return element

    async def wait_and_click(self, selector: str, is_css: bool = False, timeout: int | float | None = 60):
        """
        指定した要素の表示を待ってからクリックします。
        Args:
            selector (str): 表示を待ちたい文字列またはcssセレクタ
            is_css (bool): Trueならcssセレクタ、Falseなら文字列
            timeout (int|float|None): タイムアウト秒数
        """
        element = await self.wait_for(selector, is_css, timeout)
        await element.click()

    async def query_selector(self, css_selector: str, is_all: bool = False) -> list:
        """
        指定したcssセレクタを持つ要素を、1つまたは複数検索します。
        Args:
            css_selector (str): 探し出すcssセレクタ
            is_all (bool): Trueならquery_selector_all()、Falseならquey_selector()を実行
        Returns:
            list: 検索結果を1つ又は複数格納したリスト
        """
        await self._set_tab()
        if is_all:
            find_list = await self.tab.query_selector_all(css_selector)
        else:
            find_list = await self.tab.query_selector(css_selector)
        return find_list

    async def get_html_content(self):
        '''
        現在のタブからhtmlを取得します。
        '''
        await self._set_tab()
        html_content = await self.tab.get_content()
        return html_content

    async def set_download_path(self, path: Path):
        '''
        ブラウザからのダウンロード先パスを指定します。
        Args:
            path (Path): ダウンロード先パス
        '''
        await self.tab.set_download_path(path)

    async def close_popup(self):
        '''
        意図せず表示された別タブをクリアします。
        '''
        self.browser = self.login_handler.session.browser
        for tab in self.browser.tabs:
            if self.tab != tab:
                await tab.close()