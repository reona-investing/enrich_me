from nodriver import Element, Tab
from pathlib import Path
from typing import Any
import asyncio
import logging
from functools import wraps


def retry_on_node_error(max_retries: int = 3, delay: float = 0.5):
    """Node document エラー専用の再試行デコレータ"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_message = str(e).lower()
                    
                    if "node with given id does not belong to the document" in error_message:
                        last_exception = e
                        if attempt < max_retries:
                            logging.warning(f"Node document error on attempt {attempt + 1}. Retrying in {delay}s...")
                            await asyncio.sleep(delay)
                            continue
                    else:
                        # その他のエラーは即座に再発生
                        raise e
            
            # 最大再試行回数に達した場合
            raise last_exception
        
        return wrapper
    return decorator


class TabUtils:
    """
    nodriver の Tab オブジェクトに対する操作（URLオープン、リロード、クリック等）をラップするユーティリティクラス
    Node document エラー対応版
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
            await asyncio.sleep(t)

    @retry_on_node_error(max_retries=3, delay=0.5)
    async def wait_for(self, selector: str, is_css: bool = False, timeout: int = 60) -> Element:
        """
        指定した要素の表示を待ちます（Node document エラー対応版）

        Args:
            selector (str): 表示を待ちたい文字列またはcssセレクタ
            is_css (bool): Trueならcssセレクタ、Falseなら文字列
            timeout (int): タイムアウト秒数

        Returns:
            element: 要素

        Raises:
            ElementNotFoundError: 指定した要素がタイムアウトまでに見つからなかった場合
        """
        for i in range(timeout):
            try:
                if is_css:
                    element = await self._tab.wait_for(selector=selector, timeout=0.5)
                else:
                    element = await self._tab.wait_for(text=selector, timeout=0.5)
                return element
            except Exception as e:
                error_message = str(e).lower()
                if "-32000" in error_message:
                    # Node document エラーの場合は上位のデコレータに処理を委ねる
                    await self.wait(0.5)

        raise Exception(f"要素が見つかりませんでした: selector={selector}, is_css={is_css}")

    @retry_on_node_error(max_retries=3, delay=0.5)
    async def click_element(self, selector_text: str, is_css: bool = False, timeout: int = 60):
        """
        指定した要素の表示を待ってからクリックします（Node document エラー対応版）

        Args:
            selector_text (str): 表示を待ちたい文字列 or CSSセレクタ
            is_css (bool): TrueならCSSセレクタ、Falseなら文字列の表示を待つ
            timeout (int|float|None): タイムアウト秒数
        """
        element = await self.wait_for(selector=selector_text, is_css=is_css, timeout=timeout)
        await asyncio.sleep(0.3)
        await element.click()

    @retry_on_node_error(max_retries=3, delay=0.5)
    async def send_keys_to_element(self, selector_text: str, keys: str, is_css: bool = False):
        """
        指定した要素の表示を待ってから文字列を送信します（Node document エラー対応版）

        Args:
            selector_text (str): 表示を待ちたい文字列 or CSSセレクタ
            is_css (bool): TrueならCSSセレクタ、Falseなら文字列の表示を待つ
            keys (str): 送信する文字列
        """
        element = await self.wait_for(selector=selector_text, is_css=is_css)
        await element.send_keys(keys)

    @retry_on_node_error(max_retries=3, delay=0.5)
    async def select_all(self, css_selector: str) -> list:
        '''
        特定のcssセレクタを持つ要素をすべて取得します（Node document エラー対応版）

        Args:
            css_selector (str): 取得対象のCSSセレクタ

        Returns:
            list: 指定したcssセレクタを持つ全ての要素を格納したリスト
        '''
        await self.wait_for(css_selector, is_css=True)
        await self.wait(2)
        return await self._tab.select_all(css_selector)

    @retry_on_node_error(max_retries=3, delay=0.5)
    async def select_pulldown(self, css_selector: str):
        """
        プルダウンメニューのオプションを選択します（Node document エラー対応版）

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

    @retry_on_node_error(max_retries=3, delay=0.5)
    async def query_selector(self, css_selector: str, is_all: bool = False) -> list | Any | None:
        """
        指定したcssセレクタを持つ要素を、1つまたは複数検索します（Node document エラー対応版）

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

    @retry_on_node_error(max_retries=3, delay=0.5)
    async def get_html_content(self) -> str:
        '''
        表示されたページのhtmlを取得します（Node document エラー対応版）

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

    async def force_refresh_and_retry(self, operation_func, *args, **kwargs):
        """
        強制的にページをリフレッシュしてから操作を再実行
        Node document エラーが頻発する場合の最終手段
        """
        try:
            return await operation_func(*args, **kwargs)
        except Exception as e:
            if "node with given id does not belong to the document" in str(e).lower():
                logging.info("Persistent node document error. Force refreshing page...")
                await self.reload()
                await self.wait(3)
                return await operation_func(*args, **kwargs)
            else:
                raise e


if __name__ == '__main__':
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    import asyncio

    async def main():
        from utils.browser.browser_manager import BrowserManager
        bm = BrowserManager()
        tab = await bm.new_tab('default')
        tu = TabUtils(tab.tab._tab)
        await tu.open_url('https://www.google.co.jp/')
        
        try:
            # Node document エラーに対応したクリック操作のテスト
            await tu.click_element('検索', is_css=False)
        except Exception as e:
            logging.error(f"Error occurred: {e}")
        
        await asyncio.sleep(5)
    
    asyncio.get_event_loop().run_until_complete(main())