from utils.browser.named_tab import NamedTab
from utils.browser.tab_manager import TabManager
from utils.browser.browser_utils import BrowserUtils
from pathlib import Path
from functools import wraps
import asyncio

def retry_on_connection_error(func):
    """
    ConnectionRefusedError 発生時に reset_session_info を実行し、再試行するデコレータ。

    Args:
        func (Callable[..., Any]): デコレート対象の非同期関数。

    Returns:
        Callable[..., Any]: 再試行機能を備えた非同期関数。
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        for _ in range(2):  # 最大2回リトライ
            try:
                return await func(self, *args, **kwargs)
            except ConnectionRefusedError:
                print(f"ConnectionRefusedError 発生: {func.__name__} を再試行します。")
                await self.reset_session()
        raise ConnectionRefusedError(f"再試行に失敗しました: {func.__name__}")
    return wrapper


class BrowserManager:
    """
    ブラウザセッション全体を管理し、NamedTab を用いて複数タブを名前で管理するクラス
    """
    _lock = asyncio.Lock()

    def __init__(self):
        self.browser = None
        self.named_tabs: dict[str, NamedTab] = {} 

    async def reset_session(self):
        """
        ブラウザセッションをリセットする。
        すべてのタブを閉じ、ブラウザインスタンスをリセットする。
        """
        await BrowserUtils.clear_browser()
        self.browser = None

    @retry_on_connection_error
    async def new_tab(self, name: str, url: str = 'chrome://welcome'):
        """
        新規タブを作成し、指定した name で NamedTab として登録する。

        Args:
            name (str): タブの識別名。
            url (str, optional): タブの初期URL。デフォルトは 'chrome://welcome'。

        Returns:
            NamedTab: 作成された NamedTab インスタンス。
        """
        async with BrowserManager._lock:
            if not self.browser:
                self.browser = await BrowserUtils.launch_browser()
            new_tab_instance = await self.browser.get(url=url, new_tab=True)
            tab_manager = TabManager(new_tab_instance)
            named_tab = NamedTab(name=name, tab=tab_manager)
            self.named_tabs[name] = named_tab
            return named_tab

    def get_tab(self, name: str) -> NamedTab | None:
        """
        指定した名前のタブを取得する。

        Args:
            name (str): 取得したいタブの識別名。

        Returns:
            NamedTab | None: 取得した NamedTab インスタンス。存在しない場合は None。
        """
        return self.named_tabs.get(name)


    def rename_tab(self, name: str, new_name: str):
        """
        指定したタブの名前を変更する。

        Args:
            name (str): 変更前のタブ名。
            new_name (str): 変更後のタブ名。
        """
        if name in self.named_tabs:
            self.named_tabs[new_name] = self.named_tabs.pop(name)
        else:
            print(f"'{name}' という名前のタブはまだ存在しません。")


    async def close_tab(self, name: str):
        """
        指定した名前のタブを閉じ、管理から削除する。

        Args:
            name (str): 閉じるタブの識別名。
        """
        named_tab = self.named_tabs.pop(name, None)
        if named_tab:
            await named_tab.tab.close()

    async def close_all_tabs(self):
        """
        すべてのタブを閉じ、管理情報をクリアする
        """
        for named_tab in list(self.named_tabs.values()):
            await named_tab.tab.close()
        self.named_tabs.clear()

    async def close_popup(self):
        """
        意図せず開いた別タブを閉じる
        """
        if self.browser is not None:
            tabs_managed = [tab.tab.tab for tab in self.named_tabs.values()]
            tabs_unmanaged = [tab for tab in self.browser.tabs if tab not in tabs_managed]
            if tabs_unmanaged:
                await asyncio.gather(*(tab.close() for tab in tabs_unmanaged))


if __name__ == '__main__':
    import asyncio
    async def main():
        bm = BrowserManager()
        tab1 = await bm.new_tab('google', 'https://www.google.co.jp/')
        await asyncio.sleep(1)
        tab2 = await bm.new_tab('blank')
        tab3 = await bm.new_tab('yahoo', 'https://www.google.co.jp/')
        await asyncio.sleep(1)
        print('BrowserManagerクラスで管理中のタブ')
        print([tab.tab._tab for tab in bm.named_tabs.values()])

    
    asyncio.get_event_loop().run_until_complete(main())
