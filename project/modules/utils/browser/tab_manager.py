from nodriver import Tab
from utils.browser.tab_utils import TabUtils



class TabManager:
    """
    タブのライフサイクル管理を行い、内部で TabUtils を保持するクラス
    """
    def __init__(self, tab: Tab):
        self._tab = tab
        self._utils = TabUtils(tab)

    @property
    def tab(self) -> Tab:
        """
        nodriverの Tab インスタンスを返す
        """
        return self._tab

    @property
    def utils(self) -> TabUtils:
        """
        タブ上の操作を提供する TabUtils インスタンスを返す
        """
        return self._utils

    async def close(self):
        """
        タブを閉じる
        """
        await self._tab.close()