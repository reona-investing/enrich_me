from dataclasses import dataclass
from utils.browser.tab_manager import TabManager

@dataclass
class NamedTab:
    """
    タブの名前とその管理オブジェクト（TabManager）を保持するデータクラス
    """
    name: str
    tab: "TabManager"  # TabManager のインスタンスを保持