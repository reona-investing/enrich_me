#%% モジュールのインポート
#パスを通す
#モジュールのインポート
from trading.sbi.operations.order_manager import CancelManager
from trading.sbi.browser.sbi_browser_manager import SBIBrowserManager
import asyncio

async def main():
    browser_manager = SBIBrowserManager()
    cancel_manager = CancelManager(browser_manager)
    await cancel_manager.cancel_all_orders()

asyncio.get_event_loop().run_until_complete(main())