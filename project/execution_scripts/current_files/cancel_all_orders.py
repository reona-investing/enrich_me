#%% モジュールのインポート
#パスを通す
#モジュールのインポート
from trading.sbi.orders.manager.order_executor import SBIOrderExecutor
from trading.sbi.browser.sbi_browser_manager import SBIBrowserManager
import asyncio

async def main():
    browser_manager = SBIBrowserManager()
    order_executor = SBIOrderExecutor(browser_manager)
    await order_executor.cancel_all_orders()

asyncio.get_event_loop().run_until_complete(main())