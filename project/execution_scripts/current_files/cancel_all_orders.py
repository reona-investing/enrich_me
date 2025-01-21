#%% モジュールのインポート
#パスを通す
#モジュールのインポート
import asyncio
from trading.sbi import OrderManager, LoginHandler
async def main():
    sbi_session = LoginHandler()
    order_manager = OrderManager(sbi_session)
    await order_manager.cancel_all_orders()

asyncio.get_event_loop().run_until_complete(main())