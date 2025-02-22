#%% モジュールのインポート
#パスを通す
#モジュールのインポート
import asyncio
from trading.sbi import LoginHandler
from trading.sbi.operations.order_manager import CancelManager
async def main():
    sbi_session = LoginHandler()
    cancel_manager = CancelManager(sbi_session)
    await cancel_manager.cancel_all_orders()

asyncio.get_event_loop().run_until_complete(main())