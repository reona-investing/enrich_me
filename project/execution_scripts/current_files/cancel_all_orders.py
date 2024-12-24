#%% モジュールのインポート
#パスを通す
if __name__ == '__main__':
    from pathlib import Path
    import sys
    PROJECT_FOLDER = str(Path(__file__).parents[2])
    ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
    sys.path.append(ORIGINAL_MODULES)
#モジュールのインポート
import asyncio
from sbi import OrderManager, LoginHandler
async def main():
    sbi_session = LoginHandler()
    order_manager = OrderManager(sbi_session)
    await order_manager.cancel_all_orders()

asyncio.get_event_loop().run_until_complete(main())