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
from sbi import SBISession, SBIOrderMaker
async def main():
    sbi_session = SBISession()
    sbi_order_maker = SBIOrderMaker(sbi_session)
    await sbi_order_maker.cancel_all_orders()

asyncio.get_event_loop().run_until_complete(main())