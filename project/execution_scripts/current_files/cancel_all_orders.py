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
import sbi_operations
async def main():
    tab = await sbi_operations.sign_in()
    await sbi_operations.cancel_all_orders(tab)

asyncio.run(main())