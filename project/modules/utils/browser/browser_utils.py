import asyncio
import nodriver as uc

class BrowserUtils:
    """
    ブラウザ全体の基本操作（起動、再利用、リセットなど）を提供するクラス
    """
    BROWSER_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
    browser_instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def launch_browser(cls):
        async with cls._lock:
            if cls.browser_instance is None:
                cls.browser_instance = await uc.start(browser_executable_path=cls.BROWSER_PATH)
            return cls.browser_instance

    @classmethod
    async def reset_browser(cls):
        cls.browser_instance = None
        return await cls.launch_browser()
    
    @classmethod
    async def clear_browser(cls):
        cls.browser_instance = None



if __name__ == '__main__':
    async def main():
        bu = BrowserUtils()
        await bu.launch_browser()
        await bu.reset_browser()
        await asyncio.sleep(5)
    
    asyncio.get_event_loop().run_until_complete(main())