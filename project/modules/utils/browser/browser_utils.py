import os
import asyncio
import nodriver as uc

class BrowserUtils:
    """
    ブラウザ全体の基本操作（起動、再利用、リセットなど）を提供するクラス
    """
    BROWSER_PATH = os.getenv('BROWSER_PATH')
    USER_DATA_DIR = os.getenv('USER_DATA_DIR')
    PROFILE_DIR = os.getenv('PROFILE_DIR', 'Default')
    browser_instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def launch_browser(cls) -> uc.Browser:
        """
        ブラウザを起動し、ウインドウを最大化します。
        
        Returns:
            nodriver.Browser: ブラウザインスタンス
        """
        async with cls._lock:
            if cls.browser_instance is None:
                cls.browser_instance = await uc.start(browser_executable_path=cls.BROWSER_PATH, user_data_dir=cls.USER_DATA_DIR)
                # ブラウザ起動後、最初のタブでウインドウを最大化
                if cls.browser_instance.tabs:
                    await cls.browser_instance.tabs[0].set_window_state(state='maximized')
            return cls.browser_instance

    @classmethod
    async def clear_browser(cls):
        """
        ブラウザインスタンスを削除します。
        """
        if cls.browser_instance is not None:
            cls.browser_instance.stop()
            cls.browser_instance = None

    @classmethod
    async def reset_browser(cls) -> uc.Browser:
        """
        ブラウザを再起動し、ウインドウを最大化します。

        Returns:
            nodriver.Browser: ブラウザインスタンス
        """
        await cls.clear_browser()
        return await cls.launch_browser()
    


if __name__ == '__main__':
    async def main():
        bu = BrowserUtils()
        await bu.launch_browser()
        await bu.reset_browser()
        await asyncio.sleep(6240)
    
    asyncio.get_event_loop().run_until_complete(main())