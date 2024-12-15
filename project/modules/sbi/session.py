import os
import nodriver as uc
from dotenv import load_dotenv
from .utils.decorators import retry

load_dotenv()

class SBISession:
    BROWSER_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
    START_URL = "https://www.sbisec.co.jp/ETGate"

    def __init__(self):
        self.browser = None
        self.tab = None

    @retry()
    async def sign_in(self):
        if self.tab is None:
            self.browser = await uc.start(browser_executable_path=SBISession.BROWSER_PATH)
            self.tab = await self.browser.get(SBISession.START_URL)
            await self.tab.wait(2)
            await self._input_credentials()
            await self.tab.wait(3)

    async def _input_credentials(self):
        username = await self.tab.wait_for('input[name="user_id"]')
        await username.send_keys(os.getenv('SBI_USERNAME'))
        
        password = await self.tab.wait_for('input[name="user_password"]')
        await password.send_keys(os.getenv('SBI_LOGINPASS'))

        login = await self.tab.wait_for('input[name="ACT_login"]')
        await login.click()