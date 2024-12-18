import os
from .session_manager import BrowserSession

class LoginHandler:
    START_URL = "https://www.sbisec.co.jp/ETGate"

    def __init__(self):
        self.session = BrowserSession()

    async def sign_in(self):
        if self.session.tab is None:
            await self.session.start_browser(LoginHandler.START_URL)
            await self._input_credentials()
            await self.session.tab.wait(3)

    async def _input_credentials(self):
        username = await self.session.tab.wait_for('input[name="user_id"]')
        await username.send_keys(os.getenv('SBI_USERNAME'))
        
        password = await self.session.tab.wait_for('input[name="user_password"]')
        await password.send_keys(os.getenv('SBI_LOGINPASS'))

        login = await self.session.tab.wait_for('input[name="ACT_login"]')
        await login.click()

