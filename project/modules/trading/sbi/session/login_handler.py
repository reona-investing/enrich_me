import os
from trading.sbi.session.session_manager import BrowserSession

class LoginHandler:
    START_URL = "https://www.sbisec.co.jp/ETGate"

    def __init__(self):
        self.session = BrowserSession()
        self.tab = None

    async def sign_in(self):
        if self.tab is None:
            await self.session.start_browser(LoginHandler.START_URL)
            await self._input_credentials()
            await self.session.tab.wait(3)
            self.tab = self.session.tab
        return self.tab

    async def _input_credentials(self):
        username = await self.session.tab.wait_for('input[name="user_id"]')
        await username.send_keys(os.getenv('SBI_USERNAME'))
        
        password = await self.session.tab.wait_for('input[name="user_password"]')
        await password.send_keys(os.getenv('SBI_LOGINPASS'))
        
        await self.session.tab.wait(1)
        login = await self.session.tab.wait_for('input[name="ACT_login"]')
        await login.click()

        await self.session.tab.wait_for(text='ログイン履歴', timeout=60)


if __name__ == '__main__':
    async def main():
        lh = LoginHandler()
        await lh.sign_in()
    
    import asyncio
    asyncio.run(main())