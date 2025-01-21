import nodriver as uc
from dotenv import load_dotenv

load_dotenv()

class BrowserSession:
    BROWSER_PATH = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'

    def __init__(self):
        self.browser = None
        self.tab = None

    async def start_browser(self, url: str):
        self.browser = await uc.start(browser_executable_path=BrowserSession.BROWSER_PATH)
        self.tab = await self.browser.get(url)
        await self.tab.wait(2)
        return self.tab