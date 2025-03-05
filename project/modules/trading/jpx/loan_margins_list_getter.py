import pandas as pd
from utils.browser import BrowserUtils
import asyncio
import pandas as pd


class LoanMarginsListGetter:
    def __init__(self):
        pass
    
    async def get(self):
        HOME_PAGE = 'https://www.jpx.co.jp'
        LINKED_PAGE = f'{HOME_PAGE}/listing/others/margin/index.html'
        browser_utils = BrowserUtils()
        await browser_utils.open_url(LINKED_PAGE)
        file_brock = await browser_utils.select_element('div[class*=component-file]', is_css=True)
        file_element = await file_brock.query_selector('a[href]')
        FILE_PATH = f'{HOME_PAGE}{file_element.attrs["href"]}'
        return pd.read_excel(FILE_PATH, header=1)


if __name__ == '__main__':
    
    async def main():
        getter = LoanMarginsListGetter()
        await getter.get()
        return getter.loan_margins_list

    loan_margins_list = asyncio.get_event_loop().run_until_complete(main())
    print(loan_margins_list.dtypes)