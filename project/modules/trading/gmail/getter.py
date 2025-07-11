from utils.browser import BrowserManager
import asyncio
import re
from bs4 import BeautifulSoup



async def auth_code_getter():
    browser_manager = BrowserManager()
    named_tab = await browser_manager.new_tab(name='gmail', url='https://mail.google.com/mail/u/0/')
    await named_tab.tab.utils.wait(3)
    await named_tab.tab.utils.click_element('tr[class="zA zE"]', is_css=True) # "zA zE"は未読メールに共通のクラス名
    await named_tab.tab.utils.wait(3)
    main_body_element = await named_tab.tab.utils.wait_for('div.a3s.aiL', is_css=True) # "a3s aiL "はメール本文に共通のクラス名
    html_str = await main_body_element.get_html()
    await browser_manager.close_tab('gmail')
    soup = BeautifulSoup(html_str, 'html.parser')
    text = soup.get_text(separator='\n')

    # 認証コードの直後にある5文字の英数字を検索（大文字英字＋数字）
    match = re.search(r'認証コード.*?\n*\s*([A-Za-z0-9]+)', text)

    if match:
        code = match.group(1)
        return code
    else:
        raise Exception("認証コードが見つかりませんでした。")



if __name__ == '__main__':
    code = asyncio.get_event_loop().run_until_complete(auth_code_getter())
    print(code)