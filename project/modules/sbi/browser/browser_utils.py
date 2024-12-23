import nodriver as uc

class BrowserUtils:
    @staticmethod
    async def select_pulldown(tab: uc.core.tab.Tab, css_selector: str):
        """プルダウンメニューのオプションを選択"""
        await tab.evaluate(f'''
            var option = document.querySelector('{css_selector}');
            option.selected = true;
            var event = new Event('change', {{ bubbles: true }});
            option.parentElement.dispatchEvent(event);
        ''')
        await tab.wait(0.5)

    @staticmethod
    async def wait_and_click(tab: uc.core.tab.Tab, selector_text: str, is_css: bool = False):
        """指定した要素を待ってからクリック"""
        if is_css:
            element = await tab.select(selector_text)
        else:
            element = await tab.find(selector_text)
        await element.click()
        await tab.wait(1)
