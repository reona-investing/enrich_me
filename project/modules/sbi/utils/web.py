import nodriver as uc
import os


async def select_pulldown(tab: uc.core.tab.Tab, css_selector:str):
    # オプションをJavaScriptを使用してクリック
    await tab.evaluate(f'''
        var option = document.querySelector('{css_selector}');
        option.selected = true;
        var event = new Event('change', {{ bubbles: true }});
        option.parentElement.dispatchEvent(event);
    ''')
    await tab.wait(1)

async def wait_and_click(tab: uc.core.tab.Tab, selector_text: str, is_css=False):
    if is_css:
        button = await tab.select(selector_text)
    else:
        button = await tab.find(selector_text)
    await button.click()
    await tab.wait(1)

def get_downloaded_csv(download_folder: str, retry: int = 10, wait_func=None):
    import time
    for i in range(retry):
        deal_result_csv, _ = get_newest_two_csvs(download_folder)
        if deal_result_csv.endswith('.csv'):
            return deal_result_csv
        if wait_func:
            # awaitが使えない場合は実行側で処理
            wait_func(1)
        else:
            time.sleep(1)

def get_newest_two_csvs(directory: str) -> tuple[str, str]:
    # ディレクトリ内のすべてのファイルとサブディレクトリを取得
    files = os.listdir(directory)
    # ファイルがない場合はNoneを返す
    if not files:
        return None, None
    # ファイルのフルパスを取得
    full_paths = [os.path.join(directory, f) for f in files]
    # 保存日時がもっとも新しいファイルを見つける
    newest_file = max(full_paths, key=os.path.getmtime)
    if len(files) > 1:
        second_newest_file = sorted(full_paths, key=os.path.getmtime, reverse=True)[1]
    else:
        second_newest_file = None

    return newest_file, second_newest_file