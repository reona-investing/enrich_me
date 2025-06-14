# trading/sbi/browser のクラス仕様書

## file_utils.py

### class FileUtils
- get_downloaded_csv: ダウンロードフォルダから最新のCSVファイルを取得
- get_newest_two_csvs: ディレクトリ内の最新2つのファイルを取得

## page_navigator.py

### class PageNavigator
- __init__: SBI証券のWebサイト上の各ページへの遷移を行います。
browser_manager (SBIBrowserManager): ブラウザ及びタブの管理・操作を司ります。

## sbi_browser_manager.py

### class SBIBrowserManager
- __init__: SBI証券向けのブラウザ操作を定義します。
Args:
    browser_manager (BrowserManager): ブラウザおよびタブの操作と管理を司ります。

